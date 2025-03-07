import torch, torchvision
import torch.nn as nn

from lightly.data import LightlyDataset

import numpy as np

from skimage.transform import resize, rescale
from skimage.morphology import binary_closing, disk

import chunked_h5_dataset

from torch.utils.data import Dataset

def get_dapi_mask(dapi_image):
    small_dapi = rescale(dapi_image, 0.1)
    smm = (small_dapi > small_dapi.mean()).astype(np.uint8)
    smm = binary_closing(smm, footprint=disk(10))
    dapi_mask = resize(smm, dapi_image.shape, anti_aliasing=False)
    return dapi_mask

def prepare(input: torch.Tensor, backbone) -> torch.Tensor:
    input = input + backbone.encoder.interpolate_pos_encoding(input)
    return input

def get_one_attn(image, backbone):
    x = backbone.conv_proj(image)
    tokens = backbone.images_to_tokens(image, prepend_class_token=True)
    x = prepare(tokens, backbone)

    for i, layer in enumerate(backbone.encoder.layers):
        if i < len(backbone.encoder.layers) - 1:
            x = layer(x)
        else:
            x = layer.ln_1(x)
            o, attn = layer.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
    
    attn = attn[:, :, 0, 1:].view(attn.shape[0],attn.shape[1],14,14)
    attn = nn.functional.interpolate(attn, scale_factor=16, mode="nearest") # 224 / 14 = 16
    attn = attn.squeeze().cpu().detach()
    if attn.ndim == 3:
        attn = attn.unsqueeze(0)
    attn = attn.numpy()
    
    return attn

class ImagePatchDataset(Dataset):
    def __init__(self, image, patch_size=(224, 224), overlap=50):
        self.image = np.log1p(image)
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Calculate the step size for overlapping
        self.step_size = patch_size[0] - overlap
        
        # Calculate the number of patches along each dimension
        self.num_patches_x = (image.shape[1] - patch_size[0]) // self.step_size + 1
        self.num_patches_y = (image.shape[2] - patch_size[1]) // self.step_size + 1

    def __len__(self):
        return self.num_patches_x * self.num_patches_y

    def __getitem__(self, idx):
        patch_idx_x = idx // self.num_patches_y
        patch_idx_y = idx % self.num_patches_y
        
        start_x = patch_idx_x * self.step_size
        start_y = patch_idx_y * self.step_size
        end_x = start_x + self.patch_size[0]
        end_y = start_y + self.patch_size[1]
        
        patch = self.image[:, start_x:end_x, start_y:end_y]
        return patch, (start_x, start_y)
    
def get_attention(image, backbone, args):
    
    patch_size= (args.input_size, args.input_size)
    
    # Create dataset from one roi image
    
    mean_std, args.n_channels = chunked_h5_dataset.get_mean_std(args)

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.Normalize(mean=mean_std['mean'],
                                        std=mean_std['std']
                                        ),
        ])

    base = ImagePatchDataset(image, patch_size=patch_size, overlap=args.overlap)
    dataset = LightlyDataset.from_torch_dataset(base, transform=test_transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        drop_last=False,
    )
    
    ## Get attentions of each patch and resize to image size
    
    attentions = []
    xs = []
    ys = []
    for bimages, (x, y), _ in dataloader:
        attn = get_one_attn(bimages.cuda(), backbone)
        attentions.append(attn)
        xs.append(x.numpy())
        ys.append(y.numpy())

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    
    attentions = np.concatenate(attentions, axis=0)
    
    attentions_fullsize = attentions

    # Patch the patches back
    
    reconstructed_image = np.zeros((attentions_fullsize.shape[1], image.shape[1], image.shape[2]))
    pixel_count = np.zeros((reconstructed_image.shape[1], reconstructed_image.shape[2]))

    for (start_x, start_y, attn) in zip(xs, ys, attentions_fullsize):
        end_x = start_x + patch_size[0]
        end_y = start_y + patch_size[1]
            
        reconstructed_image[:, start_x:end_x, start_y:end_y] += attn
        pixel_count[start_x:end_x, start_y:end_y] += 1
        
    pixel_count[pixel_count==0] = 1
    reconstructed_image /= pixel_count
    
    # Make sure to exclude non-spot area
    dapi_mask = get_dapi_mask(image[0])
    reconstructed_image = reconstructed_image * dapi_mask
    
    return reconstructed_image


    