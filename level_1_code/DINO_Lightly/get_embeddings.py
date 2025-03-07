from pathlib import Path
import numpy as np

from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

import torch
import torchvision

from lightly.data import LightlyDataset

import chunked_h5_dataset

from main import DINO

import argparse

def get_embeddings(args, weightp):
    mean_std, args.n_channels = chunked_h5_dataset.get_mean_std(args)

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.Normalize(mean=mean_std['mean'],
                                         std=mean_std['std']
                                         ),
        ])
    
    # create a lightly dataset for training with augmentations
    base = chunked_h5_dataset.h5_chunk_wrapper(Path(args.data_path))
    dataset = LightlyDataset.from_torch_dataset(base, transform=test_transform)
    print('Loaded dataset with length:', dataset.__len__())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    
    model = DINO.load_from_checkpoint(weightp)
    model.eval()
    
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, fnames, _ in tqdm(dataloader):
            img = img.to(model.device)
            emb = model.student_backbone(img).flatten(start_dim=1).cpu()
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    
    np.save(args.save_dir / f'embeddings_{weightp.stem}.npy', embeddings)
    np.save(args.save_dir / f'names_{weightp.stem}.npy', filenames)
    
    print('ALL DONE!')

    
def main(args):
    weight_list = list(args.save_dir.glob('*epoch*.ckpt'))
    print(weight_list)
    
    for weightp in weight_list:
        if (args.save_dir / f'embeddings_{weightp.stem}.npy').exists():
            continue
        args.save_dir.mkdir(parents=True, exist_ok=True)
        get_embeddings(args, weightp)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for your script')
    
    parser.add_argument('--num_workers', type=int, default=7, help='Number of workers for data loading (default: 7)')
    
    parser.add_argument('--input_size', type=int, default=256, help='Input size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    
    parser.add_argument('--dataset', type=str, default='', help='Name of the dataset')
    parser.add_argument('--data_path', type=str, default="")
    
    parser.add_argument('--checkpoint_path', type=str, default="", help='Path to the checkpoint')
    parser.add_argument('--save_dir', type=Path, default="", help='Path to the checkpoint')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)