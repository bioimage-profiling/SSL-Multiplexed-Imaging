from pathlib import Path
import numpy as np

from tqdm.auto import tqdm

from sklearn.preprocessing import normalize

import torch
import torchvision

from lightly.data import LightlyDataset

import embedding_dataset

from main import MAE

import argparse

def main(args):
    # create a lightly dataset for training with augmentations
    base = embedding_dataset.embedding_dataset(args)
    mean_std = base.get_mean_std()
    
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.Normalize(mean=mean_std['mean'],
                                         std=mean_std['std']
                                         ),
        ])
    
    # create a lightly dataset for training with augmentations
    dataset = LightlyDataset.from_torch_dataset(base, transform=test_transform)
    print('Loaded dataset with length:', dataset.__len__())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    
    model = MAE.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, fnames, _ in tqdm(dataloader):
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb.cpu())
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    
    np.save(args.save_dir / f'embeddings_{Path(args.checkpoint_path).stem}.npy', embeddings)
    np.save(args.save_dir / f'names_{Path(args.checkpoint_path).stem}.npy', filenames)
    
    print('ALL DONE!')
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for your script')
    
    parser.add_argument('--num_workers', type=int, default=7, help='Number of workers for data loading (default: 7)')
    
    parser.add_argument('--input_size', type=int, default=16, help='Input size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    
    parser.add_argument('--path_embeddings', type=Path, default="")
    parser.add_argument('--path_names', type=Path, default="")
    
    parser.add_argument('--checkpoint_path', type=Path, default="")
    parser.add_argument('--save_dir', type=Path, default="")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    main(args)