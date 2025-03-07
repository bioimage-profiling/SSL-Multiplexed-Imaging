# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
from pathlib import Path
import numpy as np

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.loss import VICRegLLoss

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead

from vicregl_custom_transform import VICRegLTransform

import wandb
from lightning.pytorch.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint

import chunked_h5_dataset

class VICRegL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.momentum = args.momentum
        
        resnet = torchvision.models.resnet50()
        resnet.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = BarlowTwinsProjectionHead(2048, 2048*4, 2048*4)
        self.local_projection_head = VicRegLLocalProjectionHead(2048, 2048//4, 2048//4)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = VICRegLLoss()
        
        self.save_hyperparameters()

    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z, z_local

    def training_step(self, batch, batch_index):
        views_and_grids = batch[0]
        views = views_and_grids[: len(views_and_grids) // 2]
        grids = views_and_grids[len(views_and_grids) // 2 :]
        features = [self.forward(view) for view in views]
        loss = self.criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        )
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return optim


def main(args):
    # seed torch and numpy
    torch.manual_seed(0)
    np.random.seed(0)
    pl.seed_everything(args.seed)
    
    mean_std, args.n_channels = chunked_h5_dataset.get_mean_std(args)

    transform = VICRegLTransform(
        global_gaussian_blur_kernel_size = 5,
        local_gaussian_blur_kernel_size = 5,
        n_local_views=0,
        normalize=mean_std)
    
    # create a lightly dataset for training with augmentations
    base = chunked_h5_dataset.h5_chunk_wrapper(Path(args.data_path))
    dataset = LightlyDataset.from_torch_dataset(base, transform=transform)
    print('Loaded dataset with length:', dataset.__len__())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    print("Logged in to wandb: ", wandb.login(key=''))

    wandb_logger = WandbLogger(project=args.wandb_project_name,
                               name=args.wandb_name,
                               log_model=False,
                               save_dir=args.output_dir
                               )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch}',
        every_n_epochs=10,
        save_last=True,
        save_top_k = -1
    )
    
    model = VICRegL(args)
    # Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
    # calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        num_nodes=args.n_nodes,
        devices=args.n_devices, 
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
        logger=wandb_logger,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model=model, train_dataloaders=dataloader)
    
    
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for your script')
    
    parser.add_argument('--n_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--n_devices', type=int, default=8, help='Number of devices')
    
    parser.add_argument('--wandb_project_name', type=str, default='')
    parser.add_argument('--wandb_name', type=str, default='')
    
    parser.add_argument('--num_workers', type=int, default=7, help='Number of workers for data loading (default: 7)')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generation (default: 1)')
    
    parser.add_argument('--input_size', type=int, default=224, help='Input size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 128)')
    
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=6e-2, help='Learning rate (default: 6e-2)')
    parser.add_argument('--momentum', type=float, default=0.75, help='Momentum (default: 0.9)')
    
    parser.add_argument('--dataset', type=str, default="", help='Name of the dataset')
    parser.add_argument('--data_path', type=str, default="", help='Path to the data directory (default: data/)')
    parser.add_argument('--output_dir', type=str, default="", help='Path to the data directory (default: data/)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)