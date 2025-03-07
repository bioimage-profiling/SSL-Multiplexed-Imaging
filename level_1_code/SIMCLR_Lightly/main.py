from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from lightly.data import LightlyDataset
# from lightly.transforms import SimCLRTransform, utils
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

import wandb
from lightning.pytorch.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint

import chunked_h5_dataset
from simclr_custom_transform import SimCLRTransform

class SimCLRModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.max_epochs = args.max_epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet50()
        resnet.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 1024)

        self.criterion = NTXentLoss()
        
        self.save_hyperparameters()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('momentum', self.momentum, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('weight_decay', self.weight_decay, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

def main(args):
    # seed torch and numpy
    torch.manual_seed(0)
    np.random.seed(0)
    pl.seed_everything(args.seed)
    
    mean_std, args.n_channels = chunked_h5_dataset.get_mean_std(args)

    # define the augmentations for self-supervised learning
    transform = SimCLRTransform(
        input_size=args.input_size,
        # require invariance to flips and rotations
        hf_prob=0.5,
        vf_prob=0.5,
        rr_prob=0.5,
        min_scale=0.5,
        normalize=mean_std
    )

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

    model = SimCLRModel(args)
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         num_nodes=args.n_nodes,
                         devices=args.n_devices, 
                         accelerator="gpu", 
                         strategy="ddp",
                         logger=wandb_logger,
                         default_root_dir=args.output_dir,
                         callbacks=[checkpoint_callback]
                         )
    trainer.fit(model, dataloader)

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for your script')
    
    parser.add_argument('--n_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--n_devices', type=int, default=8, help='Number of devices')
    
    parser.add_argument('--wandb_project_name', type=str, default='')
    parser.add_argument('--wandb_name', type=str, default='')
    
    parser.add_argument('--num_workers', type=int, default=7, help='Number of workers for data loading (default: 7)')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generation (default: 1)')
    
    parser.add_argument('--input_size', type=int, default=256, help='Input size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=6e-2, help='Learning rate (default: 6e-2)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    
    parser.add_argument('--dataset', type=str, default="", help='Name of the dataset')
    parser.add_argument('--data_path', type=str, default="", help='Path to the data directory (default: data/)')
    parser.add_argument('--output_dir', type=str, default="", help='Path to the data directory (default: data/)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
