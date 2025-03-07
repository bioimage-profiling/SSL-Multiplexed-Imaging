# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn

from lightly.data import LightlyDataset
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
# from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule

import wandb
from lightning.pytorch.loggers import WandbLogger

import chunked_h5_dataset
from dino_custom_transform import DINOTransform

from pytorch_lightning.callbacks import ModelCheckpoint

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, args, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(args.n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DINO(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.momentum = args.momentum
        self.hidden_dim = 2048
        self.bottleneck_dim = 256
        self.output_dim = 65536
        self.warmup_epoch = 10
        
        self.save_hyperparameters()
        
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        backbone.patch_embed = PatchEmbed(args)
        input_dim = backbone.embed_dim

        self.student_backbone = backbone
        # self.student_head = DINOProjectionHead(
        #     input_dim, 512, 64, 2048, freeze_last_layer=1
        # )
        self.student_head = DINOProjectionHead(
            input_dim, self.hidden_dim, self.bottleneck_dim, self.output_dim, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        # self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        self.teacher_head = DINOProjectionHead(input_dim, self.hidden_dim, self.bottleneck_dim, self.output_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=self.output_dim, warmup_teacher_temp_epochs=self.warmup_epoch)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, self.momentum, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('momentum', momentum, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

def main(args):
    
    mean_std, args.n_channels = chunked_h5_dataset.get_mean_std(args)
    
    transform = DINOTransform(
        global_crop_size = 224,
        global_crop_scale = (0.4, 1.0),
        local_crop_size = 96,
        local_crop_scale = (0.05, 0.4),
        n_local_views = 8,
        hf_prob = 0.5,
        vf_prob = 0,
        rr_prob = 0,
        rr_degrees = None,
        random_gray_scale = 0.2,
        gaussian_blur = (1.0, 0.1, 0.5),
        kernel_size = 5,
        sigmas = (0.1, 2),
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
                               save_dir=args.output_dir,
                               log_model=False,
                               )

    model = DINO(args)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch}',
        every_n_epochs=10,
        save_last=True,
        save_top_k = -1
    )
    
    # Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
    # calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         num_nodes=args.n_nodes,
                         devices=args.n_devices, 
                         accelerator="gpu", 
                         strategy="ddp",
                         sync_batchnorm=True,
                         use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
                         logger=wandb_logger,
                         default_root_dir=args.output_dir,
                         callbacks=[checkpoint_callback],
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
    
    parser.add_argument('--input_size', type=int, default=256, help='Input size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=6e-2, help='Learning rate (default: 6e-2)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
    
    parser.add_argument('--dataset', type=str, default="", help='Name of the dataset')
    parser.add_argument('--data_path', type=str, default="", help='Path to the data directory (default: data/)')
    parser.add_argument('--output_dir', type=str, default="", help='Path to the data directory (default: data/)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)