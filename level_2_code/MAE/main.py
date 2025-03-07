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
from lightly.models import utils
from lightly.models.modules import masked_autoencoder
# from lightly.transforms.mae_transform import MAETransform

import wandb
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint

import embedding_dataset
from mae_custom_transform import MAETransform

import argparse

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, args, img_size=224, patch_size=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # self.proj = nn.Conv1d(768, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(args.n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, HW = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.lr = args.lr

        decoder_dim = 1024
        vit = torchvision.models.vit_b_16(pretrained=False)
        vit.conv_proj = nn.Conv2d(in_channels=args.n_channels, 
                                   out_channels=768, 
                                   kernel_size=1,
                                   stride=1
            )
        vit.encoder.pos_embedding = nn.Parameter(torch.zeros_like(vit.encoder.pos_embedding)) # Remove positional encoding as there is no meaning in arbitrary feature images
        vit.patch_size = 1
        
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        
        self.mask_ratio = args.mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size**2 * args.n_channels,
            dropout=0.2,
            attention_dropout=0.2,
        )
        # self.decoder.patch_embed = PatchEmbed(args)
        self.criterion = nn.MSELoss()
        
        self.save_hyperparameters()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        views = batch[0]
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        return optim

def main(args):
    
    args.wandb_name = args.wandb_name + '_gradacc'
    
    # seed torch and numpy
    torch.manual_seed(0)
    np.random.seed(0)
    pl.seed_everything(args.seed)
    
    # create a lightly dataset for training with augmentations
    base = embedding_dataset.embedding_dataset(args)
    mean_std = base.get_mean_std()

    transform = MAETransform(
        input_size=args.input_size,
        normalize=mean_std
    )
    
    # create a lightly dataset for training with augmentations
    dataset = LightlyDataset.from_torch_dataset(base, transform=transform)
    print('Loaded dataset with length:', dataset.__len__())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    
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
    
    model = MAE(args)
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator="gpu", 
                         logger=wandb_logger,
                         default_root_dir=args.output_dir,
                         callbacks=[checkpoint_callback],
                         accumulate_grad_batches=2
                         )
    trainer.fit(model, dataloader)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for your script')
    
    parser.add_argument('--n_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--n_devices', type=int, default=8, help='Number of devices')
    
    parser.add_argument('--wandb_project_name', type=str, default='')
    parser.add_argument('--wandb_name', type=str, default='')
    
    parser.add_argument('--num_workers', type=int, default=7, help='Number of workers for data loading (default: 7)')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generation (default: 1)')
    
    parser.add_argument('--input_size', type=int, default=16, help='Input size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=300, help='Batch size for training (default: 128)')
    
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate (default: 6e-2)')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Momentum (default: 0.9)')
    
    parser.add_argument('--output_dir', type=str, default="", help='Path to the data directory (default: data/)')
    
    parser.add_argument('--path_embeddings', type=Path, default="")
    parser.add_argument('--path_names', type=Path, default="")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
