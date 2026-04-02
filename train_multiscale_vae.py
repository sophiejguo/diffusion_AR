"""
attempt at writing custom training script for MultiScaleAE, disregard this file
"""


import os
import time
import argparse
import gc
import sys
from functools import partial
import yaml
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from diffvar_models.multiscale_vae import MultiScaleVAE
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.ema import LitEma
import dist
import wandb


torch.set_float32_matmul_precision('medium')

class MultiScaleVAELightning(pl.LightningModule):
    def __init__(
        self,
        z_channels=32,
        ch=128,
        dropout=0.0,
        conv_ks=3,
        residual_ratio=0.5,
        share_resi_ratio=4,
        beta=0.25,
        learning_rate=1e-4,
        lr_g_factor=1.0,
        betas=(0.5, 0.9),
        use_ema=False,
        image_key="image",
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_g_factor = lr_g_factor
        self.betas = betas
        self.image_key = image_key
        
        # Initialize the MultiScaleVAE model
        self.vae = MultiScaleVAE(
            z_channels=z_channels,
            ch=ch,
            dropout=dropout,
            beta=beta,
            conv_ks=conv_ks,
            share_resi_ratio=share_resi_ratio,
            resi_ratio=residual_ratio,
            v_patch_nums=v_patch_nums,
            test_mode=False,
        )
        
        # EMA support
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.vae)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def encode(self, x):
        """Encode images to latent space"""
        f = self.vae.pre_conv(self.vae.encoder(x))
        return f, None  # No quantization loss in this model

    def decode(self, f):
        """Decode latent representation back to image"""
        f_hat, _ = self.vae.multiscale(f)
        return self.vae.decoder(self.vae.post_conv(f_hat))

    def forward(self, input):
        """Full forward pass"""
        f = self.vae.pre_conv(self.vae.encoder(input))
        f_hat, mean_loss = self.vae.multiscale(f)
        reconstruction = self.vae.decoder(self.vae.post_conv(f_hat))
        return reconstruction, mean_loss

    def get_input(self, batch, k):
        """Extract input from batch"""
        x = batch[k] if isinstance(batch, dict) else batch[0]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        if torch.isnan(x).any():
            print("NaN detected in input batch!")
        return x

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        """Log images for visualization in TensorBoard"""
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        
        xrec, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        
        if self.use_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                log["reconstructions_ema"] = xrec_ema
                
        return log

    @contextmanager
    def ema_scope(self):
        """Context manager for EMA activation"""
        if self.use_ema:
            self.model_ema.store(self.vae.parameters())
            self.model_ema.copy_to(self.vae)
            yield
            self.model_ema.restore(self.vae.parameters())
        else:
            yield

    def on_train_batch_end(self, *args, **kwargs):
        """Update EMA parameters after each training batch"""
        if self.use_ema:
            self.model_ema(self.vae)

    def training_step(self, batch, batch_idx):
        """Single training step"""
        x = self.get_input(batch, self.image_key)
        reconstruction, mse_loss = self(x)
        if torch.isnan(reconstruction).any():
            print("NaN detected in reconstruction!")
        rec_loss = F.mse_loss(reconstruction, x)
        loss = rec_loss + mse_loss

        # Log to wandb
        wandb.log({
            "train/rec_loss": rec_loss.item(),
            "train/mse_loss": mse_loss.item(),
            "train/total_loss": loss.item(),
            "step": self.global_step
        })

        # Log metrics
        self.log("train/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/mse_loss", mse_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        x = self.get_input(batch, self.image_key)
        reconstruction, mse_loss = self(x)
        rec_loss = F.mse_loss(reconstruction, x)
        loss = rec_loss + mse_loss

        # Log to wandb
        wandb.log({
            "val/rec_loss": rec_loss.item(),
            "val/mse_loss": mse_loss.item(),
            "val/total_loss": loss.item(),
            "val_step": self.global_step
        })
        
        # Log metrics
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mse_loss", mse_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Optionally log sample images every few batches
        if batch_idx % 100 == 0:
            self.logger.experiment.add_images(
                "val/reconstruction",
                torch.cat([x[:4], reconstruction[:4]], dim=3),
                self.global_step
            )
        
        return loss

    def configure_optimizers(self):
        """Configure optimizers for training"""
        lr_g = self.lr_g_factor * self.learning_rate
        print(f"Using learning rate: {lr_g}")
        
        optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=lr_g,
            betas=self.betas
        )
        
        return optimizer


# Exponential Moving Average for model weights
# class LitEma:
#     def __init__(self, model, decay=0.9999):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#         self.register()

#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()

#     def __call__(self, model, decay=None):
#         if decay is None:
#             decay = self.decay
#         with torch.no_grad():
#             for name, param in model.named_parameters():
#                 if param.requires_grad:
#                     self.shadow[name].sub_((1 - decay) * (self.shadow[name] - param.data))

#     def store(self, parameters):
#         for name, param in parameters:
#             if param.requires_grad:
#                 self.backup[name] = param.data.clone()
#                 param.data.copy_(self.shadow[name])

#     def restore(self, parameters):
#         for name, param in parameters:
#             if param.requires_grad:
#                 param.data.copy_(self.backup[name])
#         self.backup = {}

#     def copy_to(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 param.data.copy_(self.shadow[name])

#     def buffers(self):
#         return self.shadow.values()




def parse_args():
    parser = argparse.ArgumentParser()
    # Model settings
    parser.add_argument('--z_channels', type=int, default=32, help='Latent channels')
    parser.add_argument('--ch', type=int, default=128, help='Base channel width')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--conv_ks', type=int, default=3, help='Kernel size for convolutional layers')
    parser.add_argument('--steps_K', type=int, default=10, help='Number of steps in the MultiScaleVAE')
    parser.add_argument('--resolutions', type=str, default=None, help='Resolutions to use in the MultiScaleVAE')
    parser.add_argument('--residual_ratio', type=float, default=0.5, help='Residual mixing ratio')
    parser.add_argument('--share_resi_ratio', type=int, default=4, help='Sharing strategy for residual transformations')
    parser.add_argument('--beta', type=float, default=0.25, help='Weighting for commitment loss')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_g_factor', type=float, default=1.0, help='Learning rate factor')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA for model weights')
    
    # Data settings
    parser.add_argument('--data_path', type=str, default='path/to/dataset', help='Dataset path')
    parser.add_argument('--data_load_reso', type=int, default=256, help='Resolution of loaded data')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--hflip', type=bool, default=True, help='Use horizontal flip for data augmentation')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Checkpoint interval in epochs')
    
    # Distributed training settings
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    # New config file argument
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Get arguments from command line
    args = parse_args()
    
    # If config file is provided, override args with config values
    if args.config:
        config = load_config(args.config)
        
        # Override model settings
        args.z_channels = config['model'].get('z_channels', args.z_channels)
        args.ch = config['model'].get('ch', args.ch)
        args.dropout = config['model'].get('dropout', args.dropout)
        args.beta = config['model'].get('beta', args.beta)
        args.residual_ratio = config['model'].get('residual_ratio', args.residual_ratio)
        args.share_resi_ratio = config['model'].get('share_resi_ratio', args.share_resi_ratio)
        
        # Override training settings
        args.batch_size = config['training'].get('batch_size', args.batch_size)
        args.epochs = config['training'].get('epochs', args.epochs)
        args.lr = config['training'].get('lr', args.lr)
        args.lr_g_factor = config['training'].get('lr_g_factor', args.lr_g_factor)
        args.use_ema = config['training'].get('use_ema', args.use_ema)
        
        # Override data settings
        args.data_path = config['data'].get('data_path', args.data_path)
        args.data_load_reso = config['data'].get('data_load_reso', args.data_load_reso)
        args.workers = config['data'].get('workers', args.workers)
        args.hflip = config['data'].get('hflip', args.hflip)
        
        # Override output settings
        args.output_dir = config['output'].get('output_dir', args.output_dir)
        args.log_dir = config['output'].get('log_dir', args.log_dir)
        args.checkpoint_interval = config['output'].get('checkpoint_interval', args.checkpoint_interval)
        
        # Override distributed settings
        args.distributed = config['distributed'].get('enabled', args.distributed)
        args.local_rank = config['distributed'].get('local_rank', args.local_rank)
    
    pl.seed_everything(42)
    
    # Initialize wandb
    wandb.init(project="diffusion-multiscale-vae", name=f"run-{time.strftime('%Y%m%d-%H%M%S')}")
    
    # Initialize distributed
    if args.distributed:
        dist.initialize()
        device = dist.get_device()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build dataset and dataloader
    print(f"Building dataset from {args.data_path}")
    num_classes, dataset_train, dataset_val = build_dataset(
        args.data_path, 
        final_reso=args.data_load_reso, 
        hflip=args.hflip
    )
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers // 2,
        pin_memory=True
    )
    
    # Create model
    model = MultiScaleVAELightning(
        z_channels=args.z_channels,
        ch=args.ch,
        dropout=args.dropout,
        beta=args.beta,
        residual_ratio=args.residual_ratio,
        share_resi_ratio=args.share_resi_ratio,
        learning_rate=args.lr,
        lr_g_factor=args.lr_g_factor,
        use_ema=args.use_ema
    )
    
    # Setup logging and checkpointing
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="multiscale_vae"
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="multiscale_vae-{epoch:02d}-{val/total_loss:.4f}",
        save_top_k=3,
        monitor="val/total_loss",
        mode="min",
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if "cuda" in str(device) else 'cpu',
        devices=dist.get_world_size() if args.distributed else 1,
        strategy='ddp_find_unused_parameters_true' if args.distributed else None,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,  # Use mixed precision for faster training
        log_every_n_steps=50,
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Final save
    final_path = os.path.join(args.output_dir, "multiscale_vae_final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Model saved to {final_path}")
    
    # Cleanup
    dist.finalize()
    
    # Save to wandb
    wandb.save(final_path)
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in training: {e}")
        if dist.initialized():
            dist.finalize()
        raise