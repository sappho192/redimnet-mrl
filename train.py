"""
Training script for MRL-enabled ReDimNet speaker recognition.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/latest.pt
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables (including WANDB_API_KEY)
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import ReDimNetMRL
from losses import MatryoshkaLoss, AAMSoftmax, SubCenterAAMSoftmax, TripletLoss
from dataset import VoxCelebDataset, create_dataloader
from pretrained import create_mrl_from_pretrained, unfreeze_backbone


class Trainer:
    """
    Trainer for MRL speaker recognition.
    """
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device(self.config['hardware']['device'])

        # Set random seed
        self._set_seed(self.config['seed'])

        # Build model
        self.model = self._build_model()

        # Build loss
        self.criterion = self._build_loss()

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # Build dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Setup logging
        self.writer, self.use_wandb = self._setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_model(self):
        """Build MRL model with optional pretrained weights."""
        model_cfg = self.config['model']
        adv_cfg = self.config['advanced']

        # Option 1: Load from pretrained torch.hub (recommended)
        if adv_cfg.get('use_pretrained', False):
            print(f"\n{'='*60}")
            print("Loading pretrained model from torch.hub")
            print(f"{'='*60}")

            freeze_backbone = adv_cfg.get('freeze_backbone_epochs', 0) > 0

            model = create_mrl_from_pretrained(
                model_name=adv_cfg.get('model_name', 'b2'),
                train_type=adv_cfg.get('train_type', 'ptn'),
                dataset=adv_cfg.get('pretrained_dataset', 'vox2'),
                embed_dim=model_cfg['embed_dim'],
                mrl_dims=model_cfg['mrl_dims'],
                device=self.device,
                freeze_backbone=freeze_backbone
            )

            if freeze_backbone:
                self.freeze_epochs = adv_cfg['freeze_backbone_epochs']
                print(f"\n🔒 Backbone frozen for first {self.freeze_epochs} epochs")
            else:
                self.freeze_epochs = 0

        # Option 2: Train from scratch
        else:
            print("\n⚠️ Training from scratch (no pretrained weights)")
            model = ReDimNetMRL(
                embed_dim=model_cfg['embed_dim'],
                mrl_dims=model_cfg['mrl_dims'],
                F=model_cfg['F'],
                C=model_cfg['C'],
                block_1d_type=model_cfg['block_1d_type'],
                block_2d_type=model_cfg['block_2d_type'],
                pooling_func=model_cfg['pooling_func'],
                out_channels=model_cfg['out_channels'],
                feat_type=model_cfg['feat_type'],
                feat_agg_dropout=self.config['training']['feat_agg_dropout'],
            )
            model = model.to(self.device)
            self.freeze_epochs = 0

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")

        return model

    def _build_loss(self):
        """Build loss function."""
        loss_cfg = self.config['loss']
        model_cfg = self.config['model']

        # Get number of classes from dataset
        train_dataset = VoxCelebDataset(
            data_dir=self.config['data']['train_dataset'],
            chunk_duration=self.config['data']['chunk_duration'],
            augmentation=False,
        )
        num_classes = len(train_dataset.speaker_to_idx)
        print(f"Number of training speakers: {num_classes}")

        # Build base loss
        if loss_cfg['type'] == 'AAMSoftmax':
            base_loss = AAMSoftmax(
                embed_dim=model_cfg['embed_dim'],
                num_classes=num_classes,
                margin=loss_cfg['margin'],
                scale=loss_cfg['scale'],
                easy_margin=loss_cfg['easy_margin'],
            )
        elif loss_cfg['type'] == 'SubCenter':
            base_loss = SubCenterAAMSoftmax(
                embed_dim=model_cfg['embed_dim'],
                num_classes=num_classes,
                num_subcenters=loss_cfg['num_subcenters'],
                margin=loss_cfg['margin'],
                scale=loss_cfg['scale'],
            )
        elif loss_cfg['type'] == 'Triplet':
            base_loss = TripletLoss(
                margin=loss_cfg['margin'],
                mining_strategy='hard',
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_cfg['type']}")

        base_loss = base_loss.to(self.device)

        # Wrap with MatryoshkaLoss
        criterion = MatryoshkaLoss(
            base_loss=base_loss,
            mrl_dims=model_cfg['mrl_dims'],
            mrl_weights=model_cfg['mrl_weights'],
        )

        return criterion

    def _build_optimizer(self):
        """Build optimizer."""
        opt_cfg = self.config['training']

        if opt_cfg['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg['learning_rate'],
                weight_decay=opt_cfg['weight_decay'],
                betas=opt_cfg['betas'],
            )
        elif opt_cfg['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg['learning_rate'],
                weight_decay=opt_cfg['weight_decay'],
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg['optimizer']}")

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        sched_cfg = self.config['training']

        if sched_cfg['lr_scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg['num_epochs'],
                eta_min=sched_cfg['min_lr'],
            )
        elif sched_cfg['lr_scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif sched_cfg['lr_scheduler'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95,
            )
        else:
            scheduler = None

        return scheduler

    def _build_dataloaders(self):
        """Build training and validation dataloaders."""
        data_cfg = self.config['data']

        train_loader = create_dataloader(
            data_dir=data_cfg['train_dataset'],
            batch_size=self.config['training']['batch_size'],
            num_workers=data_cfg['num_workers'],
            chunk_duration=data_cfg['chunk_duration'],
            shuffle=True,
            augmentation=data_cfg['augmentation'],
            pin_memory=data_cfg['pin_memory'],
        )

        val_loader = create_dataloader(
            data_dir=data_cfg['val_dataset'],
            batch_size=self.config['training']['batch_size'],
            num_workers=data_cfg['num_workers'],
            chunk_duration=data_cfg['chunk_duration'],
            shuffle=False,
            augmentation=False,
            pin_memory=data_cfg['pin_memory'],
        )

        return train_loader, val_loader

    def _setup_logging(self):
        """Setup logging (TensorBoard, wandb)."""
        log_cfg = self.config['logging']

        # Create directories
        Path(log_cfg['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(log_cfg['save_dir']).mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if log_cfg['tensorboard']:
            writer = SummaryWriter(log_dir=log_cfg['log_dir'])
        else:
            writer = None

        # Weights & Biases
        use_wandb = False
        if log_cfg['wandb']:
            try:
                import wandb

                # Initialize wandb
                wandb.init(
                    project=log_cfg['wandb_project'],
                    config=self.config,
                    name=log_cfg.get('wandb_run_name', None),
                    tags=log_cfg.get('wandb_tags', []),
                )

                # Watch model for gradients (optional)
                if log_cfg.get('wandb_watch_model', False):
                    wandb.watch(self.model, log='all', log_freq=100)

                use_wandb = True
                print(f"✅ Weights & Biases initialized")
                print(f"   Project: {log_cfg['wandb_project']}")
                print(f"   Dashboard: {wandb.run.url}")

            except ImportError:
                print("⚠️ wandb not installed, skipping W&B logging")
                print("   Install with: pip install wandb")

        return writer, use_wandb

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_losses_dict = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")

        for batch_idx, (waveforms, labels) in enumerate(pbar):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            embeddings_dict = self.model(waveforms, return_all_dims=True)

            # Compute loss
            loss, loss_dict = self.criterion(embeddings_dict, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config['training']['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )

            # Optimizer step
            if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            epoch_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in epoch_losses_dict:
                    epoch_losses_dict[key] = 0
                epoch_losses_dict[key] += val

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Step-level logging
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)

            self.global_step += 1

        # Epoch-level logging
        avg_loss = epoch_loss / len(self.train_loader)
        avg_losses = {k: v / len(self.train_loader) for k, v in epoch_losses_dict.items()}

        if self.writer:
            self.writer.add_scalar('train/loss_epoch', avg_loss, self.epoch)
            for key, val in avg_losses.items():
                self.writer.add_scalar(f'train/{key}', val, self.epoch)

        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log({
                'train/loss': avg_loss,
                'train/lr': self.optimizer.param_groups[0]['lr'],
                'epoch': self.epoch,
                **{f'train/{k}': v for k, v in avg_losses.items()}
            }, step=self.epoch)

        return avg_loss, avg_losses

    @torch.no_grad()
    def validate(self):
        """Validate model."""
        self.model.eval()
        val_loss = 0
        val_losses_dict = {}

        for waveforms, labels in tqdm(self.val_loader, desc="Validation"):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            embeddings_dict = self.model(waveforms, return_all_dims=True)

            # Compute loss
            loss, loss_dict = self.criterion(embeddings_dict, labels)

            val_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in val_losses_dict:
                    val_losses_dict[key] = 0
                val_losses_dict[key] += val

        # Average
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_losses = {k: v / len(self.val_loader) for k, v in val_losses_dict.items()}

        # Logging
        if self.writer:
            self.writer.add_scalar('val/loss', avg_val_loss, self.epoch)
            for key, val in avg_val_losses.items():
                self.writer.add_scalar(f'val/{key}', val, self.epoch)

        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log({
                'val/loss': avg_val_loss,
                **{f'val/{k}': v for k, v in avg_val_losses.items()}
            }, step=self.epoch)

        return avg_val_loss, avg_val_losses

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        save_dir = Path(self.config['logging']['save_dir'])

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        # Save latest
        torch.save(checkpoint, save_dir / 'latest.pt')

        # Save epoch checkpoint
        if self.epoch % self.config['logging']['save_interval'] == 0:
            torch.save(checkpoint, save_dir / f'epoch_{self.epoch}.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"✅ Saved best model (val_loss: {self.best_val_loss:.4f})")

            # Log best model to wandb
            if self.use_wandb:
                import wandb
                wandb.run.summary["best_val_loss"] = self.best_val_loss
                wandb.run.summary["best_epoch"] = self.epoch

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']

        print(f"\n{'='*50}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*50}\n")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Check if we need to unfreeze backbone (Stage 1 -> Stage 2)
            if hasattr(self, 'freeze_epochs') and self.freeze_epochs > 0:
                if epoch == self.freeze_epochs:
                    print(f"\n{'='*60}")
                    print(f"Epoch {epoch}: Unfreezing backbone (Stage 1 → Stage 2)")
                    print(f"{'='*60}\n")
                    unfreeze_backbone(self.model)

                    # Recreate optimizer with all parameters
                    self.optimizer = self._build_optimizer()
                    self.scheduler = self._build_scheduler()

            # Train
            train_loss, train_losses = self.train_epoch()

            # Validate
            if epoch % self.config['logging']['val_interval'] == 0:
                val_loss, val_losses = self.validate()

                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Save checkpoint
                self.save_checkpoint(is_best=is_best)

                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Best Val Loss: {self.best_val_loss:.4f}")
                for dim in self.config['model']['mrl_dims']:
                    print(f"    Loss {dim}D: {val_losses.get(f'loss_{dim}d', 0):.4f}")

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Train MRL-ReDimNet speaker recognition')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Create trainer
    trainer = Trainer(args.config)

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {trainer.epoch}")

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
