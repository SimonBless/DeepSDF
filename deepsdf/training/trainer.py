"""Trainer for DeepSDF model."""

from typing import Optional, Dict, Any
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepsdf.models.decoder import DeepSDFDecoder
from deepsdf.models.loss import DeepSDFLoss
from deepsdf.utils.config import Config


class LatentCodeManager:
    """
    Manages latent codes for shapes during training.

    Each shape is assigned a learnable latent code that is optimized
    during training along with the decoder weights.
    """

    def __init__(
        self,
        num_shapes: int,
        latent_size: int,
        device: str = "cuda",
    ) -> None:
        self.num_shapes = num_shapes
        self.latent_size = latent_size
        self.device = device

        # Initialize latent codes with normal distribution
        self.latent_codes = nn.Embedding(num_shapes, latent_size)
        nn.init.normal_(self.latent_codes.weight.data, 0.0, 1.0 / latent_size)
        self.latent_codes = self.latent_codes.to(device)

    def get_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Get latent codes for given shape indices."""
        codes: torch.Tensor = self.latent_codes(indices)
        return codes

    def parameters(self):  # type: ignore
        """Return parameters for optimization."""
        return self.latent_codes.parameters()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return self.latent_codes.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self.latent_codes.load_state_dict(state_dict)


class Trainer:
    """
    Trainer for DeepSDF model.

    Args:
        model: DeepSDF decoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to use for training
    """

    def __init__(
        self,
        model: DeepSDFDecoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Config] = None,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Use default config if not provided
        if config is None:
            config = Config()
        self.config = config

        # Initialize latent codes
        num_shapes = len(train_loader.dataset)  # type: ignore
        self.latent_manager = LatentCodeManager(
            num_shapes=num_shapes,
            latent_size=config.model.latent_size,
            device=device,
        )

        # Loss function
        self.criterion = DeepSDFLoss(
            latent_reg_lambda=config.training.latent_reg_lambda,
            clamp_dist=config.training.clamp_dist,
        )

        # Optimizers
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
        )

        self.latent_optimizer = optim.Adam(
            self.latent_manager.parameters(),
            lr=config.training.latent_lr,
        )

        # Tensorboard
        log_dir = os.path.join(config.output_dir, config.experiment_name, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Checkpoints
        self.checkpoint_dir = os.path.join(config.output_dir, config.experiment_name, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total_loss": 0.0, "sdf_loss": 0.0, "latent_reg": 0.0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            points = batch["points"].to(self.device)
            sdf_gt = batch["sdf"].to(self.device)

            # Get latent codes for this batch
            indices = torch.arange(
                batch_idx * self.config.training.batch_size,
                min(
                    (batch_idx + 1) * self.config.training.batch_size,
                    len(self.train_loader.dataset),  # type: ignore
                ),
            ).to(self.device)

            latent_codes = self.latent_manager.get_codes(indices)

            # Forward pass
            sdf_pred = self.model(latent_codes, points)

            # Compute loss
            loss, loss_dict = self.criterion(sdf_pred, sdf_gt, latent_codes)

            # Backward pass
            self.optimizer.zero_grad()
            self.latent_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.latent_optimizer.step()

            # Update statistics
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1

            # Logging
            if self.global_step % self.config.training.log_every == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)

                pbar.set_postfix(loss_dict)

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {"total_loss": 0.0, "sdf_loss": 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                points = batch["points"].to(self.device)
                sdf_gt = batch["sdf"].to(self.device)

                # Get latent codes
                indices = torch.arange(
                    batch_idx * self.config.training.batch_size,
                    min(
                        (batch_idx + 1) * self.config.training.batch_size,
                        len(self.val_loader.dataset),  # type: ignore
                    ),
                ).to(self.device)

                latent_codes = self.latent_manager.get_codes(indices)

                # Forward pass
                sdf_pred = self.model(latent_codes, points)

                # Compute loss
                _, loss_dict = self.criterion(sdf_pred, sdf_gt, latent_codes)

                # Update statistics
                for key in val_losses:
                    if key in loss_dict:
                        val_losses[key] += loss_dict[key]
                num_batches += 1

        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        # Log to tensorboard
        for key, value in val_losses.items():
            self.writer.add_scalar(f"val/{key}", value, self.current_epoch)

        return val_losses

    def save_checkpoint(self, filename: Optional[str] = None) -> None:
        """Save a checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pth"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "latent_codes_state_dict": self.latent_manager.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "latent_optimizer_state_dict": self.latent_optimizer.state_dict(),
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.latent_manager.load_state_dict(checkpoint["latent_codes_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.latent_optimizer.load_state_dict(checkpoint["latent_optimizer_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def train(self, num_epochs: Optional[int] = None) -> None:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train. If None, uses config.
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()
            print(f'Epoch {epoch}: Train Loss = {train_losses["total_loss"]:.6f}')

            # Validate
            if self.val_loader is not None:
                val_losses = self.validate()
                print(f'Epoch {epoch}: Val Loss = {val_losses["total_loss"]:.6f}')

            # Save checkpoint
            if epoch % self.config.training.checkpoint_every == 0:
                self.save_checkpoint()

            # Save latest checkpoint
            self.save_checkpoint("latest.pth")

        print("Training completed!")
        self.writer.close()
