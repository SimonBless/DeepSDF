"""Example script for training DeepSDF model."""

import argparse
import torch
from pathlib import Path

from deepsdf import DeepSDFDecoder, SDFDataset, Trainer
from deepsdf.utils import Config
from deepsdf.data import create_dataloader


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DeepSDF model")
    parser.add_argument(
        "--config",
        type=str,
        default="deepsdf/configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)
    config.data.data_dir = args.data_dir
    config.output_dir = args.output_dir

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_save_path = Path(config.output_dir) / config.experiment_name / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(config_save_path))

    # Set device
    device = config.training.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    model = DeepSDFDecoder(
        latent_size=config.model.latent_size,
        hidden_dims=config.model.hidden_dims,
        dropout_prob=config.model.dropout_prob,
        norm_layers=config.model.norm_layers,
        latent_in=config.model.latent_in,
        weight_norm=config.model.weight_norm,
    )

    # Load datasets
    print("Loading datasets...")
    train_dataset = SDFDataset(
        data_dir=config.data.data_dir,
        split=config.data.train_split,
        num_samples_per_shape=config.training.num_samples_per_shape,
    )

    val_dataset = None
    val_loader = None
    try:
        val_dataset = SDFDataset(
            data_dir=config.data.data_dir,
            split=config.data.val_split,
            num_samples_per_shape=config.training.num_samples_per_shape,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )
    except (ValueError, FileNotFoundError):
        print("Validation dataset not found, training without validation")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )

    print(f"Training dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation dataset size: {len(val_dataset)}")

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
