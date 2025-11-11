"""Example script for reconstructing shapes from trained DeepSDF model."""

import argparse
import torch
from pathlib import Path

from deepsdf import DeepSDFDecoder, ShapeReconstructor
from deepsdf.utils import Config


def main() -> None:
    """Main reconstruction function."""
    parser = argparse.ArgumentParser(description="Reconstruct shapes using DeepSDF")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reconstructions",
        help="Path to output directory",
    )
    parser.add_argument(
        "--latent-code",
        type=str,
        default=None,
        help="Path to latent code file (.npy or .pt)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Grid resolution for marching cubes",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=0,
        help="Number of random latent codes to generate and reconstruct",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Load config from checkpoint
    config_dict = checkpoint.get("config", {})
    config = Config.from_dict(config_dict)

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

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create reconstructor
    reconstructor = ShapeReconstructor(model, device=device)

    # Reconstruct from provided latent code
    if args.latent_code is not None:
        print(f"Loading latent code from {args.latent_code}")

        if args.latent_code.endswith(".npy"):
            import numpy as np

            latent_code = torch.from_numpy(np.load(args.latent_code)).float()
        elif args.latent_code.endswith(".pt"):
            latent_code = torch.load(args.latent_code)
        else:
            raise ValueError("Latent code must be .npy or .pt file")

        print(f"Reconstructing mesh (resolution={args.resolution})...")
        mesh = reconstructor.reconstruct_mesh(
            latent_code,
            resolution=args.resolution,
        )

        output_path = output_dir / "reconstruction.obj"
        reconstructor.save_mesh(mesh, str(output_path))
        print(f"Saved reconstruction to {output_path}")

    # Generate and reconstruct random latent codes
    if args.num_random > 0:
        print(f"Generating {args.num_random} random shapes...")

        for i in range(args.num_random):
            # Sample random latent code from standard normal
            latent_code = torch.randn(config.model.latent_size)

            print(f"Reconstructing shape {i+1}/{args.num_random}...")
            mesh = reconstructor.reconstruct_mesh(
                latent_code,
                resolution=args.resolution,
            )

            output_path = output_dir / f"random_shape_{i:03d}.obj"
            reconstructor.save_mesh(mesh, str(output_path))

        print(f"Saved {args.num_random} random reconstructions to {output_dir}")


if __name__ == "__main__":
    main()
