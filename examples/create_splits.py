"""Example script for creating train/val/test splits from existing preprocessed SDF data."""

import argparse
import numpy as np
from pathlib import Path
import json


def main() -> None:
    """Main split creation function."""
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from existing SDF data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed SDF data (folders with .npz files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Output directory for split files (defaults to data-dir)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training samples (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of validation samples (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Set output directory
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .npz files
    npz_files = list(data_dir.glob("**/*.npz"))
    
    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_files)} .npz files")

    # Extract sample names (folder names or file stems)
    # Check if files are in subdirectories
    sample_names = []
    for npz_file in npz_files:
        # Get the parent folder name if file is in a subdirectory
        if npz_file.parent != data_dir:
            # Use the folder name as the sample identifier
            sample_name = npz_file.parent.name
        else:
            # Use the file stem if files are directly in data_dir
            sample_name = npz_file.stem
        
        if sample_name not in sample_names:
            sample_names.append(sample_name)

    print(f"Found {len(sample_names)} unique samples")

    # Validate ratios
    if args.train_ratio + args.val_ratio > 1.0:
        print(
            f"Error: train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) "
            f"must be <= 1.0"
        )
        return

    # Shuffle samples
    np.random.shuffle(sample_names)

    # Calculate split sizes
    num_samples = len(sample_names)
    num_train = int(num_samples * args.train_ratio)
    num_val = int(num_samples * args.val_ratio)

    # Create splits
    train_samples = sample_names[:num_train]
    val_samples = sample_names[num_train : num_train + num_val]
    test_samples = sample_names[num_train + num_val :]

    # Save splits as JSON files
    train_file = output_dir / "train.json"
    val_file = output_dir / "val.json"
    test_file = output_dir / "test.json"

    with open(train_file, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"Saved {len(train_samples)} training samples to {train_file}")

    with open(val_file, "w") as f:
        json.dump(val_samples, f, indent=2)
    print(f"Saved {len(val_samples)} validation samples to {val_file}")

    with open(test_file, "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"Saved {len(test_samples)} test samples to {test_file}")

    # Print summary
    print("\nSplit Summary:")
    print(f"  Total samples: {num_samples}")
    print(f"  Train: {len(train_samples)} ({len(train_samples)/num_samples*100:.1f}%)")
    print(f"  Val:   {len(val_samples)} ({len(val_samples)/num_samples*100:.1f}%)")
    print(f"  Test:  {len(test_samples)} ({len(test_samples)/num_samples*100:.1f}%)")
    print(f"\nSplit files saved to: {output_dir}")


if __name__ == "__main__":
    main()
