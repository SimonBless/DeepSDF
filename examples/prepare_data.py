"""Example script for preparing SDF data from meshes."""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from deepsdf.data import create_sdf_samples


def main() -> None:
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare SDF data from meshes")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing mesh files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for SDF samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of samples per mesh",
    )
    parser.add_argument(
        "--variance",
        type=float,
        default=0.05,
        help="Variance for near-surface sampling",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training samples",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of validation samples",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all mesh files
    input_dir = Path(args.input_dir)
    mesh_extensions = [".obj", ".off", ".ply", ".stl"]
    mesh_files = []
    for ext in mesh_extensions:
        mesh_files.extend(list(input_dir.glob(f"**/*{ext}")))

    print(f"Found {len(mesh_files)} mesh files")

    if len(mesh_files) == 0:
        print("No mesh files found!")
        return

    # Process each mesh
    all_sample_names = []
    for i, mesh_path in enumerate(tqdm(mesh_files, desc="Processing meshes")):
        try:
            # Generate SDF samples
            points, sdf = create_sdf_samples(
                str(mesh_path),
                num_samples=args.num_samples,
                variance=args.variance,
                normalize=True,
            )

            # Save as .npz
            sample_name = f"{mesh_path.stem}_{i:06d}"
            output_path = output_dir / f"{sample_name}.npz"
            np.savez(str(output_path), points=points, sdf=sdf)

            all_sample_names.append(sample_name)

        except Exception as e:
            print(f"Failed to process {mesh_path}: {e}")
            continue

    print(f"Processed {len(all_sample_names)} meshes successfully")

    # Create train/val/test splits
    np.random.shuffle(all_sample_names)

    num_train = int(len(all_sample_names) * args.train_ratio)
    num_val = int(len(all_sample_names) * args.val_ratio)

    train_samples = all_sample_names[:num_train]
    val_samples = all_sample_names[num_train : num_train + num_val]
    test_samples = all_sample_names[num_train + num_val :]

    # Save splits
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_samples, f, indent=2)

    with open(output_dir / "val.json", "w") as f:
        json.dump(val_samples, f, indent=2)

    with open(output_dir / "test.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    print(
        f"Created splits: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )
    print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    main()
