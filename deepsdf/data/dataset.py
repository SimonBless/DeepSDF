"""Dataset and DataLoader for DeepSDF."""

from typing import Optional, Callable, List, Dict
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SDFDataset(Dataset):
    """
    Dataset for SDF samples.

    This dataset loads pre-computed SDF samples from disk.
    Expected format: Each sample is a .npz file containing 'points' and 'sdf' arrays.

    Args:
        data_dir: Directory containing .npz sample files
        split: Dataset split ('train', 'val', or 'test')
        num_samples_per_shape: Number of points to sample per shape (default: 10000)
        transform: Optional transform to apply to samples
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_samples_per_shape: int = 10000,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_samples_per_shape = num_samples_per_shape
        self.transform = transform

        # Load dataset split
        split_file = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                self.sample_files = json.load(f)
        else:
            # If no split file, use all .npz files in the directory
            self.sample_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

        if len(self.sample_files) == 0:
            raise ValueError(f"No samples found in {data_dir} for split {split}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'points': 3D coordinates of shape (num_samples, 3)
                - 'sdf': SDF values of shape (num_samples, 1)
                - 'shape_id': Shape identifier
        """
        sample_file = self.sample_files[idx]

        # Handle both direct .npz files and subdirectory structure
        if sample_file.endswith(".npz"):
            sample_path = os.path.join(self.data_dir, sample_file)
        else:
            # Check if it's a subdirectory with sdf.npz inside
            subdir_path = os.path.join(self.data_dir, sample_file, "sdf.npz")
            direct_path = os.path.join(self.data_dir, sample_file + ".npz")
            
            if os.path.exists(subdir_path):
                sample_path = subdir_path
            elif os.path.exists(direct_path):
                sample_path = direct_path
            else:
                raise FileNotFoundError(
                    f"Could not find SDF data for sample {sample_file}. "
                    f"Tried: {subdir_path} and {direct_path}"
                )

        # Load sample
        data = np.load(sample_path)
        
        # Handle different data formats
        if "points" in data and "sdf" in data:
            # Standard format: separate points and sdf arrays
            points = data["points"]
            sdf = data["sdf"]
        elif "pos" in data and "neg" in data:
            # DeepSDF format: pos/neg samples with [x, y, z, sdf] format
            pos_samples = data["pos"]  # Shape: (N, 4) where last column is SDF
            neg_samples = data["neg"]  # Shape: (M, 4) where last column is SDF
            
            # Combine positive and negative samples
            all_samples = np.vstack([pos_samples, neg_samples])
            points = all_samples[:, :3]  # First 3 columns are xyz
            sdf = all_samples[:, 3:4]    # Last column is SDF value
        else:
            raise ValueError(
                f"Unknown data format in {sample_path}. "
                f"Expected 'points'/'sdf' or 'pos'/'neg' keys, got: {list(data.keys())}"
            )

        # Subsample if needed
        if len(points) > self.num_samples_per_shape:
            indices = np.random.choice(len(points), self.num_samples_per_shape, replace=False)
            points = points[indices]
            sdf = sdf[indices]
        elif len(points) < self.num_samples_per_shape:
            # Pad with repeated samples if not enough
            indices = np.random.choice(len(points), self.num_samples_per_shape, replace=True)
            points = points[indices]
            sdf = sdf[indices]

        # Convert to tensors
        sample = {
            "points": torch.from_numpy(points).float(),
            "sdf": torch.from_numpy(sdf).float(),
            "shape_id": sample_file.replace(".npz", ""),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SDFSamplesDataset(Dataset):
    """
    In-memory dataset for SDF samples.

    This dataset holds all samples in memory for faster access.

    Args:
        samples: List of dictionaries, each containing 'points' and 'sdf'
        num_samples_per_shape: Number of points to sample per shape
    """

    def __init__(
        self,
        samples: List[Dict[str, np.ndarray]],
        num_samples_per_shape: int = 10000,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.num_samples_per_shape = num_samples_per_shape

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        sample = self.samples[idx]
        points = sample["points"]
        sdf = sample["sdf"]

        # Subsample
        if len(points) > self.num_samples_per_shape:
            indices = np.random.choice(len(points), self.num_samples_per_shape, replace=False)
            points = points[indices]
            sdf = sdf[indices]

        return {
            "points": torch.from_numpy(points).float(),
            "sdf": torch.from_numpy(sdf).float(),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
