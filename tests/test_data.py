"""Tests for data utilities."""

import pytest
import numpy as np

from deepsdf.data.dataset import SDFSamplesDataset


def test_sdf_samples_dataset() -> None:
    """Test in-memory SDF samples dataset."""
    # Create dummy samples
    samples = []
    for i in range(10):
        samples.append(
            {
                "points": np.random.randn(1000, 3).astype(np.float32),
                "sdf": np.random.randn(1000, 1).astype(np.float32),
            }
        )

    dataset = SDFSamplesDataset(samples, num_samples_per_shape=500)

    assert len(dataset) == 10

    sample = dataset[0]
    assert "points" in sample
    assert "sdf" in sample
    assert sample["points"].shape == (500, 3)
    assert sample["sdf"].shape == (500, 1)


def test_dataset_subsampling() -> None:
    """Test that dataset properly subsamples points."""
    samples = [
        {
            "points": np.random.randn(10000, 3).astype(np.float32),
            "sdf": np.random.randn(10000, 1).astype(np.float32),
        }
    ]

    dataset = SDFSamplesDataset(samples, num_samples_per_shape=1000)

    sample = dataset[0]
    assert sample["points"].shape[0] == 1000
    assert sample["sdf"].shape[0] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
