"""Tests for DeepSDF decoder model."""

import pytest
import torch
from deepsdf.models.decoder import DeepSDFDecoder


def test_decoder_initialization() -> None:
    """Test decoder initialization."""
    model = DeepSDFDecoder(latent_size=256)
    assert model.latent_size == 256
    assert len(model.layers) > 0


def test_decoder_forward() -> None:
    """Test forward pass."""
    batch_size = 4
    num_points = 1000
    latent_size = 256
    
    model = DeepSDFDecoder(latent_size=latent_size)
    
    latent_vector = torch.randn(batch_size, latent_size)
    xyz = torch.randn(batch_size, num_points, 3)
    
    output = model(latent_vector, xyz)
    
    assert output.shape == (batch_size, num_points, 1)
    assert not torch.isnan(output).any()


def test_decoder_inference() -> None:
    """Test inference mode."""
    latent_size = 256
    num_points = 500
    
    model = DeepSDFDecoder(latent_size=latent_size)
    
    # Single latent vector
    latent_vector = torch.randn(latent_size)
    xyz = torch.randn(num_points, 3)
    
    output = model.inference(latent_vector, xyz)
    
    assert output.shape == (1, num_points, 1)
    assert not torch.isnan(output).any()


def test_decoder_custom_config() -> None:
    """Test decoder with custom configuration."""
    model = DeepSDFDecoder(
        latent_size=128,
        hidden_dims=[256, 256, 256, 256],
        dropout_prob=0.1,
        weight_norm=False,
    )
    
    assert model.latent_size == 128
    assert len(model.layers) == 5  # 4 hidden + 1 output


def test_decoder_output_range() -> None:
    """Test that output is bounded by tanh."""
    model = DeepSDFDecoder(latent_size=256)
    
    latent_vector = torch.randn(2, 256)
    xyz = torch.randn(2, 100, 3) * 10  # Large input values
    
    output = model(latent_vector, xyz)
    
    # Tanh output should be in [-1, 1]
    assert output.min() >= -1.0
    assert output.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
