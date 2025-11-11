"""Tests for loss functions."""

import pytest
import torch
from deepsdf.models.loss import DeepSDFLoss, SDFLoss


def test_sdf_loss() -> None:
    """Test basic SDF loss."""
    loss_fn = SDFLoss(clamp_dist=0.1)
    
    pred = torch.randn(4, 1000, 1)
    gt = torch.randn(4, 1000, 1)
    
    loss = loss_fn(pred, gt)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_deepsdf_loss_without_latent() -> None:
    """Test DeepSDF loss without latent regularization."""
    loss_fn = DeepSDFLoss(latent_reg_lambda=1e-4, clamp_dist=0.1)
    
    pred = torch.randn(4, 1000, 1)
    gt = torch.randn(4, 1000, 1)
    
    total_loss, loss_dict = loss_fn(pred, gt, latent_codes=None)
    
    assert total_loss.item() >= 0
    assert 'sdf_loss' in loss_dict
    assert 'total_loss' in loss_dict


def test_deepsdf_loss_with_latent() -> None:
    """Test DeepSDF loss with latent regularization."""
    loss_fn = DeepSDFLoss(latent_reg_lambda=1e-4, clamp_dist=0.1)
    
    pred = torch.randn(4, 1000, 1)
    gt = torch.randn(4, 1000, 1)
    latent = torch.randn(4, 256)
    
    total_loss, loss_dict = loss_fn(pred, gt, latent_codes=latent)
    
    assert total_loss.item() >= 0
    assert 'sdf_loss' in loss_dict
    assert 'latent_reg' in loss_dict
    assert 'total_loss' in loss_dict


def test_loss_clamping() -> None:
    """Test that SDF values are clamped."""
    loss_fn = SDFLoss(clamp_dist=0.1)
    
    # Create values outside clamping range
    pred = torch.ones(4, 1000, 1) * 2.0
    gt = torch.ones(4, 1000, 1) * -2.0
    
    loss = loss_fn(pred, gt)
    
    # Loss should be bounded by 2 * clamp_dist
    assert loss.item() <= 0.2 + 0.01  # Small margin for numerical errors


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
