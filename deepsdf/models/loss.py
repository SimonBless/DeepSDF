"""Loss functions for DeepSDF training."""

from typing import Optional
import torch
import torch.nn as nn


class DeepSDFLoss(nn.Module):
    """
    Combined loss function for DeepSDF training.
    
    This includes:
    - L1 loss for SDF prediction
    - L2 regularization for latent codes
    
    Args:
        latent_reg_lambda: Weight for latent code regularization (default: 1e-4)
        clamp_dist: Maximum distance for clamping SDF values (default: 0.1)
    """
    
    def __init__(
        self,
        latent_reg_lambda: float = 1e-4,
        clamp_dist: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_reg_lambda = latent_reg_lambda
        self.clamp_dist = clamp_dist
        self.l1_loss = nn.L1Loss(reduction='mean')
    
    def forward(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        latent_codes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the loss.
        
        Args:
            pred_sdf: Predicted SDF values of shape (batch_size, num_points, 1)
            gt_sdf: Ground truth SDF values of shape (batch_size, num_points, 1)
            latent_codes: Latent codes of shape (batch_size, latent_size) for regularization
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Clamp SDF values to avoid extreme values
        pred_sdf_clamped = torch.clamp(pred_sdf, -self.clamp_dist, self.clamp_dist)
        gt_sdf_clamped = torch.clamp(gt_sdf, -self.clamp_dist, self.clamp_dist)
        
        # SDF reconstruction loss (L1)
        sdf_loss = self.l1_loss(pred_sdf_clamped, gt_sdf_clamped)
        
        # Latent code regularization (L2)
        if latent_codes is not None:
            latent_reg = torch.mean(torch.norm(latent_codes, dim=1))
            total_loss = sdf_loss + self.latent_reg_lambda * latent_reg
            
            loss_dict = {
                'total_loss': total_loss.item(),
                'sdf_loss': sdf_loss.item(),
                'latent_reg': latent_reg.item(),
            }
        else:
            total_loss = sdf_loss
            loss_dict = {
                'total_loss': total_loss.item(),
                'sdf_loss': sdf_loss.item(),
            }
        
        return total_loss, loss_dict


class SDFLoss(nn.Module):
    """
    Simple L1 loss for SDF prediction with optional clamping.
    
    Args:
        clamp_dist: Maximum distance for clamping SDF values (default: 0.1)
    """
    
    def __init__(self, clamp_dist: float = 0.1) -> None:
        super().__init__()
        self.clamp_dist = clamp_dist
        self.l1_loss = nn.L1Loss(reduction='mean')
    
    def forward(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the L1 loss between predicted and ground truth SDF.
        
        Args:
            pred_sdf: Predicted SDF values
            gt_sdf: Ground truth SDF values
        
        Returns:
            L1 loss value
        """
        # Clamp SDF values
        pred_sdf_clamped = torch.clamp(pred_sdf, -self.clamp_dist, self.clamp_dist)
        gt_sdf_clamped = torch.clamp(gt_sdf, -self.clamp_dist, self.clamp_dist)
        
        return self.l1_loss(pred_sdf_clamped, gt_sdf_clamped)
