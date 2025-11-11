"""DeepSDF Decoder Network Implementation."""

from typing import Optional, List
import torch
import torch.nn as nn


class DeepSDFDecoder(nn.Module):
    """
    DeepSDF Decoder network with skip connections.

    This decoder takes a latent code concatenated with 3D coordinates
    and predicts the signed distance value at those coordinates.

    Args:
        latent_size: Dimension of the latent code (default: 256)
        hidden_dims: List of hidden layer dimensions
            (default: [512, 512, 512, 512, 512, 512, 512, 512])
        dropout_prob: Dropout probability (default: 0.2)
        norm_layers: List of layer indices where layer normalization is applied
        latent_in: List of layer indices where latent code is concatenated
            (for skip connections)
        weight_norm: Whether to use weight normalization
    """

    def __init__(
        self,
        latent_size: int = 256,
        hidden_dims: Optional[List[int]] = None,
        dropout_prob: float = 0.2,
        norm_layers: Optional[List[int]] = None,
        latent_in: Optional[List[int]] = None,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 512, 512, 512, 512, 512]

        if norm_layers is None:
            norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]

        if latent_in is None:
            latent_in = [4]

        self.latent_size = latent_size
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.weight_norm = weight_norm

        # Input: 3D coordinates (3) + latent code (latent_size)
        dims = [latent_size + 3] + hidden_dims + [1]

        self.num_layers = len(dims)
        self.layers = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer_idx in range(self.num_layers - 1):
            # Determine input dimension for this layer
            if layer_idx + 1 in latent_in:
                # Add skip connection from latent code
                in_dim = dims[layer_idx] + latent_size
            else:
                in_dim = dims[layer_idx]

            out_dim = dims[layer_idx + 1]

            # Create linear layer
            linear = nn.Linear(in_dim, out_dim)

            # Apply weight normalization if specified
            if weight_norm:
                linear = nn.utils.weight_norm(linear)

            self.layers.append(linear)

            # Add layer normalization
            if layer_idx in norm_layers:
                self.layer_norms.append(nn.LayerNorm(out_dim))
            else:
                self.layer_norms.append(nn.Identity())

            # Add dropout (not applied to output layer)
            if layer_idx < self.num_layers - 2:
                self.dropout.append(nn.Dropout(dropout_prob))
            else:
                self.dropout.append(nn.Identity())

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, latent_vector: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            latent_vector: Latent code of shape (batch_size, latent_size)
            xyz: 3D coordinates of shape (batch_size, num_points, 3)

        Returns:
            Predicted SDF values of shape (batch_size, num_points, 1)
        """
        batch_size = xyz.shape[0]
        num_points = xyz.shape[1]

        # Expand latent vector to match number of points
        latent_repeat = latent_vector.unsqueeze(1).expand(-1, num_points, -1)

        # Concatenate latent code with coordinates
        x = torch.cat([latent_repeat, xyz], dim=2)
        x = x.reshape(-1, self.latent_size + 3)

        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            # Add skip connection if needed
            if layer_idx + 1 in self.latent_in and layer_idx > 0:
                latent_flat = latent_vector.unsqueeze(1).expand(-1, num_points, -1)
                latent_flat = latent_flat.reshape(-1, self.latent_size)
                x = torch.cat([x, latent_flat], dim=1)

            # Linear transformation
            x = layer(x)

            # Apply layer norm
            x = self.layer_norms[layer_idx](x)

            # Apply activation (ReLU for hidden layers, Tanh for output)
            if layer_idx < len(self.layers) - 1:
                x = self.relu(x)
                x = self.dropout[layer_idx](x)
            else:
                # Output layer - use tanh to bound the output
                x = self.tanh(x)

        # Reshape to (batch_size, num_points, 1)
        x = x.reshape(batch_size, num_points, 1)

        return x

    def inference(self, latent_vector: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Inference mode forward pass (no dropout).

        Args:
            latent_vector: Latent code of shape (latent_size,) or (batch_size, latent_size)
            xyz: 3D coordinates of shape (num_points, 3) or (batch_size, num_points, 3)

        Returns:
            Predicted SDF values
        """
        self.eval()
        with torch.no_grad():
            # Handle single latent vector
            if latent_vector.dim() == 1:
                latent_vector = latent_vector.unsqueeze(0)

            # Handle single batch of points
            if xyz.dim() == 2:
                xyz = xyz.unsqueeze(0)

            return self.forward(latent_vector, xyz)
