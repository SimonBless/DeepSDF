"""Shape reconstruction from latent codes."""

from typing import Tuple
import numpy as np
import torch
from skimage import measure
import trimesh

from deepsdf.models.decoder import DeepSDFDecoder


class ShapeReconstructor:
    """
    Reconstruct 3D shapes from latent codes using marching cubes.

    Args:
        model: Trained DeepSDF decoder
        device: Device to use for inference
    """

    def __init__(
        self,
        model: DeepSDFDecoder,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def create_grid(
        self,
        resolution: int = 256,
        bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """
        Create a 3D grid of points.

        Args:
            resolution: Grid resolution per dimension
            bounds: Min and max values for the grid

        Returns:
            Grid points of shape (resolution^3, 3)
        """
        min_val, max_val = bounds
        x = np.linspace(min_val, max_val, resolution)
        y = np.linspace(min_val, max_val, resolution)
        z = np.linspace(min_val, max_val, resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        return torch.from_numpy(grid_points).float()

    def evaluate_grid(
        self,
        latent_code: torch.Tensor,
        grid_points: torch.Tensor,
        batch_size: int = 32768,
    ) -> np.ndarray:
        """
        Evaluate SDF values on a grid.

        Args:
            latent_code: Latent code of shape (latent_size,)
            grid_points: Grid points of shape (num_points, 3)
            batch_size: Batch size for evaluation

        Returns:
            SDF values of shape (num_points,)
        """
        num_points = grid_points.shape[0]
        sdf_values = []

        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                batch_points = grid_points[i : i + batch_size].to(self.device)
                batch_points = batch_points.unsqueeze(0)  # Add batch dimension

                latent_batch = latent_code.unsqueeze(0).to(self.device)

                batch_sdf = self.model.inference(latent_batch, batch_points)
                sdf_values.append(batch_sdf.squeeze().cpu().numpy())

        result: np.ndarray = np.concatenate(sdf_values)
        return result

    def reconstruct_mesh(
        self,
        latent_code: torch.Tensor,
        resolution: int = 256,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        level: float = 0.0,
    ) -> trimesh.Trimesh:
        """
        Reconstruct a mesh from a latent code using marching cubes.

        Args:
            latent_code: Latent code of shape (latent_size,)
            resolution: Grid resolution
            bounds: Spatial bounds for reconstruction
            level: Iso-surface level (0.0 for the surface)

        Returns:
            Reconstructed mesh
        """
        # Create grid
        grid_points = self.create_grid(resolution, bounds)

        # Evaluate SDF
        sdf_values = self.evaluate_grid(latent_code, grid_points)

        # Reshape to 3D grid
        sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

        # Run marching cubes
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                sdf_grid,
                level=level,
                spacing=[(bounds[1] - bounds[0]) / resolution] * 3,
            )

            # Offset vertices to correct position
            verts = verts + np.array([bounds[0], bounds[0], bounds[0]])

            # Create mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

            return mesh
        except (ValueError, RuntimeError) as e:
            print(f"Marching cubes failed: {e}")
            # Return empty mesh
            return trimesh.Trimesh()

    def reconstruct_batch(
        self,
        latent_codes: torch.Tensor,
        resolution: int = 256,
        bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> list[trimesh.Trimesh]:
        """
        Reconstruct multiple meshes from a batch of latent codes.

        Args:
            latent_codes: Latent codes of shape (batch_size, latent_size)
            resolution: Grid resolution
            bounds: Spatial bounds for reconstruction

        Returns:
            List of reconstructed meshes
        """
        meshes = []
        for i in range(latent_codes.shape[0]):
            mesh = self.reconstruct_mesh(
                latent_codes[i],
                resolution=resolution,
                bounds=bounds,
            )
            meshes.append(mesh)

        return meshes

    def save_mesh(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
    ) -> None:
        """
        Save a mesh to file.

        Args:
            mesh: Mesh to save
            output_path: Output file path
        """
        mesh.export(output_path)
        print(f"Mesh saved to {output_path}")
