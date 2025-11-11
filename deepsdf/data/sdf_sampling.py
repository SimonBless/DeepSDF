"""Utilities for SDF sampling from meshes."""

from typing import Tuple
import numpy as np
import trimesh


def sample_sdf_from_mesh(
    mesh: trimesh.Trimesh,
    num_samples: int = 10000,
    variance: float = 0.05,
    sample_near_surface: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points and their SDF values from a mesh.

    Args:
        mesh: Input mesh
        num_samples: Total number of points to sample
        variance: Variance for sampling around the surface
        sample_near_surface: Fraction of samples near the surface (vs. uniform in space)

    Returns:
        points: Sampled 3D points of shape (num_samples, 3)
        sdf_values: SDF values at those points of shape (num_samples, 1)
    """
    num_surface_samples = int(num_samples * sample_near_surface)
    num_uniform_samples = num_samples - num_surface_samples

    # Sample points near the surface
    surface_points, face_indices = mesh.sample(num_surface_samples, return_index=True)

    # Add noise to surface points
    noise = np.random.normal(0, variance, surface_points.shape)
    near_surface_points = surface_points + noise

    # Sample points uniformly in the bounding box
    bounds = mesh.bounds
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(num_uniform_samples, 3))

    # Combine all points
    all_points = np.vstack([near_surface_points, uniform_points])

    # Compute SDF values using trimesh
    # Positive outside, negative inside
    closest_points, distances, face_ids = mesh.nearest.on_surface(all_points)

    # Determine if points are inside or outside using ray casting
    is_inside = mesh.contains(all_points)

    # Compute signed distances
    sdf_values = distances.copy()
    sdf_values[is_inside] *= -1

    return all_points.astype(np.float32), sdf_values.reshape(-1, 1).astype(np.float32)


def normalize_mesh(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh:
    """
    Normalize mesh to fit within a unit sphere.

    Args:
        mesh: Input mesh
        scale: Scale factor (default: 1.0 for unit sphere)

    Returns:
        Normalized mesh
    """
    # Center the mesh
    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Scale to unit sphere
    max_dist: float = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
    mesh.vertices *= scale / max_dist

    return mesh


def load_mesh(path: str, normalize: bool = True) -> trimesh.Trimesh:
    """
    Load and optionally normalize a mesh.

    Args:
        path: Path to mesh file
        normalize: Whether to normalize the mesh

    Returns:
        Loaded mesh
    """
    loaded = trimesh.load(path, force="mesh")
    # Force to Trimesh type
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Loaded mesh is not a Trimesh: {type(loaded)}")
    mesh: trimesh.Trimesh = loaded

    if normalize:
        mesh = normalize_mesh(mesh)

    return mesh


def create_sdf_samples(
    mesh_path: str,
    num_samples: int = 10000,
    variance: float = 0.05,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create SDF samples from a mesh file.

    Args:
        mesh_path: Path to mesh file
        num_samples: Number of samples to generate
        variance: Variance for near-surface sampling
        normalize: Whether to normalize the mesh

    Returns:
        points: Sampled 3D points
        sdf_values: SDF values at those points
    """
    mesh = load_mesh(mesh_path, normalize=normalize)
    return sample_sdf_from_mesh(mesh, num_samples=num_samples, variance=variance)
