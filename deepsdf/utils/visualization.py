"""Visualization utilities for DeepSDF."""

from pathlib import Path
from typing import Optional, List
import numpy as np
import trimesh

# Try to import optional visualization libraries
try:
    import k3d
    K3D_AVAILABLE = True
except ImportError:
    K3D_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def visualize_mesh(vertices: np.ndarray, faces: np.ndarray, 
                   flip_axes: bool = False, output_path: Optional[str] = None):
    """
    Visualize mesh using k3d for interactive 3D viewing.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        flip_axes: Whether to flip axes for different coordinate system
        output_path: If provided, save as HTML file
    """
    if not K3D_AVAILABLE:
        print("k3d not available. Install with: pip install k3d")
        return
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    
    plot = k3d.plot(name='mesh', grid_visible=False, 
                    grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    
    if flip_axes:
        rot_matrix = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        vertices = vertices @ rot_matrix
    
    plt_mesh = k3d.mesh(vertices, faces, color=0xd0d0d0, flat_shading=False)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(plot.get_snapshot())
        print(f"Saved interactive visualization to {output_path}")
    else:
        plot.display()


def visualize_meshes(meshes: List, flip_axes: bool = False, output_path: Optional[str] = None):
    """
    Visualize multiple meshes side by side using k3d.
    
    Args:
        meshes: List of (vertices, faces) tuples or trimesh objects
        flip_axes: Whether to flip axes
        output_path: If provided, save as HTML file
    """
    if not K3D_AVAILABLE:
        print("k3d not available. Install with: pip install k3d")
        return
    
    assert len(meshes) <= 3, "Maximum 3 meshes supported"
    
    plot = k3d.plot(name='meshes', grid_visible=False,
                    grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    
    offsets = [[-32, -32, 0], [0, -32, 0], [32, -32, 0]]
    
    for mesh_idx, mesh in enumerate(meshes):
        # Handle both tuple format and trimesh objects
        if isinstance(mesh, trimesh.Trimesh):
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32)
        else:
            vertices = np.array(mesh[0], dtype=np.float32)
            faces = np.array(mesh[1], dtype=np.uint32)
        
        if flip_axes:
            vertices[:, 2] = vertices[:, 2] * -1
            vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
        
        vertices += np.array(offsets[mesh_idx], dtype=np.float32)
        
        plt_mesh = k3d.mesh(vertices, faces, color=0xd0d0d0, flat_shading=False)
        plot += plt_mesh
        plt_mesh.shader = '3d'
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(plot.get_snapshot())
        print(f"Saved interactive visualization to {output_path}")
    else:
        plot.display()


def visualize_pointcloud(point_cloud: np.ndarray, point_size: float = 0.01,
                        colors: Optional[np.ndarray] = None, flip_axes: bool = False,
                        name: str = 'point_cloud', output_path: Optional[str] = None):
    """
    Visualize point cloud using k3d.
    
    Args:
        point_cloud: Points (N, 3)
        point_size: Size of each point
        colors: Optional per-point colors
        flip_axes: Whether to flip axes
        name: Plot name
        output_path: If provided, save as HTML file
    """
    if not K3D_AVAILABLE:
        print("k3d not available. Install with: pip install k3d")
        return
    
    point_cloud = point_cloud.copy()
    plot = k3d.plot(name=name, grid_visible=False,
                    grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
        point_cloud[:, 1] = point_cloud[:, 1] * -1
    
    plt_points = k3d.points(
        positions=point_cloud.astype(np.float32),
        point_size=point_size,
        colors=colors if colors is not None else [],
        color=0xd0d0d0
    )
    plot += plt_points
    plt_points.shader = '3d'
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(plot.get_snapshot())
        print(f"Saved interactive visualization to {output_path}")
    else:
        plot.display()


def visualize_sdf(sdf: np.ndarray, filename: Path) -> None:
    """
    Visualize SDF grid as colored cubes.
    
    Args:
        sdf: SDF values on a cubic grid
        filename: Output path for mesh file
    """
    from matplotlib import cm, colors
    
    assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2], "SDF grid must be cubic"
    print(f"Creating SDF visualization for {sdf.shape[0]}^3 grid...")
    
    voxels = np.stack(
        np.meshgrid(range(sdf.shape[0]), range(sdf.shape[1]), range(sdf.shape[2]))
    ).reshape(3, -1).T
    
    # Normalize SDF values
    sdf_vis = sdf.copy()
    sdf_vis[sdf_vis < 0] /= np.abs(sdf_vis[sdf_vis < 0]).max()
    sdf_vis[sdf_vis > 0] /= sdf_vis[sdf_vis > 0].max()
    sdf_vis /= 2.
    
    # Create cube corners
    corners = np.array([
        [-.25, -.25, -.25], [.25, -.25, -.25],
        [-.25, .25, -.25], [.25, .25, -.25],
        [-.25, -.25, .25], [.25, -.25, .25],
        [-.25, .25, .25], [.25, .25, .25]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)
    
    # Scale factors and colors
    scale_factors = sdf_vis[tuple(voxels.T)].repeat(8, axis=0)
    cube_vertex_colors = cm.get_cmap('seismic')(
        colors.Normalize(vmin=-1, vmax=1)(scale_factors)
    )[:, :3]
    
    scale_factors[scale_factors < 0] *= .25
    cube_vertices = voxels.repeat(8, axis=0) + corners * scale_factors[:, np.newaxis]
    
    # Create faces
    faces = np.array([
        [1, 0, 2], [2, 3, 1], [5, 1, 3], [3, 7, 5],
        [4, 5, 7], [7, 6, 4], [0, 4, 6], [6, 2, 0],
        [3, 2, 6], [6, 7, 3], [5, 4, 0], [0, 1, 5]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)
    
    cube_faces = faces + (np.arange(0, voxels.shape[0]) * 8)[
        np.newaxis, :
    ].repeat(12, axis=0).T.flatten()[:, np.newaxis]
    
    # Export mesh
    mesh = trimesh.Trimesh(
        vertices=cube_vertices,
        faces=cube_faces,
        vertex_colors=cube_vertex_colors,
        process=False
    )
    mesh.export(str(filename))
    print(f"Exported SDF visualization to {filename}")


def compare_meshes(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh,
                   labels: List[str] = None, output_path: Optional[str] = None):
    """
    Compare two meshes side by side (e.g., original vs reconstructed).
    
    Args:
        mesh1: First mesh (e.g., original)
        mesh2: Second mesh (e.g., reconstruction)
        labels: Labels for the meshes
        output_path: If provided, save as HTML file
    """
    if not K3D_AVAILABLE:
        print("k3d not available. Install with: pip install k3d")
        return
    
    if labels is None:
        labels = ["Mesh 1", "Mesh 2"]
    
    plot = k3d.plot(name='comparison', grid_visible=False,
                    grid=(-1.5, -1.5, -1.5, 1.5, 1.5, 1.5))
    
    # First mesh (left, red)
    v1 = np.array(mesh1.vertices, dtype=np.float32)
    v1 = v1 - np.array([1.2, 0, 0])  # Offset left
    f1 = np.array(mesh1.faces, dtype=np.uint32)
    plt_mesh1 = k3d.mesh(v1, f1, color=0xd00d0d, flat_shading=False)
    plot += plt_mesh1
    plt_mesh1.shader = '3d'
    
    # Second mesh (right, green)
    v2 = np.array(mesh2.vertices, dtype=np.float32)
    v2 = v2 + np.array([1.2, 0, 0])  # Offset right
    f2 = np.array(mesh2.faces, dtype=np.uint32)
    plt_mesh2 = k3d.mesh(v2, f2, color=0x0dd00d, flat_shading=False)
    plot += plt_mesh2
    plt_mesh2.shader = '3d'
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(plot.get_snapshot())
        print(f"Saved comparison to {output_path}")
    else:
        plot.display()
