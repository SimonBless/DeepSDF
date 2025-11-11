"""Visualize reconstructions from a trained DeepSDF model."""

import argparse
from pathlib import Path
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from deepsdf.inference.shape_reconstructor import ShapeReconstructor
from deepsdf.utils.config import load_config
from deepsdf.utils.visualization import (
    visualize_mesh, visualize_meshes, visualize_pointcloud,
    visualize_sdf, compare_meshes,
    K3D_AVAILABLE, MATPLOTLIB_AVAILABLE
)

# Try to import matplotlib for local plotting
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    pass

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deepsdf import DeepSDFDecoder, ShapeReconstructor
from deepsdf.utils import Config

# Try to import k3d for interactive visualization
try:
    import k3d
    K3D_AVAILABLE = True
except ImportError:
    K3D_AVAILABLE = False
    print("Note: k3d not available. Install with 'pip install k3d' for interactive 3D visualization.")


def visualize_mesh_k3d(mesh, flip_axes=False, name='mesh', output_path=None):
    """
    Visualize mesh using k3d for interactive 3D viewing.
    
    Args:
        mesh: trimesh.Trimesh object
        flip_axes: Whether to flip axes for different coordinate system
        name: Name for the plot
        output_path: If provided, save HTML file instead of displaying
    """
    if not K3D_AVAILABLE:
        print("k3d not available, skipping interactive visualization")
        return
    
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    
    if flip_axes:
        rot_matrix = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        vertices = vertices @ rot_matrix
    
    plot = k3d.plot(name=name, grid_visible=False, 
                    grid=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
    
    plt_mesh = k3d.mesh(vertices, faces, color=0x3b7dd6, flat_shading=False)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    
    if output_path:
        # Save as HTML
        with open(output_path, 'w') as f:
            f.write(plot.get_snapshot())
        print(f"Saved interactive visualization to {output_path}")
    else:
        plot.display()


def visualize_multiple_meshes_k3d(meshes, titles=None, flip_axes=False, output_path=None):
    """
    Visualize multiple meshes side by side using k3d.
    
    Args:
        meshes: List of trimesh.Trimesh objects
        titles: List of titles for each mesh
        flip_axes: Whether to flip axes
        output_path: If provided, save HTML file
    """
    if not K3D_AVAILABLE:
        print("k3d not available, skipping interactive visualization")
        return
    
    plot = k3d.plot(name='meshes', grid_visible=False,
                    grid=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0))
    
    num_meshes = len(meshes)
    spacing = 2.5
    
    colors = [0x3b7dd6, 0xd63b3b, 0x3bd63b, 0xd6d63b, 0xd63bd6, 0x3bd6d6]
    
    for i, mesh in enumerate(meshes):
        if len(mesh.vertices) == 0:
            continue
            
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.uint32)
        
        if flip_axes:
            vertices[:, 2] = vertices[:, 2] * -1
            vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
        
        # Offset meshes horizontally
        offset = np.array([i * spacing - (num_meshes - 1) * spacing / 2, 0, 0])
        vertices = vertices + offset
        
        color = colors[i % len(colors)]
        plt_mesh = k3d.mesh(vertices, faces, color=color, flat_shading=False)
        plot += plt_mesh
        plt_mesh.shader = '3d'
        
        # Add title as text (if k3d supports it in your version)
        if titles and i < len(titles):
            # Text labels would go here if supported
            pass
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(plot.get_snapshot())
        print(f"Saved interactive visualization to {output_path}")
    else:
        plot.display()


def visualize_mesh_matplotlib(mesh, output_path=None, title="Reconstructed Mesh"):
    """
    Visualize mesh using matplotlib.
    
    Args:
        mesh: trimesh.Trimesh object
        output_path: Optional path to save the visualization
        title: Plot title
    """
    if len(mesh.vertices) == 0:
        print("Warning: Empty mesh, cannot visualize")
        return
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D plot
    ax = fig.add_subplot(221, projection='3d')
    vertices = mesh.vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='blue', marker='.', s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\n{len(vertices)} vertices, {len(mesh.faces)} faces')
    
    # Set equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # XY projection
    ax2 = fig.add_subplot(222)
    ax2.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=1, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # XZ projection
    ax3 = fig.add_subplot(223)
    ax3.scatter(vertices[:, 0], vertices[:, 2], c='blue', s=1, alpha=0.3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # YZ projection
    ax4 = fig.add_subplot(224)
    ax4.scatter(vertices[:, 1], vertices[:, 2], c='blue', s=1, alpha=0.3)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main() -> None:
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize DeepSDF reconstructions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--shape-indices",
        type=int,
        nargs="+",
        default=None,
        help="Indices of shapes to reconstruct (from training set)",
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=5,
        help="Number of random shapes to reconstruct if --shape-indices not specified",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Grid resolution for marching cubes (default: 128, higher=slower but better quality)",
    )
    parser.add_argument(
        "--save-meshes",
        action="store_true",
        help="Save meshes as .obj files",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive k3d visualizations (HTML files)",
    )
    parser.add_argument(
        "--no-matplotlib",
        action="store_true",
        help="Skip matplotlib static visualizations",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")

    # Load config
    config = Config.from_dict(checkpoint["config"])
    
    # Load latent codes
    latent_dict = checkpoint["latent_codes_state_dict"]
    # The key might be either 'weight' or 'latent_codes.weight'
    if "weight" in latent_dict:
        all_latent_codes = latent_dict["weight"]
    elif "latent_codes.weight" in latent_dict:
        all_latent_codes = latent_dict["latent_codes.weight"]
    else:
        raise KeyError(f"Could not find latent codes. Available keys: {list(latent_dict.keys())}")
    num_total_shapes = all_latent_codes.shape[0]
    print(f"  Total shapes in training: {num_total_shapes}")
    print(f"  Latent size: {all_latent_codes.shape[1]}")

    # Create model
    print("\nCreating model...")
    model = DeepSDFDecoder(
        latent_size=config.model.latent_size,
        hidden_dims=config.model.hidden_dims,
        dropout_prob=config.model.dropout_prob,
        norm_layers=config.model.norm_layers,
        latent_in=config.model.latent_in,
        weight_norm=config.model.weight_norm,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create reconstructor
    reconstructor = ShapeReconstructor(model, device=device)

    # Determine which shapes to reconstruct
    if args.shape_indices is not None:
        shape_indices = args.shape_indices
        print(f"\nReconstructing specified shapes: {shape_indices}")
    else:
        # Select random shapes
        shape_indices = np.random.choice(num_total_shapes, 
                                        size=min(args.num_shapes, num_total_shapes),
                                        replace=False).tolist()
        print(f"\nReconstructing {len(shape_indices)} random shapes")

    # Reconstruct each shape
    reconstructed_meshes = []
    for i, idx in enumerate(shape_indices):
        if idx >= num_total_shapes:
            print(f"Warning: Shape index {idx} out of range (max: {num_total_shapes-1}), skipping")
            continue
            
        print(f"\n[{i+1}/{len(shape_indices)}] Reconstructing shape {idx}...")
        latent_code = all_latent_codes[idx]
        
        print(f"  Latent code stats: mean={latent_code.mean():.4f}, std={latent_code.std():.4f}")
        
        # Reconstruct mesh
        print(f"  Running marching cubes (resolution={args.resolution})...")
        mesh = reconstructor.reconstruct_mesh(
            latent_code,
            resolution=args.resolution,
        )
        
        if len(mesh.vertices) == 0:
            print(f"  Warning: Empty mesh for shape {idx}, skipping")
            continue
        
        print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Save mesh
        if args.save_meshes:
            mesh_path = output_dir / f"shape_{idx:04d}.obj"
            reconstructor.save_mesh(mesh, str(mesh_path))
        
        # Matplotlib visualization (unless disabled)
        if not args.no_matplotlib:
            vis_path = output_dir / f"shape_{idx:04d}_visualization.png"
            visualize_mesh_matplotlib(
                mesh,
                output_path=str(vis_path),
                title=f"Shape {idx} (Epoch {checkpoint.get('epoch', 'N/A')})"
            )
        
        # Interactive k3d visualization
        if args.interactive and K3D_AVAILABLE:
            html_path = output_dir / f"shape_{idx:04d}_interactive.html"
            visualize_mesh_k3d(
                mesh,
                name=f"Shape {idx}",
                output_path=str(html_path)
            )
        
        # Store for batch visualization
        reconstructed_meshes.append((idx, mesh))

    print(f"\n✓ Done! Visualizations saved to {output_dir}")
    
    # Create a summary plot if multiple shapes
    if len(reconstructed_meshes) > 1 and not args.no_matplotlib:
        print("\nCreating summary visualization...")
        create_summary_plot(output_dir, shape_indices)
    
    # Create interactive multi-mesh view
    if args.interactive and K3D_AVAILABLE and len(reconstructed_meshes) > 1:
        print("\nCreating interactive multi-mesh visualization...")
        meshes = [mesh for _, mesh in reconstructed_meshes[:6]]  # Limit to 6 meshes
        titles = [f"Shape {idx}" for idx, _ in reconstructed_meshes[:6]]
        multi_html_path = output_dir / "all_shapes_interactive.html"
        visualize_multiple_meshes_k3d(meshes, titles=titles, output_path=str(multi_html_path))
    
    # Print summary
    print(f"\n{'='*60}")
    print("Visualization Summary:")
    print(f"{'='*60}")
    print(f"Total shapes reconstructed: {len(reconstructed_meshes)}")
    if args.save_meshes:
        print(f"✓ OBJ mesh files saved")
    if not args.no_matplotlib:
        print(f"✓ Matplotlib visualizations saved (.png)")
    if args.interactive and K3D_AVAILABLE:
        print(f"✓ Interactive visualizations saved (.html)")
        print(f"  Open in browser: {output_dir}/shape_XXXX_interactive.html")
    print(f"{'='*60}")


def create_summary_plot(output_dir, shape_indices):
    """Create a summary plot with all visualizations."""
    viz_files = [output_dir / f"shape_{idx:04d}_visualization.png" 
                 for idx in shape_indices]
    viz_files = [f for f in viz_files if f.exists()]
    
    if len(viz_files) == 0:
        return
    
    n = len(viz_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, viz_file in enumerate(viz_files):
        img = plt.imread(viz_file)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Shape {shape_indices[i]}', fontsize=12)
    
    # Hide unused subplots
    for i in range(len(viz_files), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    summary_path = output_dir / "summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Summary saved to {summary_path}")
    plt.close()


if __name__ == "__main__":
    main()
