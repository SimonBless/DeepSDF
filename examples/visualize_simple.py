"""Simple script to visualize reconstructions using the new visualization utilities."""

import argparse
from pathlib import Path
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from deepsdf.inference.reconstructor import ShapeReconstructor
from deepsdf.utils.visualization import (
    visualize_mesh, visualize_meshes, 
    K3D_AVAILABLE, MATPLOTLIB_AVAILABLE
)

# Try to import matplotlib for fallback plotting
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    plt = None


def visualize_mesh_matplotlib(mesh: trimesh.Trimesh, output_path: Path) -> None:
    """Visualize mesh with matplotlib (4-panel view)."""
    if not MATPLOTLIB_AVAILABLE or plt is None:
        print("Matplotlib not available for visualization")
        return
        
    fig = plt.figure(figsize=(16, 4))
    
    # 3D view
    ax1 = fig.add_subplot(141, projection='3d')
    mesh_collection = Poly3DCollection(
        mesh.vertices[mesh.faces], alpha=0.7, facecolor='cyan', edgecolor='black'
    )
    ax1.add_collection3d(mesh_collection)
    
    scale = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    scale = scale.max() / 2
    mid = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
    ax1.set_xlim(mid[0] - scale, mid[0] + scale)
    ax1.set_ylim(mid[1] - scale, mid[1] + scale)
    ax1.set_zlim(mid[2] - scale, mid[2] + scale)
    ax1.set_title('3D View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # XY projection
    ax2 = fig.add_subplot(142)
    ax2.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces, linewidth=0.5)
    ax2.set_title('XY Projection')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    
    # XZ projection
    ax3 = fig.add_subplot(143)
    ax3.triplot(mesh.vertices[:, 0], mesh.vertices[:, 2], mesh.faces, linewidth=0.5)
    ax3.set_title('XZ Projection')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_aspect('equal')
    
    # YZ projection
    ax4 = fig.add_subplot(144)
    ax4.triplot(mesh.vertices[:, 1], mesh.vertices[:, 2], mesh.faces, linewidth=0.5)
    ax4.set_title('YZ Projection')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize DeepSDF reconstructions")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    parser.add_argument("--output-dir", type=str, default="output/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--num-shapes", type=int, default=5,
                       help="Number of shapes to reconstruct")
    parser.add_argument("--shape-indices", type=int, nargs="+", default=None,
                       help="Specific shape indices to reconstruct")
    parser.add_argument("--resolution", type=int, default=128,
                       help="Grid resolution for reconstruction")
    parser.add_argument("--save-meshes", action="store_true",
                       help="Save reconstructed meshes as .obj files")
    parser.add_argument("--interactive", action="store_true",
                       help="Create interactive k3d visualizations (HTML)")
    parser.add_argument("--no-matplotlib", action="store_true",
                       help="Skip matplotlib static visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for reconstruction")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Global step: {checkpoint.get('global_step', 'unknown')}")
    
    latent_codes = checkpoint['latent_codes_state_dict']['weight']
    num_total_shapes = latent_codes.size(0)
    latent_size = latent_codes.size(1)
    print(f"  Total shapes in training: {num_total_shapes}")
    print(f"  Latent size: {latent_size}")
    
    # Determine which shapes to reconstruct
    if args.shape_indices is not None:
        shape_indices = args.shape_indices
    else:
        shape_indices = np.random.choice(num_total_shapes, args.num_shapes, replace=False)
    
    # Initialize reconstructor
    reconstructor = ShapeReconstructor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Reconstruct shapes
    meshes = []
    mesh_info = []
    
    for i, idx in enumerate(tqdm(shape_indices, desc="Reconstructing shapes")):
        print(f"\n[{i+1}/{len(shape_indices)}] Reconstructing shape {idx}...")
        
        # Get latent code
        latent = latent_codes[idx].detach().cpu().numpy()
        print(f"  Latent code stats: mean={latent.mean():.4f}, std={latent.std():.4f}")
        
        # Reconstruct mesh
        mesh = reconstructor.reconstruct(
            latent_code=latent,
            resolution=args.resolution
        )
        
        print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        meshes.append(mesh)
        mesh_info.append({'index': idx, 'latent': latent})
        
        # Save individual visualizations
        if args.save_meshes:
            mesh_path = output_dir / f"shape_{idx}.obj"
            mesh.export(str(mesh_path))
        
        if not args.no_matplotlib:
            viz_path = output_dir / f"shape_{idx}_matplotlib.png"
            visualize_mesh_matplotlib(mesh, viz_path)
        
        if args.interactive and K3D_AVAILABLE:
            html_path = output_dir / f"shape_{idx}_interactive.html"
            visualize_mesh(
                mesh.vertices, mesh.faces,
                flip_axes=False,
                output_path=str(html_path)
            )
    
    # Create summary visualizations
    print(f"\n✓ Done! Visualizations saved to {output_dir}")
    print(f"Total shapes reconstructed: {len(meshes)}")
    
    if args.save_meshes:
        print(f"✓ OBJ mesh files saved")
    
    if not args.no_matplotlib:
        print(f"✓ Matplotlib visualizations saved (.png)")
        
        # Create summary grid
        if len(meshes) > 1:
            n_cols = min(3, len(meshes))
            n_rows = (len(meshes) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, 
                                    subplot_kw={'projection': '3d'},
                                    figsize=(6*n_cols, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (mesh, info) in enumerate(zip(meshes, mesh_info)):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                mesh_collection = Poly3DCollection(
                    mesh.vertices[mesh.faces], alpha=0.7, 
                    facecolor='cyan', edgecolor='black'
                )
                ax.add_collection3d(mesh_collection)
                
                scale = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max() / 2
                mid = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
                ax.set_xlim(mid[0] - scale, mid[0] + scale)
                ax.set_ylim(mid[1] - scale, mid[1] + scale)
                ax.set_zlim(mid[2] - scale, mid[2] + scale)
                ax.set_title(f"Shape {info['index']}")
                
            # Hide empty subplots
            for i in range(len(meshes), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / "summary_grid.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Summary grid saved")
    
    if args.interactive:
        if K3D_AVAILABLE and len(meshes) > 1:
            # Create multi-mesh visualization
            mesh_list = [(m.vertices, m.faces) for m in meshes[:min(3, len(meshes))]]
            html_path = output_dir / "multi_mesh_interactive.html"
            visualize_meshes(
                mesh_list,
                flip_axes=False,
                output_path=str(html_path)
            )
            print(f"✓ Interactive multi-mesh visualization saved (.html)")
        elif not K3D_AVAILABLE:
            print("⚠ k3d not available. Install with: pip install k3d")


if __name__ == "__main__":
    main()
