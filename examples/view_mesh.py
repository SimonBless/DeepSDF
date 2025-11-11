"""Simple script to view meshes interactively using trimesh."""

import argparse
import trimesh
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="View 3D meshes interactively")
    parser.add_argument(
        "mesh_files",
        type=str,
        nargs="+",
        help="Path(s) to mesh file(s) (.obj, .ply, .stl, etc.)",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply smoothing to the mesh",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Show wireframe",
    )
    
    args = parser.parse_args()
    
    for mesh_file in args.mesh_files:
        mesh_path = Path(mesh_file)
        
        if not mesh_path.exists():
            print(f"Error: File not found: {mesh_file}")
            continue
        
        print(f"\nLoading {mesh_path.name}...")
        
        try:
            # Load mesh
            mesh = trimesh.load(str(mesh_path))
            
            # Print info
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Bounds: {mesh.bounds}")
            print(f"  Watertight: {mesh.is_watertight}")
            print(f"  Volume: {mesh.volume:.4f}")
            
            # Apply smoothing if requested
            if args.smooth:
                print("  Applying Laplacian smoothing...")
                trimesh.smoothing.filter_laplacian(mesh, iterations=5)
            
            # Show mesh
            print(f"  Opening viewer...")
            scene = trimesh.Scene(mesh)
            
            if args.wireframe:
                scene.add_geometry(mesh, geom_name="wireframe")
            
            scene.show()
            
        except Exception as e:
            print(f"Error loading mesh: {e}")


if __name__ == "__main__":
    main()
