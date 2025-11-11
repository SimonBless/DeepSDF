# DeepSDF Visualization Utilities

Comprehensive visualization tools for DeepSDF model outputs.

## Overview

This package provides both high-level scripts and low-level utilities for visualizing DeepSDF reconstructions:

- **Static visualizations** with Matplotlib (4-panel views, grids)
- **Interactive 3D** with k3d (HTML exports, Jupyter integration)
- **Mesh exports** (.obj files for external tools)
- **Point cloud** and **SDF grid** visualizations

## Quick Start

### 1. Install Optional Dependencies

```bash
# For interactive 3D visualization
uv pip install k3d

# For matplotlib plotting (usually already installed)
uv pip install matplotlib
```

### 2. Visualize Your Model

```bash
# Basic usage
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --num-shapes 5 \
  --save-meshes

# With interactive HTML
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --num-shapes 5 \
  --resolution 128 \
  --save-meshes \
  --interactive
```

## Components

### High-Level Scripts

#### `examples/visualize_reconstructions.py`
Main visualization script with full functionality:
- Loads trained model checkpoints
- Reconstructs meshes using marching cubes
- Creates matplotlib and k3d visualizations
- Exports .obj files

See [VISUALIZATION.md](docs/VISUALIZATION.md) for full documentation.

#### `examples/view_mesh.py`
Simple mesh viewer for .obj files:
```bash
uv run python examples/view_mesh.py output/visualizations/shape_0.obj
```

### Low-Level Utilities

#### `deepsdf/utils/visualization.py`

Reusable visualization functions:

```python
from deepsdf.utils.visualization import (
    visualize_mesh,          # Single mesh (k3d)
    visualize_meshes,        # Multiple meshes side-by-side
    visualize_pointcloud,    # Point cloud visualization
    visualize_sdf,           # SDF grid as colored cubes
    compare_meshes,          # Compare original vs reconstruction
)
```

**Examples:**

```python
import trimesh
from deepsdf.utils.visualization import visualize_mesh

# Load a mesh
mesh = trimesh.load("shape_0.obj")

# Interactive visualization in Jupyter
visualize_mesh(mesh.vertices, mesh.faces)

# Save as standalone HTML
visualize_mesh(
    mesh.vertices, 
    mesh.faces,
    output_path="visualization.html"
)
```

```python
from deepsdf.utils.visualization import compare_meshes

# Compare original and reconstruction
original = trimesh.load("original.obj")
reconstructed = trimesh.load("reconstructed.obj")

compare_meshes(
    original, 
    reconstructed,
    labels=["Original", "Reconstructed"],
    output_path="comparison.html"
)
```

## Visualization Types

### 1. Matplotlib (Static)

4-panel view showing:
- 3D perspective
- XY projection
- XZ projection  
- YZ projection

Plus summary grids for multiple shapes.

### 2. K3d (Interactive)

Interactive 3D viewer with:
- Rotation, zoom, pan controls
- Smooth shading
- HTML export for sharing
- Jupyter notebook integration
- Side-by-side comparison mode

### 3. Mesh Export

Standard .obj files viewable in:
- MeshLab
- Blender
- MeshMixer
- CloudCompare
- Any 3D viewer

## API Reference

### `visualize_mesh(vertices, faces, flip_axes=False, output_path=None)`

Visualize a single mesh with k3d.

**Args:**
- `vertices`: (N, 3) array of vertex positions
- `faces`: (M, 3) array of face indices
- `flip_axes`: Apply coordinate system transformation
- `output_path`: Save as HTML file (None = display inline)

### `visualize_meshes(meshes, flip_axes=False, output_path=None)`

Visualize multiple meshes side by side (max 3).

**Args:**
- `meshes`: List of (vertices, faces) tuples or trimesh objects
- `flip_axes`: Apply coordinate system transformation
- `output_path`: Save as HTML file

### `visualize_pointcloud(point_cloud, point_size=0.01, colors=None, ...)`

Visualize a point cloud.

**Args:**
- `point_cloud`: (N, 3) array of point positions
- `point_size`: Size of rendered points
- `colors`: Optional per-point colors
- `output_path`: Save as HTML file

### `visualize_sdf(sdf, filename)`

Visualize SDF values on a 3D grid as colored cubes.

**Args:**
- `sdf`: 3D array of SDF values
- `filename`: Output mesh file path

### `compare_meshes(mesh1, mesh2, labels=None, output_path=None)`

Compare two meshes side by side (e.g., original vs reconstruction).

**Args:**
- `mesh1`: First trimesh object
- `mesh2`: Second trimesh object
- `labels`: List of labels for the meshes
- `output_path`: Save as HTML file

## Configuration

All visualization functions gracefully handle missing dependencies:

```python
from deepsdf.utils.visualization import K3D_AVAILABLE, MATPLOTLIB_AVAILABLE

if K3D_AVAILABLE:
    # Use interactive visualization
    visualize_mesh(vertices, faces)
else:
    print("Install k3d for interactive visualization")

if MATPLOTLIB_AVAILABLE:
    # Use static plotting
    plt.figure()
    ...
```

## Performance Notes

### Resolution vs Speed

Grid resolution affects reconstruction time and quality:

| Resolution | Time/Shape | Quality | Use Case |
|------------|-----------|---------|----------|
| 64 | ~5s | Rough | Quick preview |
| 96 | ~10s | Decent | Development |
| 128 | ~20s | Good | Production |
| 256 | 2-3min | Excellent | Publication |

### Memory Usage

- Resolution 64: ~500MB RAM
- Resolution 128: ~2GB RAM
- Resolution 256: ~4GB RAM

For large batches, use lower resolution or process fewer shapes at once.

## Complete Example

```python
import torch
import numpy as np
from deepsdf.inference.reconstructor import ShapeReconstructor
from deepsdf.models.decoder import DeepSDFDecoder
from deepsdf.utils.visualization import visualize_mesh, compare_meshes

# Load checkpoint
checkpoint = torch.load("checkpoint.pth")
latent_codes = checkpoint['latent_codes_state_dict']['weight']

# Create model
model = DeepSDFDecoder(latent_size=256)
model.load_state_dict(checkpoint['model_state_dict'])

# Create reconstructor
reconstructor = ShapeReconstructor(model)

# Reconstruct a shape
latent = latent_codes[0].detach().cpu().numpy()
mesh = reconstructor.reconstruct(latent, resolution=128)

# Visualize
visualize_mesh(
    mesh.vertices,
    mesh.faces,
    output_path="shape_0.html"
)

# Export mesh
mesh.export("shape_0.obj")

print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
```

## Documentation

- [VISUALIZATION.md](docs/VISUALIZATION.md) - Complete visualization guide
- [INFERENCE.md](docs/INFERENCE.md) - Inference and reconstruction
- [API.md](docs/API.md) - Full API reference

## Examples Directory

- `examples/visualize_reconstructions.py` - Main visualization script
- `examples/view_mesh.py` - Simple mesh viewer
- `examples/train.py` - Training script
- `examples/prepare_data.py` - Data preparation

## Tips & Tricks

1. **Jupyter Notebooks**: Visualizations display inline automatically
2. **Batch Processing**: Use `--num-shapes` for random sampling
3. **Specific Shapes**: Use `--shape-indices` to choose which shapes
4. **High Quality**: Use `--resolution 256` for publication-ready outputs
5. **Web Sharing**: HTML files are standalone and can be shared

## Troubleshooting

**ImportError: k3d not available**
```bash
uv pip install k3d
```

**Low quality meshes**
- Increase resolution (96 → 128 → 256)
- Check training progress (use later checkpoints)
- Verify latent codes are reasonable (mean≈0, std<0.2)

**Out of memory**
- Reduce resolution (128 → 96 → 64)
- Use `--device cpu`
- Process fewer shapes at once

**Blank visualizations**
- Check mesh is not empty: `len(mesh.vertices) > 0`
- Verify SDF values are reasonable (not all same sign)
- Try different resolution

## License

Same as DeepSDF project.
