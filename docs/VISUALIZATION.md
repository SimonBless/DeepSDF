# DeepSDF Visualization Guide

This guide shows how to visualize your trained DeepSDF models.

## Quick Start

### Basic Visualization (Matplotlib Only)

```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/visualizations \
  --num-shapes 5 \
  --resolution 128 \
  --save-meshes
```

### Interactive Visualization (with k3d)

First, install k3d:
```bash
uv pip install k3d
```

Then run with interactive mode:
```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/visualizations \
  --num-shapes 5 \
  --resolution 128 \
  --save-meshes \
  --interactive
```

## Command-Line Options

- `--checkpoint PATH`: Path to model checkpoint (required)
- `--output-dir PATH`: Where to save visualizations (default: output/visualizations)
- `--num-shapes N`: Number of random shapes to reconstruct (default: 5)
- `--shape-indices I1 I2 ...`: Specific shape indices to reconstruct
- `--resolution N`: Grid resolution for marching cubes (default: 128)
  - Higher = better quality but slower (try 64, 96, 128, 256)
- `--save-meshes`: Save .obj mesh files
- `--interactive`: Create interactive HTML visualizations (requires k3d)
- `--no-matplotlib`: Skip static matplotlib visualizations
- `--device DEVICE`: cuda or cpu (default: auto-detect)

## Output Files

The script creates:

1. **OBJ files** (if `--save-meshes`): `shape_{index}.obj`
   - Can open in MeshLab, Blender, etc.

2. **Matplotlib visualizations**: `shape_{index}_matplotlib.png`
   - 4-panel view: 3D + XY/XZ/YZ projections
   - Summary grid: `summary_grid.png`

3. **Interactive HTML** (if `--interactive` and k3d installed):
   - `shape_{index}_interactive.html` - Individual shapes
   - `multi_mesh_interactive.html` - Side-by-side comparison

## Examples

### Reconstruct specific shapes
```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --shape-indices 0 10 20 30 40 \
  --resolution 256 \
  --save-meshes \
  --interactive
```

### High-quality reconstruction
```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --num-shapes 3 \
  --resolution 256 \
  --save-meshes \
  --interactive
```

### Quick preview (low resolution)
```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --num-shapes 10 \
  --resolution 64
```

## Utility Modules

The `deepsdf.utils.visualization` module provides reusable functions:

```python
from deepsdf.utils.visualization import (
    visualize_mesh,           # Single mesh k3d visualization
    visualize_meshes,         # Multiple meshes side-by-side
    visualize_pointcloud,     # Point cloud visualization
    visualize_sdf,            # SDF grid as colored cubes
    compare_meshes,           # Compare two meshes
)

# Example: Visualize a mesh
import trimesh
mesh = trimesh.load("shape_0.obj")

# Save as interactive HTML
visualize_mesh(
    mesh.vertices, 
    mesh.faces,
    output_path="my_visualization.html"
)

# Or display in Jupyter
visualize_mesh(mesh.vertices, mesh.faces)
```

## Tips

1. **Resolution vs Speed**:
   - 64: Very fast, rough preview (~5s per shape)
   - 96: Fast, decent quality (~10s per shape)
   - 128: Good balance (~20s per shape)
   - 256: High quality, slow (~2-3min per shape)

2. **Memory**:
   - Resolution 256 needs ~4GB RAM per shape
   - Use lower resolution if you run out of memory

3. **Interactive Visualization**:
   - Works in Jupyter notebooks (displays inline)
   - Saves as standalone HTML files
   - Can rotate, zoom, pan the 3D view

4. **Viewing Meshes**:
   - Simple viewer: `uv run python examples/view_mesh.py shape_0.obj`
   - Or use external tools: MeshLab, Blender, MeshMixer

## Troubleshooting

### "k3d not available"
```bash
uv pip install k3d
```

### "Matplotlib not available"
```bash
uv pip install matplotlib
```

### Low quality reconstructions
- Try higher resolution (e.g., 256)
- Check if model trained enough epochs
- Verify latent codes have reasonable values (mean≈0, std≈0.05-0.1)

### Out of memory
- Reduce resolution (e.g., 64 or 96)
- Reconstruct fewer shapes at once
- Use CPU instead of GPU: `--device cpu`
