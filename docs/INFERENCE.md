# DeepSDF Inference and Visualization Guide

## Overview

After training your DeepSDF model, you can reconstruct 3D shapes from learned latent codes and visualize them.

## Quick Start

### 1. Visualize Trained Shapes

Reconstruct and visualize shapes from your training set:

```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/visualizations \
  --num-shapes 5 \
  --resolution 128 \
  --save-meshes
```

**Parameters:**
- `--checkpoint`: Path to trained model checkpoint
- `--output-dir`: Where to save visualizations
- `--num-shapes`: How many random shapes to reconstruct
- `--resolution`: Grid resolution (64=fast, 128=balanced, 256=high quality)
- `--save-meshes`: Also save .obj mesh files

### 2. Reconstruct Specific Shapes

If you know which shapes you want to reconstruct (by their index in the training set):

```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/specific_shapes \
  --shape-indices 0 10 50 100 \
  --resolution 128 \
  --save-meshes
```

### 3. High-Quality Reconstruction

For publication-quality meshes, use higher resolution:

```bash
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/high_quality \
  --num-shapes 3 \
  --resolution 256 \
  --save-meshes
```

**Note:** Higher resolution takes longer:
- Resolution 64: ~5-10 seconds per shape
- Resolution 128: ~30-60 seconds per shape
- Resolution 256: ~3-5 minutes per shape

## Output Files

The visualization script creates:

1. **`shape_XXXX.obj`** - 3D mesh file (if `--save-meshes` is used)
   - Can be opened in Blender, MeshLab, or any 3D viewer
   
2. **`shape_XXXX_visualization.png`** - Multi-view visualization
   - Contains 4 views: 3D rendering + XY, XZ, YZ projections
   
3. **`summary.png`** - Grid of all reconstructed shapes

## Viewing 3D Meshes

### Option 1: MeshLab (Recommended)
```bash
# Install MeshLab
sudo apt install meshlab  # Ubuntu/Debian

# Open a mesh
meshlab output/visualizations/shape_0064.obj
```

### Option 2: Blender
```bash
# Open in Blender
blender output/visualizations/shape_0064.obj
```

### Option 3: Python Viewer

Create a simple interactive viewer:

```python
import trimesh

# Load mesh
mesh = trimesh.load('output/visualizations/shape_0064.obj')

# Show in viewer
mesh.show()
```

## Understanding the Output

### Checkpoint Information
When you run reconstruction, you'll see:
```
Checkpoint info:
  Epoch: 223                    # Training epochs completed
  Global step: 58554            # Total training iterations
  Total shapes in training: 1092  # Number of training shapes
  Latent size: 256              # Dimension of latent space
```

### Latent Code Statistics
For each shape:
```
Latent code stats: mean=-0.0045, std=0.0830
```
- **Mean close to 0**: Good! The latent codes are well-regularized
- **Std ~0.05-0.15**: Normal range for trained codes
- **Std > 0.5**: May indicate overfitting or training issues

### Mesh Quality
```
Mesh: 5934 vertices, 11864 faces
```
- More vertices/faces = more detail
- Empty meshes indicate reconstruction failed (may need more training)

## Advanced Usage

### Reconstruct with Different Checkpoints

Compare early vs late training:

```bash
# Early checkpoint (epoch 0)
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/checkpoint_epoch_0.pth \
  --output-dir output/epoch_0 \
  --shape-indices 0 1 2 \
  --resolution 128

# Final checkpoint
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/final \
  --shape-indices 0 1 2 \
  --resolution 128
```

### Batch Processing

Reconstruct all training shapes:

```bash
# Get number of shapes from checkpoint
NUM_SHAPES=$(uv run python -c "
import torch
ckpt = torch.load('output/training/deepsdf_low_memory/checkpoints/latest.pth', map_location='cpu')
print(ckpt['latent_codes_state_dict']['weight'].shape[0])
")

# Reconstruct all (warning: this takes a long time!)
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/all_shapes \
  --shape-indices $(seq 0 $((NUM_SHAPES-1))) \
  --resolution 64
```

### Extract Latent Codes

Save latent codes for further analysis:

```python
import torch
import numpy as np

# Load checkpoint
checkpoint = torch.load('output/training/deepsdf_low_memory/checkpoints/latest.pth')
latent_codes = checkpoint['latent_codes_state_dict']['weight'].numpy()

# Save as numpy array
np.save('output/latent_codes.npy', latent_codes)

print(f"Saved latent codes: {latent_codes.shape}")
# Output: Saved latent codes: (1092, 256)
```

## Troubleshooting

### Empty Meshes

**Symptoms:** `Warning: Empty mesh for shape X, skipping`

**Causes:**
- Model not trained enough
- Shape has extreme SDF values
- Resolution too low

**Solutions:**
1. Use a later checkpoint (more training)
2. Try higher resolution
3. Check if specific shapes consistently fail

### Out of Memory

**Symptoms:** CUDA OOM error during reconstruction

**Solutions:**
1. Lower resolution: `--resolution 64` or `--resolution 32`
2. Reconstruct fewer shapes at once
3. Use CPU: Edit the code to set `device = "cpu"`

### Poor Quality Reconstructions

**Symptoms:** Meshes look blobby or incomplete

**Causes:**
- Not enough training
- Poor data quality
- Wrong hyperparameters

**Solutions:**
1. Train longer
2. Use checkpoint from later epoch
3. Increase resolution for visualization
4. Check training loss (should be decreasing)

## Example Workflow

Complete pipeline from training to visualization:

```bash
# 1. Train model
uv run python examples/train.py \
  --config deepsdf/configs/low_memory_config.yaml \
  --data-dir data/sdf_sofas \
  --output-dir output/my_training

# 2. Visualize some results
uv run python examples/visualize_reconstructions.py \
  --checkpoint output/my_training/deepsdf_low_memory/checkpoints/latest.pth \
  --output-dir output/my_visualizations \
  --num-shapes 10 \
  --resolution 128 \
  --save-meshes

# 3. View in browser or image viewer
xdg-open output/my_visualizations/summary.png

# 4. Open mesh in MeshLab
meshlab output/my_visualizations/shape_0000.obj
```

## Tips for Best Results

1. **Resolution Trade-off**
   - Use 64 for quick previews
   - Use 128 for general visualization
   - Use 256 only for final high-quality outputs

2. **Check Training Progress**
   - Reconstruct periodically during training
   - Compare different checkpoints
   - Look for improvement over epochs

3. **Shape Selection**
   - Start with a few random shapes
   - If some look good, reconstruct more
   - Identify which types of shapes work best

4. **Mesh Post-Processing**
   - Use MeshLab for cleaning/smoothing
   - Apply Blender for rendering
   - Export to other formats as needed

## Next Steps

- Compare your reconstructions with original meshes
- Experiment with interpolation between latent codes
- Try shape completion tasks
- Export meshes for use in other applications
