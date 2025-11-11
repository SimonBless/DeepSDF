# Quick Start Guide

Get started with DeepSDF in minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/SimonBless/DeepSDF.git
cd DeepSDF

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

## 1. Prepare Your Data

### Option A: Use Example Meshes

```python
from deepsdf.data import create_sdf_samples
import numpy as np

# Generate SDF samples from a mesh
points, sdf_values = create_sdf_samples(
    mesh_path='path/to/your/mesh.obj',
    num_samples=100000,
)

# Save as .npz file
np.savez('data/shape_001.npz', points=points, sdf=sdf_values)
```

### Option B: Use the Data Preparation Script

```bash
python examples/prepare_data.py \
    --input-dir meshes/ \
    --output-dir data/sdf_samples/ \
    --num-samples 100000
```

## 2. Train a Model

### Quick Training

```python
import torch
from deepsdf import DeepSDFDecoder, SDFDataset, Trainer
from deepsdf.utils import Config
from deepsdf.data import create_dataloader

# Create model
model = DeepSDFDecoder(latent_size=256)

# Load data
dataset = SDFDataset(data_dir='data/sdf_samples', split='train')
loader = create_dataloader(dataset, batch_size=32)

# Train
config = Config()
trainer = Trainer(model, loader, config=config)
trainer.train(num_epochs=100)
```

### Using the Training Script

```bash
python examples/train.py \
    --config deepsdf/configs/default_config.yaml \
    --data-dir data/sdf_samples/ \
    --output-dir output/
```

## 3. Reconstruct Shapes

### From a Trained Model

```python
from deepsdf import DeepSDFDecoder, ShapeReconstructor
import torch

# Load model
model = DeepSDFDecoder()
checkpoint = torch.load('output/my_experiment/checkpoints/latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create reconstructor
reconstructor = ShapeReconstructor(model)

# Reconstruct from a latent code
latent_code = torch.randn(256)  # Random latent code
mesh = reconstructor.reconstruct_mesh(latent_code, resolution=256)
reconstructor.save_mesh(mesh, 'output/reconstructed.obj')
```

### Using the Reconstruction Script

```bash
# Reconstruct from checkpoint
python examples/reconstruct.py \
    --checkpoint output/my_experiment/checkpoints/latest.pth \
    --num-random 5 \
    --resolution 256 \
    --output-dir reconstructions/
```

## 4. Monitor Training

Start TensorBoard to monitor training:

```bash
tensorboard --logdir output/my_experiment/logs/
```

Open your browser to http://localhost:6006

## Next Steps

- Read the [Training Guide](docs/TRAINING.md) for detailed training instructions
- Check the [API Documentation](docs/API.md) for complete API reference
- Explore the `examples/` directory for more usage examples

## Tips

1. **Start Small**: Begin with a small dataset (10-50 shapes) to test your pipeline
2. **Monitor Loss**: Watch the SDF loss - it should decrease steadily
3. **Adjust Learning Rates**: If training is unstable, reduce learning rates
4. **Increase Resolution**: For better reconstruction quality, increase marching cubes resolution
5. **Use GPU**: Training is much faster on GPU - set `device: cuda` in config

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `num_samples_per_shape` 
- Use smaller model with fewer hidden units

### Poor Reconstruction
- Train longer (more epochs)
- Increase `latent_size`
- Use more training data
- Reduce `latent_reg_lambda`

### Training Diverges
- Reduce `learning_rate` and `latent_lr`
- Increase `latent_reg_lambda`
- Check data quality

## Example Workflow

```bash
# 1. Prepare data
python examples/prepare_data.py \
    --input-dir meshes/ \
    --output-dir data/sdf_samples/

# 2. Train model
python examples/train.py \
    --data-dir data/sdf_samples/ \
    --output-dir output/

# 3. Reconstruct shapes
python examples/reconstruct.py \
    --checkpoint output/deepsdf_experiment/checkpoints/latest.pth \
    --num-random 10 \
    --output-dir reconstructions/

# 4. View results
# Open reconstructions/*.obj in a 3D viewer
```

Happy training! 🚀
