# DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation

A modern, clean reimplementation of the [DeepSDF paper](https://arxiv.org/abs/1901.05103) with state-of-the-art Python coding practices.

## Features

- 🏗️ **Modular Architecture**: Clean separation of concerns with well-defined modules
- 📦 **Modern Python Packaging**: Using uv package manager with pyproject.toml
- 🎯 **Type Hints**: Full type annotations throughout the codebase
- 📚 **Comprehensive Documentation**: Detailed docstrings and usage examples
- ⚙️ **Configuration-Driven**: YAML/JSON configuration files for easy experimentation
- 🧪 **Testing Infrastructure**: Unit and integration tests
- 🎨 **Code Quality**: Black, Flake8, MyPy for consistent code style

## Installation

### Prerequisites

First, install [uv](https://github.com/astral-sh/uv) if you haven't already:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### From Source

```bash
git clone https://github.com/SimonBless/DeepSDF.git
cd DeepSDF
uv sync
```

### Development Installation

```bash
uv sync --all-extras
```

## Quick Start

### Training

```python
import torch
from deepsdf import DeepSDFDecoder, SDFDataset, Trainer
from deepsdf.utils import Config
from deepsdf.data import create_dataloader

# Load configuration
config = Config.from_yaml('deepsdf/configs/default_config.yaml')

# Create model
model = DeepSDFDecoder(
    latent_size=config.model.latent_size,
    hidden_dims=config.model.hidden_dims,
)

# Load data
train_dataset = SDFDataset(
    data_dir=config.data.data_dir,
    split=config.data.train_split,
)
train_loader = create_dataloader(
    train_dataset,
    batch_size=config.training.batch_size,
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    config=config,
)

# Train
trainer.train()
```

### Inference

```python
import torch
from deepsdf import DeepSDFDecoder, ShapeReconstructor

# Load trained model
model = DeepSDFDecoder(latent_size=256)
checkpoint = torch.load('output/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create reconstructor
reconstructor = ShapeReconstructor(model)

# Reconstruct shape from latent code
latent_code = torch.randn(256)
mesh = reconstructor.reconstruct_mesh(latent_code, resolution=256)

# Save mesh
reconstructor.save_mesh(mesh, 'output/reconstructed.obj')
```

### Data Preparation

```python
from deepsdf.data import create_sdf_samples
import numpy as np

# Generate SDF samples from a mesh
points, sdf_values = create_sdf_samples(
    mesh_path='path/to/mesh.obj',
    num_samples=100000,
)

# Save as .npz file
np.savez('data/shape_001.npz', points=points, sdf=sdf_values)
```

## Project Structure

```
DeepSDF/
├── deepsdf/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── decoder.py          # DeepSDF decoder architecture
│   │   └── loss.py             # Loss functions
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes
│   │   └── sdf_sampling.py     # SDF sampling utilities
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          # Training loop and management
│   ├── inference/
│   │   ├── __init__.py
│   │   └── reconstructor.py    # Mesh reconstruction
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration management
│   └── configs/
│       └── default_config.yaml # Default configuration
├── tests/
│   └── ...                      # Unit and integration tests
├── setup.py
├── pyproject.toml
├── pyproject.toml
└── README.md
```

## Configuration

All training parameters can be configured via YAML files. See `deepsdf/configs/default_config.yaml` for the default configuration.

Key configuration options:

- **Model**: Latent size, hidden dimensions, dropout, normalization
- **Training**: Batch size, learning rates, regularization, checkpointing
- **Data**: Data directories, splits, number of samples

## Model Architecture

The DeepSDF decoder is an 8-layer MLP with:
- 512 hidden units per layer
- Skip connections at layer 4
- Layer normalization
- Dropout for regularization
- Weight normalization

Input: Concatenation of latent code (256-dim) and 3D coordinates (3-dim)
Output: Signed distance value (1-dim)

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run black deepsdf/
```

### Type Checking

```bash
uv run mypy deepsdf/
```

### Linting

```bash
uv run flake8 deepsdf/
```

## Citation

If you use this code in your research, please cite the original DeepSDF paper:

```bibtex
@inproceedings{park2019deepsdf,
  title={DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation},
  author={Park, Jeong Joon and Florence, Peter and Straub, Julian and Newcombe, Richard and Lovegrove, Steven},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={165--174},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original DeepSDF paper and implementation
- PyTorch team for the deep learning framework
- Trimesh for mesh processing utilities