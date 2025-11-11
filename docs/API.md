# API Documentation

Complete API reference for DeepSDF.

## Models

### DeepSDFDecoder

The core decoder network that maps latent codes and 3D coordinates to SDF values.

```python
class DeepSDFDecoder(nn.Module):
    def __init__(
        self,
        latent_size: int = 256,
        hidden_dims: Optional[List[int]] = None,
        dropout_prob: float = 0.2,
        norm_layers: Optional[List[int]] = None,
        latent_in: Optional[List[int]] = None,
        weight_norm: bool = True,
    )
```

**Parameters:**
- `latent_size`: Dimension of latent code (default: 256)
- `hidden_dims`: Hidden layer dimensions (default: [512]*8)
- `dropout_prob`: Dropout probability (default: 0.2)
- `norm_layers`: Indices of layers with normalization
- `latent_in`: Indices where latent code is concatenated
- `weight_norm`: Use weight normalization

**Methods:**

```python
def forward(
    self,
    latent_vector: torch.Tensor,  # (batch_size, latent_size)
    xyz: torch.Tensor,              # (batch_size, num_points, 3)
) -> torch.Tensor:                  # (batch_size, num_points, 1)
    """Predict SDF values."""
```

```python
def inference(
    self,
    latent_vector: torch.Tensor,
    xyz: torch.Tensor,
) -> torch.Tensor:
    """Inference mode (no dropout)."""
```

### DeepSDFLoss

Combined loss function with SDF prediction and latent regularization.

```python
class DeepSDFLoss(nn.Module):
    def __init__(
        self,
        latent_reg_lambda: float = 1e-4,
        clamp_dist: float = 0.1,
    )
```

**Methods:**

```python
def forward(
    self,
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    latent_codes: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute loss and return loss components."""
```

## Data

### SDFDataset

Dataset for loading pre-computed SDF samples.

```python
class SDFDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_samples_per_shape: int = 10000,
        transform: Optional[Callable] = None,
    )
```

**Parameters:**
- `data_dir`: Directory containing `.npz` sample files
- `split`: Dataset split ('train', 'val', 'test')
- `num_samples_per_shape`: Number of points per shape
- `transform`: Optional data transform

**Methods:**

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    Returns:
        Dictionary with:
        - 'points': (num_samples, 3)
        - 'sdf': (num_samples, 1)
        - 'shape_id': str
    """
```

### Data Utilities

```python
def sample_sdf_from_mesh(
    mesh: trimesh.Trimesh,
    num_samples: int = 10000,
    variance: float = 0.05,
    sample_near_surface: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points and SDF values from mesh."""
```

```python
def normalize_mesh(
    mesh: trimesh.Trimesh,
    scale: float = 1.0,
) -> trimesh.Trimesh:
    """Normalize mesh to unit sphere."""
```

```python
def create_sdf_samples(
    mesh_path: str,
    num_samples: int = 10000,
    variance: float = 0.05,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create SDF samples from mesh file."""
```

## Training

### Trainer

Main training class with built-in latent code optimization.

```python
class Trainer:
    def __init__(
        self,
        model: DeepSDFDecoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Config] = None,
        device: str = 'cuda',
    )
```

**Methods:**

```python
def train(self, num_epochs: Optional[int] = None) -> None:
    """Train the model."""

def train_epoch(self) -> Dict[str, float]:
    """Train for one epoch."""

def validate(self) -> Dict[str, float]:
    """Validate the model."""

def save_checkpoint(self, filename: Optional[str] = None) -> None:
    """Save checkpoint."""

def load_checkpoint(self, checkpoint_path: str) -> None:
    """Load checkpoint."""
```

### LatentCodeManager

Manages learnable latent codes for shapes.

```python
class LatentCodeManager:
    def __init__(
        self,
        num_shapes: int,
        latent_size: int,
        device: str = 'cuda',
    )
```

**Methods:**

```python
def get_codes(self, indices: torch.Tensor) -> torch.Tensor:
    """Get latent codes for shape indices."""
```

## Inference

### ShapeReconstructor

Reconstruct 3D meshes from latent codes.

```python
class ShapeReconstructor:
    def __init__(
        self,
        model: DeepSDFDecoder,
        device: str = 'cuda',
    )
```

**Methods:**

```python
def reconstruct_mesh(
    self,
    latent_code: torch.Tensor,
    resolution: int = 256,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    level: float = 0.0,
) -> trimesh.Trimesh:
    """Reconstruct mesh using marching cubes."""
```

```python
def reconstruct_batch(
    self,
    latent_codes: torch.Tensor,
    resolution: int = 256,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> list[trimesh.Trimesh]:
    """Reconstruct multiple meshes."""
```

```python
def save_mesh(
    self,
    mesh: trimesh.Trimesh,
    output_path: str,
) -> None:
    """Save mesh to file."""
```

## Configuration

### Config

Configuration dataclass for DeepSDF.

```python
@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str = "deepsdf_experiment"
    output_dir: str = "./output"
```

**Methods:**

```python
@classmethod
def from_yaml(cls, path: str) -> "Config":
    """Load from YAML file."""

@classmethod
def from_json(cls, path: str) -> "Config":
    """Load from JSON file."""

def to_yaml(self, path: str) -> None:
    """Save to YAML file."""

def to_json(self, path: str) -> None:
    """Save to JSON file."""
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    latent_size: int = 256
    hidden_dims: list[int] = field(default_factory=lambda: [512]*8)
    dropout_prob: float = 0.2
    norm_layers: list[int] = field(default_factory=lambda: [0,1,2,3,4,5,6,7])
    latent_in: list[int] = field(default_factory=lambda: [4])
    weight_norm: bool = True
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 2000
    learning_rate: float = 1e-4
    latent_lr: float = 1e-3
    latent_reg_lambda: float = 1e-4
    clamp_dist: float = 0.1
    num_samples_per_shape: int = 10000
    checkpoint_every: int = 100
    log_every: int = 10
    num_workers: int = 4
    device: str = "cuda"
```

### DataConfig

```python
@dataclass
class DataConfig:
    data_dir: str = "./data"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
```

## Example Usage

### Complete Training Pipeline

```python
from deepsdf import DeepSDFDecoder, SDFDataset, Trainer
from deepsdf.utils import Config
from deepsdf.data import create_dataloader

# Configuration
config = Config.from_yaml('config.yaml')

# Model
model = DeepSDFDecoder(latent_size=config.model.latent_size)

# Data
dataset = SDFDataset(data_dir=config.data.data_dir, split='train')
loader = create_dataloader(dataset, batch_size=config.training.batch_size)

# Training
trainer = Trainer(model, loader, config=config)
trainer.train()
```

### Reconstruction

```python
from deepsdf import DeepSDFDecoder, ShapeReconstructor
import torch

# Load model
model = DeepSDFDecoder()
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Reconstruct
reconstructor = ShapeReconstructor(model)
latent = torch.randn(256)
mesh = reconstructor.reconstruct_mesh(latent, resolution=256)
reconstructor.save_mesh(mesh, 'output.obj')
```
