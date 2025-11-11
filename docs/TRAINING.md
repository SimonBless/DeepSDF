# Training Guide

This guide provides detailed instructions for training a DeepSDF model.

## Data Preparation

Before training, you need to prepare SDF samples from your 3D meshes.

### Step 1: Organize Your Meshes

Place your mesh files (`.obj`, `.ply`, `.off`, `.stl`) in a directory:

```
meshes/
├── shape_001.obj
├── shape_002.obj
└── ...
```

### Step 2: Generate SDF Samples

Use the provided data preparation script:

```bash
python examples/prepare_data.py \
    --input-dir meshes/ \
    --output-dir data/sdf_samples/ \
    --num-samples 100000 \
    --variance 0.05
```

This will:
- Sample 100,000 points per mesh
- Compute SDF values at each point
- Save samples as `.npz` files
- Create train/val/test splits

## Configuration

Create or modify a configuration file (e.g., `my_config.yaml`):

```yaml
experiment_name: my_experiment
output_dir: ./output

model:
  latent_size: 256
  hidden_dims: [512, 512, 512, 512, 512, 512, 512, 512]
  dropout_prob: 0.2
  weight_norm: true

training:
  batch_size: 32
  num_epochs: 2000
  learning_rate: 0.0001
  latent_lr: 0.001
  latent_reg_lambda: 0.0001
  clamp_dist: 0.1
  checkpoint_every: 100
  device: cuda

data:
  data_dir: ./data/sdf_samples
  train_split: train
  val_split: val
```

## Training

### Basic Training

```bash
python examples/train.py \
    --config my_config.yaml \
    --data-dir data/sdf_samples/ \
    --output-dir output/
```

### Resume from Checkpoint

```bash
python examples/train.py \
    --config my_config.yaml \
    --data-dir data/sdf_samples/ \
    --output-dir output/ \
    --resume output/my_experiment/checkpoints/latest.pth
```

## Monitoring Training

### TensorBoard

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir output/my_experiment/logs/
```

Open your browser to `http://localhost:6006` to view:
- Training and validation losses
- SDF prediction metrics
- Latent code regularization

### Checkpoints

Checkpoints are saved to `output/my_experiment/checkpoints/`:
- `latest.pth`: Most recent checkpoint
- `checkpoint_epoch_N.pth`: Checkpoint at epoch N

Each checkpoint contains:
- Model weights
- Latent codes
- Optimizer states
- Training configuration

## Hyperparameter Tuning

### Key Hyperparameters

1. **Latent Size** (`latent_size`): Dimension of shape latent codes
   - Default: 256
   - Larger values = more expressive but slower

2. **Learning Rates**:
   - `learning_rate`: For decoder network (default: 1e-4)
   - `latent_lr`: For latent codes (default: 1e-3)

3. **Regularization** (`latent_reg_lambda`): Controls latent code magnitude
   - Default: 1e-4
   - Increase to prevent overfitting

4. **SDF Clamping** (`clamp_dist`): Maximum SDF distance
   - Default: 0.1
   - Focus learning on near-surface regions

### Recommended Settings

**For small datasets (<100 shapes):**
```yaml
model:
  latent_size: 128
training:
  batch_size: 16
  learning_rate: 0.0001
  latent_reg_lambda: 0.0005
```

**For large datasets (>1000 shapes):**
```yaml
model:
  latent_size: 256
training:
  batch_size: 64
  learning_rate: 0.0005
  latent_reg_lambda: 0.0001
```

## Custom Training Script

For more control, create a custom training script:

```python
import torch
from deepsdf import DeepSDFDecoder, SDFDataset, Trainer
from deepsdf.utils import Config
from deepsdf.data import create_dataloader

# Load config
config = Config.from_yaml('my_config.yaml')

# Create model with custom architecture
model = DeepSDFDecoder(
    latent_size=256,
    hidden_dims=[512] * 8,
    dropout_prob=0.2,
)

# Load dataset
train_dataset = SDFDataset(
    data_dir='data/sdf_samples',
    split='train',
    num_samples_per_shape=10000,
)

train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    config=config,
)

# Train
trainer.train(num_epochs=2000)
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce `num_samples_per_shape`
- Use gradient accumulation

### Loss Not Decreasing

- Check data quality (visualize SDF samples)
- Increase learning rate
- Reduce `latent_reg_lambda`

### Unstable Training

- Reduce learning rates
- Increase `latent_reg_lambda`
- Use gradient clipping

### Poor Reconstruction Quality

- Train longer (more epochs)
- Increase `latent_size`
- Increase `num_samples_per_shape`
- Improve data quality
