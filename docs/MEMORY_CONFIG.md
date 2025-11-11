# Memory Configuration Guide for DeepSDF Training

## CUDA Out of Memory Error - Solution

The CUDA out of memory error occurs when the GPU doesn't have enough VRAM to handle the training batch. The memory usage is primarily determined by:

1. **Batch Size**: Number of shapes processed simultaneously
2. **Samples Per Shape**: Number of 3D points sampled from each shape
3. **Model Size**: Number and size of hidden layers

### Memory Calculation

**Total points per batch = `batch_size × num_samples_per_shape`**

Example configurations:
- Default: `32 × 10,000 = 320,000 points/batch` → **Requires ~10-12GB VRAM**
- Low memory: `4 × 2,048 = 8,192 points/batch` → **Requires ~4-6GB VRAM**
- Ultra low: `2 × 1,024 = 2,048 points/batch` → **Requires ~2-3GB VRAM**

## Available Configurations

### 1. Default Config (`default_config.yaml`)
```yaml
batch_size: 32
num_samples_per_shape: 10000
```
**Best for**: GPUs with 10GB+ VRAM (e.g., RTX 3080, RTX 4090, A100)

### 2. Low Memory Config (`low_memory_config.yaml`)
```yaml
batch_size: 4
num_samples_per_shape: 2048
```
**Best for**: GPUs with 6GB VRAM (e.g., RTX 2060, GTX 1660 Ti)
**Usage**: `--config deepsdf/configs/low_memory_config.yaml`

### 3. Ultra Low Memory Config (`ultra_low_memory_config.yaml`)
```yaml
batch_size: 2
num_samples_per_shape: 1024
```
**Best for**: GPUs with 4GB VRAM or less, or for testing
**Usage**: `--config deepsdf/configs/ultra_low_memory_config.yaml`

## Training Commands

### For 6GB GPU (Recommended for your setup)
```bash
uv run python examples/train.py \
  --config deepsdf/configs/low_memory_config.yaml \
  --data-dir data/sdf_sofas \
  --output-dir output/deepsdf_training
```

### For 4GB GPU or testing
```bash
uv run python examples/train.py \
  --config deepsdf/configs/ultra_low_memory_config.yaml \
  --data-dir data/sdf_sofas \
  --output-dir output/deepsdf_training
```

## Additional Memory Saving Tips

1. **Reduce model size**: Use fewer or smaller hidden layers
2. **Gradient accumulation**: Simulate larger batches without more memory (requires code modification)
3. **Mixed precision training**: Use FP16 instead of FP32 (requires code modification)
4. **Use CPU**: Add `device: cpu` in config (much slower but no memory limit)

## Troubleshooting

### Still getting OOM errors?
- Reduce `batch_size` further (try 1)
- Reduce `num_samples_per_shape` (try 512)
- Check GPU memory: `nvidia-smi`
- Close other GPU processes
- Try CPU training temporarily

### Training too slow?
- Increase `batch_size` gradually until you hit memory limits
- Increase `num_workers` for faster data loading (if CPU/RAM allows)
- Ensure data is on SSD, not HDD
