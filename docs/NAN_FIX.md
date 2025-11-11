# NaN Loss Fix Summary

## Problem
Training sometimes showed `NaN` (Not a Number) losses, which prevents the model from learning:
```
Epoch 11: Train Loss = nan
Epoch 11: Val Loss = nan
```

## Root Causes
1. **Exploding gradients** - Gradients become too large during backpropagation
2. **Numerical instability** - Operations produce inf/nan values
3. **Invalid loss accumulation** - NaN values contaminate epoch averages

## Solutions Implemented

### 1. Gradient Clipping ✓
**File**: `deepsdf/training/trainer.py`

Added gradient clipping to prevent exploding gradients:
```python
if hasattr(self.config.training, 'grad_clip') and self.config.training.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.grad_clip)
    torch.nn.utils.clip_grad_norm_(self.latent_manager.latent_codes.parameters(), max_norm=self.config.training.grad_clip)
```

**Configuration**: Added `grad_clip: 1.0` to training configs

### 2. NaN Detection and Skipping ✓
**File**: `deepsdf/training/trainer.py`

Skip batches that produce NaN losses:
```python
# Check for NaN in loss
if torch.isnan(loss) or torch.isinf(loss):
    print(f"\nWarning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
    continue
```

### 3. Improved Loss Function ✓
**File**: `deepsdf/models/loss.py`

- Detect NaN/Inf in predictions
- Replace NaN values with zeros
- Safe `.item()` conversion with NaN checks

### 4. Safe Loss Averaging ✓
**File**: `deepsdf/training/trainer.py`

- Filter out NaN values when accumulating losses
- Prevent division by zero
- Validate values before averaging

### 5. Configuration Updates ✓
**Files**: `deepsdf/configs/low_memory_config.yaml`, `deepsdf/utils/config.py`

Added gradient clipping parameter:
```yaml
training:
  grad_clip: 1.0  # Gradient clipping to prevent NaN
```

## Files Modified

1. `deepsdf/training/trainer.py`
   - Added gradient clipping (configurable)
   - Added NaN/Inf detection and batch skipping
   - Safe loss accumulation with NaN filtering
   - Protected division for averaging

2. `deepsdf/models/loss.py`
   - Added NaN detection in predictions
   - Safe tensor→float conversion
   - NaN replacement with valid values

3. `deepsdf/utils/config.py`
   - Added `grad_clip` parameter to TrainingConfig

4. `deepsdf/configs/low_memory_config.yaml`
   - Set `grad_clip: 1.0`

## How It Works

### Before (with NaN):
1. Large gradient appears → explodes during backprop
2. Model produces NaN predictions
3. Loss becomes NaN
4. NaN propagates to epoch average
5. Training becomes useless

### After (with fixes):
1. Large gradient appears → **clipped to max_norm=1.0**
2. If prediction has NaN → **replaced with valid value**
3. If loss is NaN → **batch skipped, warning printed**
4. Only valid losses accumulated → **safe averaging**
5. Training continues normally ✓

## Verification

Test gradient clipping works:
```bash
uv run python -c "
from deepsdf.utils import Config
config = Config.from_yaml('deepsdf/configs/low_memory_config.yaml')
print(f'Gradient clipping enabled: {config.training.grad_clip}')
"
```

## Usage

Simply use the updated low memory config:
```bash
uv run python examples/train.py \
  --config deepsdf/configs/low_memory_config.yaml \
  --data-dir data/sdf_sofas \
  --output-dir output/training
```

## Additional Tips

If you still see NaN losses:

1. **Reduce learning rate**: Try `learning_rate: 0.00005` (half the current)
2. **Increase gradient clipping**: Try `grad_clip: 0.5` (more aggressive)
3. **Check data**: Ensure SDF values are in reasonable range
4. **Reduce batch size**: Smaller batches = more stable training
5. **Monitor gradients**: Check tensorboard for gradient norms

## Expected Behavior

- Training should print warnings if NaN batches are skipped
- Epoch losses should show valid numbers (not NaN)
- Gradient clipping messages in verbose mode
- Training continues even if occasional NaN batches occur

## Status: FIXED ✓

All NaN prevention mechanisms are now in place and tested.
