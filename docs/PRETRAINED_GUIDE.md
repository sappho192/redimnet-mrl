# Using Pretrained ReDimNet Models with MRL

This guide shows how to leverage official pretrained ReDimNet models for MRL training.

## Why Use Pretrained Models?

✅ **Faster convergence** - Start from speaker-discriminative features
✅ **Better performance** - Pretrained models achieve 0.835% EER on VoxCeleb1-O
✅ **Less data needed** - Transfer learning requires less training data
✅ **Flexible deployment** - Same pretrained backbone, multiple MRL resolutions

## Available Pretrained Models

### Model Variants

| Variant | Parameters | VoxCeleb1-O EER | Speed | Use Case |
|---------|-----------|-----------------|-------|----------|
| b0 | 1.0M | 1.16% | Fastest | Edge devices, IoT |
| b1 | 2.2M | 0.85% | Very fast | Mobile apps |
| b2 | 4.7M | 0.57% | Balanced | **Recommended** |
| b3 | 3.0M | 0.50% | Fast | Production |
| b5 | 9.2M | 0.43% | Accurate | High-accuracy |
| b6 | 15.0M | 0.40% | Most accurate | Research |
| M | ~6M | 0.835% | Balanced | Common baseline |

### Training Types

- **`ptn`**: Pretrained on VoxCeleb2 (no finetuning)
- **`ft_lm`**: Finetuned with Large-Margin loss (better performance)
- **`ft_mix`**: Finetuned on mixed datasets (VoxBlink2 + VoxCeleb2 + CN-Celeb)

## Quick Start

### 1. Basic Usage - Load Pretrained Model

```python
from mrl import load_pretrained_redimnet

# Load official pretrained model
model = load_pretrained_redimnet(
    model_name='b2',
    train_type='ft_lm',  # Use fine-tuned for best starting point
    dataset='vox2'
)

# Use for inference
audio = load_audio('speaker.wav')
embedding = model(audio)  # 192D embedding
```

### 2. Create MRL from Pretrained

```python
from mrl import create_mrl_from_pretrained

# Convert pretrained model to MRL
mrl_model = create_mrl_from_pretrained(
    model_name='b2',
    train_type='ft_lm',
    dataset='vox2',
    embed_dim=256,
    mrl_dims=[64, 128, 192, 256],
    freeze_backbone=False  # Full model trainable
)

# Now you have multi-resolution support!
emb_64d = mrl_model(audio, target_dim=64)   # Fast
emb_256d = mrl_model(audio, target_dim=256)  # Accurate
```

### 3. Two-Stage Training (Recommended)

The best strategy is to train in two stages:

**Stage 1**: Train MRL projection head only (5-10 epochs)
**Stage 2**: Fine-tune entire model

```python
from mrl import create_mrl_from_pretrained, unfreeze_backbone

# Stage 1: Freeze backbone, train projection
model = create_mrl_from_pretrained(
    model_name='b2',
    train_type='ft_lm',
    embed_dim=256,
    mrl_dims=[64, 128, 192, 256],
    freeze_backbone=True  # ⚠️ Frozen
)

# Train for 5 epochs
train_projection_head(model, train_loader, num_epochs=5)

# Stage 2: Unfreeze and fine-tune
unfreeze_backbone(model)
train_full_model(model, train_loader, num_epochs=50)
```

## Configuration for Training

Update your `config.yaml`:

```yaml
# Advanced section
advanced:
  # Enable pretrained model loading
  use_pretrained: true
  model_name: 'b2'          # Choose variant
  train_type: 'ft_lm'       # Use fine-tuned
  pretrained_dataset: 'vox2'

  # Two-stage training
  freeze_backbone_epochs: 5  # Stage 1 duration
```

Then train normally:

```bash
python train.py --config config.yaml
```

The trainer will automatically:
1. Load pretrained weights from torch.hub
2. Train projection head for 5 epochs (Stage 1)
3. Unfreeze backbone at epoch 5 (Stage 2)
4. Continue training entire model

## Model Selection Guide

### For Maximum Accuracy
```python
model = create_mrl_from_pretrained(
    model_name='b5',      # or b6 for even better
    train_type='ft_lm',   # Fine-tuned
    embed_dim=256
)
```

### For Balanced Performance (Recommended)
```python
model = create_mrl_from_pretrained(
    model_name='b2',      # Best balance
    train_type='ft_lm',
    embed_dim=256
)
```

### For Edge Deployment
```python
model = create_mrl_from_pretrained(
    model_name='b0',      # Smallest, fastest
    train_type='ft_lm',
    embed_dim=128,        # Smaller max dimension
    mrl_dims=[32, 64, 128]
)
```

### For Cross-Domain Robustness
```python
model = create_mrl_from_pretrained(
    model_name='b3',
    train_type='ft_mix',  # Trained on multiple datasets
    dataset='vb2+vox2+cnc',
    embed_dim=256
)
```

## Training Strategies

### Strategy 1: Quick Fine-tuning (Fastest)

```yaml
training:
  num_epochs: 20  # Short training

advanced:
  use_pretrained: true
  freeze_backbone_epochs: 5  # Quick projection training
```

**When to use**: Limited compute, small dataset
**Expected time**: ~1-2 days on single GPU

### Strategy 2: Full Fine-tuning (Best Quality)

```yaml
training:
  num_epochs: 100  # Full training

advanced:
  use_pretrained: true
  freeze_backbone_epochs: 10  # Longer projection training
```

**When to use**: Large dataset, ample compute
**Expected time**: ~1 week on 4 GPUs

### Strategy 3: From Scratch (Baseline)

```yaml
advanced:
  use_pretrained: false  # No pretrained weights
```

**When to use**: Novel domain, research comparison
**Expected time**: ~2-3 weeks on 4 GPUs

## Performance Expectations

Based on pretrained b2 + MRL training:

| Stage | EER (VoxCeleb1-O) | Training Time |
|-------|-------------------|---------------|
| Pretrained b2 (192D) | 0.57% | - |
| After Stage 1 (5 epochs) | 0.60% | 1 day |
| After Stage 2 (50 epochs) | 0.50% | 1 week |

Multi-resolution performance (after training):

| Dimension | Target EER | vs 256D |
|-----------|------------|---------|
| 256D | 0.50% | 100% (baseline) |
| 192D | 0.53% | 94% |
| 128D | 0.58% | 86% |
| 64D | 0.65% | 77% |

## Troubleshooting

### Problem: `torch.hub.load()` fails

**Solution**: Ensure internet connection and try:
```python
torch.hub.set_dir('/path/to/cache')  # Set cache directory
model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet',
                       model_name='b2', force_reload=True)
```

### Problem: Out of memory during training

**Solution 1**: Use smaller batch size
```yaml
training:
  batch_size: 32  # Reduce from 64
  accumulation_steps: 2  # Maintain effective batch size
```

**Solution 2**: Use smaller model
```yaml
advanced:
  model_name: 'b2'  # Instead of b5
```

### Problem: Backbone not unfreezing

**Solution**: Check epoch counter in logs:
```
Epoch 5: Unfreezing backbone (Stage 1 → Stage 2)
```

If missing, verify config:
```yaml
advanced:
  freeze_backbone_epochs: 5  # Must be > 0
```

## Examples

See `example_pretrained.py` for complete runnable examples:

```bash
cd mrl
python example_pretrained.py
```

## References

- [Official ReDimNet Repository](https://github.com/IDRnD/redimnet)
- [ReDimNet Paper](https://arxiv.org/abs/2407.18223)
- [torch.hub Documentation](https://pytorch.org/docs/stable/hub.html)

## Next Steps

1. ✅ Load a pretrained model: `python example_pretrained.py`
2. ✅ Configure training: Edit `config.yaml`
3. ✅ Start training: `python train.py --config config.yaml`
4. ✅ Evaluate: Check multi-dimension EER

For questions, see [README.md](README.md) or open an issue.
