# GPU Requirements for MRL Training

**Your Hardware**: NVIDIA RTX 5060 Ti 16GB
**Verdict**: ✅ **Perfect for this task!**

---

## Quick Answer

**16GB VRAM is MORE than sufficient** for training MRL-ReDimNet with recommended settings.

**Recommended batch size**: 32-48 (leaves headroom)
**Maximum batch size**: 64-80 (will use ~12-14GB)

---

## 1. Memory Breakdown

### Model Parameters (ReDimNet-b2)

| Component | Size | Memory |
|-----------|------|--------|
| Model weights | 4.7M params | ~19 MB |
| Gradients | 4.7M params | ~19 MB |
| Optimizer states (AdamW) | 2x params | ~38 MB |
| **Total (model)** | | **~76 MB** |

**Conclusion**: Model itself is tiny! Memory is dominated by batch data.

---

### Training Batch Memory

**For batch_size = 32** (recommended):

```
Audio input:
  - Shape: [32, 1, 48000]
  - Size: 32 × 48000 × 4 bytes = 6 MB

MelBanks features:
  - Shape: [32, 72, 300]  (72 freq bins, ~300 frames)
  - Size: 32 × 72 × 300 × 4 = 2.8 MB

Backbone activations (6 stages):
  - Stage outputs accumulate: ~500 MB - 1 GB

Projection head:
  - Pooled features: [32, 1024] = 0.1 MB
  - MRL outputs (4 dims): [32, 64+128+192+256] = ~0.3 MB

Loss computation:
  - Embeddings + logits: ~50-100 MB

Total: ~2-3 GB for batch_size=32
```

**Memory scaling**:
- Batch size 16: ~1.5 GB
- Batch size 32: ~2.5 GB ✅ **Recommended**
- Batch size 48: ~4 GB
- Batch size 64: ~5-6 GB
- Batch size 128: ~10-12 GB

---

## 2. Memory Usage by Training Stage

### Stage 1: Projection Head Training (Frozen Backbone)

**Memory**: ~2-4 GB (batch_size=32)

```
Model weights: 76 MB (only projection trainable)
Batch data: 2-3 GB
Cached backbone outputs: Minimal (backbone frozen)

Total: ~3-4 GB
```

**Your 16GB**: ✅ Plenty of headroom - can use batch_size=64 easily

### Stage 2: Full Model Fine-tuning

**Memory**: ~4-6 GB (batch_size=32)

```
Model weights: 76 MB (all trainable)
Gradients: Entire model
Batch data: 2-3 GB
Backbone activations: Need to store for backprop

Total: ~5-6 GB
```

**Your 16GB**: ✅ Still plenty - can use batch_size=48-64

---

## 3. Recommended Settings for RTX 5060 Ti 16GB

### Conservative (Safe, Efficient)

```yaml
training:
  batch_size: 32
  accumulation_steps: 1

hardware:
  mixed_precision: true  # Reduces memory by ~30%
```

**Expected memory usage**: ~4-5 GB
**Training speed**: ~100-120 iterations/second
**Utilization**: ~30-40% of your VRAM

### Balanced (Recommended)

```yaml
training:
  batch_size: 48
  accumulation_steps: 1

hardware:
  mixed_precision: true
```

**Expected memory usage**: ~6-8 GB
**Training speed**: ~80-100 iterations/second
**Utilization**: ~40-50% of your VRAM

### Aggressive (Maximum Speed)

```yaml
training:
  batch_size: 64
  accumulation_steps: 1

hardware:
  mixed_precision: true
```

**Expected memory usage**: ~8-10 GB
**Training speed**: ~60-80 iterations/second
**Utilization**: ~50-65% of your VRAM

### Ultra (Maximum Batch Size)

```yaml
training:
  batch_size: 96
  accumulation_steps: 1

hardware:
  mixed_precision: true
  compile: true  # PyTorch 2.0+ optimization
```

**Expected memory usage**: ~12-14 GB
**Training speed**: ~40-50 iterations/second
**Utilization**: ~75-85% of your VRAM

---

## 4. Memory Optimization Techniques

### 1. Mixed Precision Training (Automatic)

**Already included in config**:
```yaml
hardware:
  mixed_precision: true
```

**Benefit**: Reduces memory by ~30-40%
**Impact**: No accuracy loss
**Speed**: 1.5-2x faster training

**How it works**: Uses FP16 for most operations, FP32 for stability

### 2. Gradient Accumulation

If you want effective batch_size=128 but don't have memory:

```yaml
training:
  batch_size: 32          # Fit in memory
  accumulation_steps: 4   # Effective batch = 32×4 = 128
```

**Memory**: Same as batch_size=32
**Effective training**: Same as batch_size=128
**Trade-off**: 4x slower

### 3. Gradient Checkpointing (If Needed)

For extreme memory constraint (not needed for you):

```python
# Enable in model initialization
model = ReDimNetMRL(
    ...,
    use_checkpoint=True  # Trades compute for memory
)
```

**Benefit**: ~40% less memory
**Cost**: ~30% slower training

---

## 5. Actual Memory Test

Let me give you exact commands to test memory usage:

### Test 1: Model Size

```python
import torch
from mrl import create_mrl_from_pretrained

# Load model
model = create_mrl_from_pretrained(
    model_name='b2',
    train_type='ptn',
    embed_dim=256,
    mrl_dims=[64, 128, 192, 256],
    device='cuda'
)

# Check memory
print(f"Model memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
# Expected: ~100-200 MB
```

### Test 2: Forward Pass

```python
# Create dummy batch
batch_size = 32
audio = torch.randn(batch_size, 1, 48000).cuda()

# Forward pass
torch.cuda.reset_peak_memory_stats()
embeddings = model(audio, return_all_dims=True)

peak_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak memory (batch={batch_size}): {peak_mem:.2f} MB")
# Expected: ~500-1000 MB
```

### Test 3: Training Step

```python
from mrl.losses import MatryoshkaLoss, AAMSoftmax

# Create loss
loss_fn = MatryoshkaLoss(
    base_loss=AAMSoftmax(256, 5994).cuda(),
    mrl_dims=[64, 128, 192, 256]
)

# Training step
torch.cuda.reset_peak_memory_stats()
labels = torch.randint(0, 5994, (batch_size,)).cuda()
loss, _ = loss_fn(embeddings, labels)
loss.backward()

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak memory (training, batch={batch_size}): {peak_mem:.2f} GB")
# Expected: ~2-3 GB for batch=32
```

---

## 6. Training Time Estimates (Single RTX 5060 Ti)

### With Recommended Settings (batch_size=32)

**VoxCeleb2** (5,994 speakers, 1M utterances):
- Iterations per epoch: ~34,000 (1M / 32)
- Time per iteration: ~0.5 seconds
- Time per epoch: ~4.7 hours

**Stage 1** (5 epochs):
- Total: ~23 hours (~1 day)

**Stage 2** (50 epochs):
- Total: ~235 hours (~10 days)

**Complete training**: ~11 days

### With Larger Batch (batch_size=64)

- Time per epoch: ~3 hours
- Stage 1 (5 epochs): ~15 hours
- Stage 2 (50 epochs): ~150 hours (~6 days)
- **Complete training**: ~7 days

---

## 7. Comparison: Your GPU vs Others

| GPU | VRAM | Batch Size | Training Time | Your GPU |
|-----|------|------------|---------------|----------|
| RTX 3060 | 12GB | 24 | 14 days | |
| **RTX 5060 Ti** | **16GB** | **48-64** | **7-10 days** | ✅ **You have this** |
| RTX 3090 | 24GB | 96 | 5 days | |
| A100 40GB | 40GB | 128 | 3 days | |
| A100 80GB | 80GB | 256 | 2 days | |

**Your position**: Middle-high tier, very capable!

---

## 8. Multi-GPU Support (Future Expansion)

Your current setup: 1× RTX 5060 Ti 16GB

**If you add more GPUs**:

```yaml
hardware:
  distributed: true
  world_size: 2  # Number of GPUs
```

**With 2× RTX 5060 Ti**:
- Effective batch: 64×2 = 128
- Training time: ~3-4 days
- Memory per GPU: Same (distributed)

---

## 9. Memory Monitoring During Training

### Built-in Logging

Our trainer already logs GPU memory:

```python
# In train.py (already included)
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / "
      f"{torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
```

### External Monitoring

**Option 1: nvidia-smi**
```bash
watch -n 1 nvidia-smi
```

**Option 2: nvtop** (better)
```bash
sudo apt install nvtop
nvtop
```

**Option 3: TensorBoard**
```python
# Memory usage logged automatically
tensorboard --logdir logs/mrl_redimnet
```

---

## 10. Troubleshooting

### Out of Memory (OOM)

**If you see**: `RuntimeError: CUDA out of memory`

**Solutions** (in order):

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 24  # Down from 32
   ```

2. **Enable gradient accumulation**:
   ```yaml
   training:
     batch_size: 16
     accumulation_steps: 2  # Effective: 32
   ```

3. **Clear cache** (add to train.py):
   ```python
   torch.cuda.empty_cache()  # After each epoch
   ```

4. **Enable gradient checkpointing**:
   ```yaml
   advanced:
     gradient_checkpointing: true
   ```

### Memory Leaks

**Symptoms**: Memory usage grows over time

**Solution**:
```python
# Explicitly delete intermediate tensors
del embeddings, loss
torch.cuda.empty_cache()
```

---

## 11. Optimal Configuration for Your GPU

**Create this config**: `config_5060ti.yaml`

```yaml
# Optimized for RTX 5060 Ti 16GB
model:
  embed_dim: 256
  mrl_dims: [64, 128, 192, 256]
  F: 72
  C: 12
  out_channels: 512

training:
  batch_size: 48              # Sweet spot for 16GB
  accumulation_steps: 1
  num_epochs: 100
  learning_rate: 0.0001
  max_grad_norm: 1.0

hardware:
  device: 'cuda:0'
  mixed_precision: true       # Essential for efficiency
  compile: false              # Set true if PyTorch 2.0+

data:
  num_workers: 8              # Matches typical CPU cores
  pin_memory: true
  prefetch_factor: 2

advanced:
  use_pretrained: true
  model_name: 'b2'
  train_type: 'ft_lm'
  freeze_backbone_epochs: 5
```

---

## 12. Expected Training Experience

### Starting Training

```bash
python train.py --config config_5060ti.yaml
```

**You'll see**:
```
Loading pretrained ReDimNet-b2 (ft_lm, vox2)...
✅ Successfully loaded pretrained model
   Embedding dimension: 192
   Total parameters: 4,700,000

Transferring weights from pretrained model...
   ✅ Backbone: 425 layers transferred
   🆕 MRL projection: Randomly initialized

Model parameters: 4,812,000 (trainable: 524,288)

GPU Memory: 0.35GB / 0.35GB

Epoch 1/100
  [Stage 1: Backbone Frozen]
  Train Loss: 2.345
  GPU Memory: 4.2GB / 5.8GB peak
  Time: 4.3 hours

Epoch 5/100
  Train Loss: 1.234
  Val Loss: 1.456
  ✅ Saved best model (val_loss: 1.456)

Epoch 6/100
  [Stage 2: Unfreezing backbone]
  🔓 All parameters trainable
  Train Loss: 1.123
  GPU Memory: 5.1GB / 7.2GB peak
  Time: 4.8 hours
```

---

## 13. Summary

### Your Setup: RTX 5060 Ti 16GB

✅ **Excellent** for this task
✅ **Recommended batch size**: 48
✅ **Expected memory usage**: 5-8GB (50% utilization)
✅ **Training time**: ~7-10 days
✅ **No memory optimization needed**

### What You Can Do

- ✅ Train full MRL model with all features
- ✅ Use pretrained models (b0-b6)
- ✅ Run with comfortable batch sizes
- ✅ Enable mixed precision for speed
- ✅ Have headroom for experimentation

### What You Can't Do (Limitations)

- ⚠️ Very large batch sizes (128+) without accumulation
- ⚠️ Multiple large models in memory simultaneously
- ⚠️ Train larger models (b6) with maximum batch size

**But none of these matter for standard MRL training!**

---

## 14. Quick Test

Before starting full training, test your setup:

```bash
cd mrl

# Quick memory test
python << 'PYEOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
print(f"Current usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

# Test model loading
from mrl import create_mrl_from_pretrained
model = create_mrl_from_pretrained('b2', 'ptn', 'vox2', device='cuda')
print(f"Model loaded: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

# Test forward pass
audio = torch.randn(32, 1, 48000).cuda()
emb = model(audio, return_all_dims=True)
print(f"After forward: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
print("✅ GPU test passed!")
PYEOF
```

---

**Conclusion**: Your RTX 5060 Ti 16GB is **perfect** for this project. You have plenty of VRAM and will have a smooth training experience!

Start training with: `python train.py --config config.yaml`
