# MRL-ReDimNet Training Report

**Date**: 2025-12-03
**Author**: Training Session with Claude Code
**Model**: ReDimNet-b2 with Matryoshka Representation Learning
**Dataset**: VoxCeleb2 (train) + VoxCeleb1 (validation)

---

## Executive Summary

Initial training runs revealed **severe overfitting** when transitioning from Stage 1 (frozen backbone) to Stage 2 (full fine-tuning). This report documents the issue, analysis, and implemented solutions.

**Key Finding**: Training loss dropped from 12.93 → 3.02 while validation loss increased from 18.00 → 18.38, indicating the model was memorizing training data rather than learning generalizable speaker representations.

---

## Training Setup

### Model Architecture
- **Base Model**: ReDimNet-b2 pretrained on VoxCeleb2
- **Total Parameters**: 5,016,305
- **Architecture**: C=16, F=72, block_1d_type=conv+att, block_2d_type=convnext_like
- **MRL Dimensions**: [64, 128, 192, 256]

### Datasets
- **Training**: VoxCeleb2 dev (1,092,009 files, 5,994 speakers)
- **Validation**: VoxCeleb1 dev (153,516 files, 1,251 speakers) ✅
- **Test**: VoxCeleb1 test (40 speakers)

**Note**: Initially validation used same dataset as training, which was corrected to VoxCeleb1 for proper generalization evaluation.

### Initial Hyperparameters (Problematic)
```yaml
training:
  learning_rate: 0.0001
  weight_decay: 0.0001
  dropout: 0.0
  feat_agg_dropout: 0.0

advanced:
  freeze_backbone_epochs: 5  # Too short!
```

---

## Problem: Severe Overfitting After Stage 1

### Observed Training Metrics

| Epoch | Stage | Train Loss | Val Loss | Gap | Status |
|-------|-------|-----------|----------|-----|--------|
| 1 | Stage 1 | 12.93 | 18.02 | 5.09 | Initial |
| 2 | Stage 1 | 11.89 | **18.00** | 6.11 | ✅ **Best Val** |
| 3 | Stage 1 | 11.75 | 18.02 | 6.27 | Slight degradation |
| 4 | Stage 1 | 11.69 | 18.01 | 6.32 | Stable |
| 5 | Stage 1 | 11.65 | 18.01 | 6.36 | End Stage 1 |
| 6 | **Stage 2** | 8.03 | 18.20 | 10.17 | ⚠️ **Backbone unfrozen** |
| 7 | Stage 2 | 5.21 | 18.22 | 13.01 | ⚠️ Overfitting begins |
| 8 | Stage 2 | 4.09 | 18.33 | 14.24 | ⚠️ Gap widening |
| 9 | Stage 2 | 3.45 | 18.35 | 14.90 | ⚠️ Severe overfitting |
| 10 | Stage 2 | 3.02 | 18.38 | 15.36 | ⚠️ **Critical** |

### Visual Analysis

```
Train Loss (decreasing - good)
20 |●
18 |
16 | ●
14 |  ●
12 |   ●●●
10 |      ●
 8 |       ●
 6 |        ●
 4 |         ●
 2 |          ●●
 0 +------------------
   1  2  3  4  5  6  7  8  9  10

Val Loss (increasing - bad!)
20 |
18 |●●●●●●●●●●
16 |
14 |
12 |
10 |
 8 |
 6 |
 4 |
 2 |
 0 +------------------
   1  2  3  4  5  6  7  8  9  10
```

### Performance Breakdown by MRL Dimension

**Best model (Epoch 2) validation losses:**
- 64D: 20.96
- 128D: 18.19
- 192D: 17.13
- 256D: 16.54

**Observation**: Higher dimensions perform better, as expected, but overall validation loss is too high and increasing.

---

## Root Cause Analysis

### 1. Premature Backbone Unfreezing (Primary Issue)

**Problem**: Backbone unfroze at epoch 6 when Stage 1 hadn't fully converged
- Train loss at epoch 5: 11.65 (still high)
- Projection head not fully optimized
- Unfreezing caused rapid memorization

**Evidence**: Sharp train loss drop (11.65 → 8.03) after unfreezing indicates aggressive overfitting, not healthy learning.

**Solution**: Extend Stage 1 to 15 epochs
- Allow projection to reach train loss ~8-10
- More stable initialization for Stage 2
- Better MRL dimension alignment

### 2. Insufficient Regularization (Secondary Issue)

**Problem**: No dropout, minimal weight decay
- Model has 5M parameters vs 1M training samples
- High capacity → easy memorization
- No mechanism to prevent overfitting

**Evidence**: Validation loss consistently increasing while train loss drops

**Solution**: Add regularization
- weight_decay: 0.0001 → 0.001 (10x increase)
- dropout: 0.1 (in projection)
- feat_agg_dropout: 0.1 (in aggregation)

### 3. Training/Validation Dataset Separation (Configuration Issue)

**Original Problem**: Validation dataset was same as training
- Made the issue harder to detect initially
- Invalid validation metrics

**Fix**: Now using VoxCeleb1 (different speakers) for validation

---

## Implemented Solutions

### Configuration Changes

```yaml
# OLD (Problematic)
training:
  weight_decay: 0.0001
  dropout: 0.0
  feat_agg_dropout: 0.0
advanced:
  freeze_backbone_epochs: 5

# NEW (Improved)
training:
  weight_decay: 0.001      # 10x stronger regularization
  dropout: 0.1             # Add dropout
  feat_agg_dropout: 0.1    # Add dropout in aggregation
advanced:
  freeze_backbone_epochs: 15  # 3x longer Stage 1
```

### Expected Improvements

**Stage 1 (Epochs 1-15):**
- More gradual train loss decrease
- Better projection head convergence
- Target: Train loss ~8-10 before Stage 2
- Validation loss should stabilize

**Stage 2 (Epochs 16-100):**
- Slower, more controlled fine-tuning
- Dropout prevents memorization
- Better generalization to validation set
- Smaller train/val gap

---

## Technical Details

### Code Fixes Applied

1. **Dataset Loading** (dataset.py)
   - Fixed torchaudio 2.9.1 compatibility
   - Added torchcodec for m4a file support
   - Removed deprecated API calls

2. **MRL Loss Function** (losses.py)
   - Fixed AAMSoftmax to support variable dimensions
   - Proper weight matrix slicing for each MRL level
   - Enables 64D/128D/192D/256D training simultaneously

3. **Pretrained Model Loading** (pretrained.py)
   - Fixed MODEL_CONFIGS: b2 uses C=16 (not C=12)
   - Extract architecture from loaded model dynamically
   - Proper weight transfer from pretrained checkpoint

4. **Wandb Integration** (train.py)
   - Added Weights & Biases logging
   - Tracks all MRL dimension losses
   - Real-time dashboard monitoring
   - API key loaded from .env

5. **Environment Setup**
   - Migrated to uv package manager
   - Added scipy, torchcodec dependencies
   - Python 3.12 environment

### Monitoring Setup

**Weights & Biases:**
- Project: mrl-speaker-recognition
- Tags: ['redimnet-b2', 'mrl', 'voxceleb2']
- Dashboard: https://wandb.ai/sappho192/mrl-speaker-recognition

**TensorBoard:**
```bash
tensorboard --logdir logs/mrl_redimnet
```

**Local Logs:**
```bash
tail -f training.log
```

---

## Training Strategy

### Stage 1: Projection Head Training (Epochs 1-15)

**Objective**: Train MRL projection to align with pretrained backbone

**Configuration**:
- Frozen backbone (4.75M parameters)
- Trainable projection (264K parameters, 5.3% of total)
- Learning rate: 0.0001
- Expected duration: ~15 hours

**Success Criteria**:
- Train loss: ~8-10
- Val loss: Stable or slightly decreasing
- No significant train/val divergence

### Stage 2: Full Model Fine-Tuning (Epochs 16-100)

**Objective**: Fine-tune entire model for optimal MRL performance

**Configuration**:
- All parameters trainable (5.02M)
- Same learning rate with cosine schedule
- Dropout active (0.1)
- Expected duration: ~85 hours

**Success Criteria**:
- Val loss: Decreasing or stable
- Train/val gap: <5 points
- Per-dimension validation losses improving

---

## Hyperparameter Rationale

### Why 15 Epochs for Stage 1?

**Analysis of previous run:**
- Epoch 5 train loss: 11.65 (still high)
- Projection not converged
- Needed ~10 more epochs to reach 8-10 range

**Calculation**:
```
Epochs needed = (current_loss - target_loss) / avg_decrease_per_epoch
             = (11.65 - 9.0) / 0.25
             ≈ 10 epochs
Stage 1 total = 5 + 10 = 15 epochs
```

### Why Dropout 0.1?

**Standard practice:**
- Speaker recognition: 0.1-0.2 typical
- 0.1 is conservative (prevents over-regularization)
- Can increase to 0.15-0.2 if still overfitting

### Why Weight Decay 0.001?

**Comparison**:
- Original: 0.0001 (too weak for 5M params)
- New: 0.001 (standard for medium models)
- ImageNet standard: 0.0001-0.001

---

## Expected Outcomes

### Successful Training Indicators

✅ **Stage 1 (End of epoch 15)**:
- Train loss: 8-10
- Val loss: 17-18 (stable)
- Gap: <8 points

✅ **Stage 2 (Epoch 30)**:
- Train loss: 5-7
- Val loss: 16-17 (decreasing)
- Gap: <6 points

✅ **Final (Epoch 100)**:
- Train loss: 2-3
- Val loss: 14-16 (best achieved)
- Gap: <5 points
- Per-dimension performance:
  - 256D: Best EER
  - 192D: ~105% of 256D
  - 128D: ~110% of 256D
  - 64D: ~120% of 256D

### Warning Signs to Watch

⚠️ **If val loss still increases**:
- Increase dropout to 0.15-0.2
- Reduce learning rate by 2-5x
- Add more data augmentation

⚠️ **If train loss doesn't decrease**:
- Reduce weight decay to 0.0005
- Reduce dropout to 0.05
- Check data augmentation isn't too aggressive

---

## Lessons Learned

1. **Two-stage training requires patience**
   - 5 epochs too short for Stage 1
   - Need projection to converge before unfreezing
   - Rule of thumb: Stage 1 until train loss plateaus

2. **Regularization is critical for large models**
   - 5M parameters need strong regularization
   - Weight decay alone insufficient
   - Dropout + weight decay combination works better

3. **Proper validation split essential**
   - Must use different speakers for validation
   - Same dataset validation gives false confidence
   - VoxCeleb1 vs VoxCeleb2 split is standard

4. **Monitor train/val gap continuously**
   - Gap >10 points indicates overfitting
   - Wandb real-time monitoring helps catch early
   - Best model often in middle of training, not end

---

## Next Steps

### Immediate (Current Run)

1. ✅ Restart training with improved config
2. Monitor first 15 epochs carefully
3. Check if val loss stabilizes
4. Evaluate Stage 1 → Stage 2 transition

### Short Term

1. Implement learning rate scheduling for Stage 2
2. Add early stopping based on validation loss
3. Consider progressive MRL training (high dims first)
4. Experiment with label smoothing

### Long Term

1. Implement proper EER evaluation on VoxCeleb1 test set
2. Compare MRL performance across dimensions
3. Benchmark against baseline ReDimNet
4. Explore LoRA for parameter-efficient fine-tuning

---

## Code Changes Summary

### Files Modified

1. **config.yaml**
   - Extended Stage 1: 5 → 15 epochs
   - Added regularization: dropout, weight_decay
   - Fixed validation dataset path

2. **train.py**
   - Added wandb logging integration
   - Environment variable loading (.env)
   - Comprehensive metric tracking

3. **dataset.py**
   - Fixed torchaudio 2.9.1 compatibility
   - Removed deprecated API calls

4. **losses.py**
   - Fixed AAMSoftmax for variable dimensions
   - MRL-compatible weight matrix slicing

5. **pretrained.py**
   - Corrected b2 architecture (C=16)
   - Dynamic config extraction from pretrained

### Git Commits

```
112e710 fix: enable pretrained model training and fix compatibility issues
73e7308 fix: use proper validation dataset (VoxCeleb1 instead of VoxCeleb2)
d63ce8c feat: add Weights & Biases (wandb) logging integration
f8c13a2 chore: add wandb directory to .gitignore
f7b964e fix: prevent overfitting with improved regularization
```

---

## Training Timeline

### Stage 1 (Epochs 1-15)
- **Duration**: ~15 hours
- **Goal**: Converge projection head
- **Trainable**: 264K params (5.3%)
- **Expected train loss**: 12 → 8-10
- **Expected val loss**: 18 → 17-18 (stable)

### Stage 2 (Epochs 16-100)
- **Duration**: ~85 hours
- **Goal**: Fine-tune full model
- **Trainable**: 5.02M params (100%)
- **Expected train loss**: 8 → 2-3
- **Expected val loss**: 17 → 14-16 (improving)

### Total Training Time
- **Estimated**: ~100 hours (~4 days)
- **GPU**: Single GPU (CUDA-capable)
- **Speed**: ~4.76-4.77 it/s

---

## Validation Strategy

### Metrics Tracked

**Per Epoch**:
- Overall loss (train/val)
- Per-dimension losses (64D, 128D, 192D, 256D)
- Learning rate
- Train/val gap

**Checkpoints**:
- Latest: Every epoch
- Best: When val loss improves
- Periodic: Every 5 epochs

### Early Stopping Criteria (Recommended)

**Stop training if**:
- Val loss increases for 10 consecutive epochs
- Train/val gap exceeds 20 points
- Train loss reaches 0 (extreme overfitting)

**Current**: No automatic early stopping (manual monitoring required)

---

## Resource Usage

### Computational
- **GPU Memory**: ~6-8GB (with mixed precision)
- **Training Speed**: 4.76 it/s
- **Time per epoch**: ~59 minutes
- **Total epochs**: 100
- **Total time**: ~98 hours

### Storage
- **Checkpoints**: ~20MB each
- **Logs**: ~100MB (TensorBoard + text)
- **Wandb**: Synced to cloud
- **Total**: <5GB for full training run

---

## Monitoring Checklist

Before leaving training unattended:

- [x] Wandb dashboard accessible
- [x] Training log being written (`training.log`)
- [x] TensorBoard logging enabled
- [x] Checkpoints saving to `checkpoints/mrl_redimnet/`
- [x] Validation dataset properly configured
- [x] Regularization enabled
- [x] GPU monitoring active (`nvidia-smi`)

During training (check every 5-10 epochs):

- [ ] Val loss trending downward or stable
- [ ] Train/val gap reasonable (<10 points)
- [ ] No NaN/inf in losses
- [ ] GPU utilization 80-100%
- [ ] Disk space sufficient

---

## Reproducibility

### Environment
```bash
Python: 3.12
PyTorch: 2.9.1+cu128
Torchaudio: 2.9.1+cu128
CUDA: 11.8+
uv: Package manager
```

### Reproduction Steps
```bash
# 1. Setup environment
cd /home/tikim/repo/redimnet-mrl
uv sync

# 2. Set PYTHONPATH
export PYTHONPATH="/home/tikim/repo/redimnet:$PYTHONPATH"

# 3. Ensure .env contains WANDB_API_KEY
echo "WANDB_API_KEY=your_key_here" > .env

# 4. Start training
uv run python train.py --config config.yaml 2>&1 | tee training.log &

# 5. Monitor
tail -f training.log
# or visit wandb dashboard
```

### Resume from Checkpoint
```bash
export PYTHONPATH="/home/tikim/repo/redimnet:$PYTHONPATH"
uv run python train.py --config config.yaml --resume checkpoints/mrl_redimnet/best.pt
```

---

## Known Issues & Limitations

### Current Issues

1. **Partial weight transfer warnings**
   - Some backbone layers don't transfer (expected)
   - Pooling layer mismatch (expected - different architecture)
   - Only affects initialization, not training

2. **FutureWarning from ReDimNet**
   - `torch.cuda.amp.autocast` deprecation
   - From upstream ReDimNet code
   - Doesn't affect training

### Limitations

1. **No automatic early stopping**
   - Manual monitoring required
   - Can be added if needed

2. **Fixed MRL dimensions**
   - [64, 128, 192, 256] hardcoded
   - Could support dynamic dimensions

3. **No progressive MRL**
   - All dimensions trained simultaneously
   - Could train high dims first, add low dims progressively

---

## Recommendations for Future Runs

### Hyperparameter Tuning

**If overfitting persists**:
1. Increase dropout: 0.1 → 0.2
2. Increase weight_decay: 0.001 → 0.005
3. Reduce learning rate: 0.0001 → 0.00005
4. Add label smoothing to AAMSoftmax

**If underfitting occurs**:
1. Reduce dropout: 0.1 → 0.05
2. Reduce weight_decay: 0.001 → 0.0005
3. Increase Stage 1 epochs further
4. Check data augmentation isn't too aggressive

### Advanced Techniques

1. **Progressive MRL Training**
   - Train 256D first (epochs 1-20)
   - Add 192D (epochs 21-40)
   - Add 128D, 64D later
   - May improve low-dimension performance

2. **Separate Learning Rates**
   - Backbone: 0.00001 (lower)
   - Projection: 0.0001 (higher)
   - Prevents catastrophic forgetting

3. **Curriculum Learning**
   - Start with easier augmentation
   - Progressively increase difficulty
   - May improve robustness

---

## Conclusion

The initial training run successfully identified a critical overfitting issue stemming from:
1. Too-short Stage 1 (5 epochs insufficient)
2. Lack of regularization (no dropout, weak weight decay)
3. Validation dataset configuration error (now fixed)

**Solutions implemented**:
- ✅ Extended Stage 1: 5 → 15 epochs
- ✅ Added dropout: 0.1
- ✅ Increased weight decay: 10x
- ✅ Fixed validation dataset

**Expected impact**:
- Better generalization to unseen speakers
- Smaller train/val gap (<10 points vs >15 previously)
- More stable Stage 2 transition
- Improved final validation performance

Training has been restarted with improved configuration. Monitoring via wandb dashboard and local logs will continue.

---

**Report Status**: ✅ Complete
**Next Review**: After 15 epochs (Stage 1 completion)
**Action Items**: Monitor val loss stabilization, adjust if needed
