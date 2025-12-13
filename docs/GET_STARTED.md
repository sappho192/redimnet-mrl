# Get Started with ReDimNet-MRL

**Welcome!** Choose your path:

## 🚀 Quick Start (No Training) - Recommended ⭐

**Use our pre-trained checkpoint** - ready to use immediately!

```bash
# 1. Install PyTorch
pip install torch torchaudio

# 2. Clone and download checkpoint
git clone https://github.com/sappho192/redimnet-mrl.git
cd redimnet-mrl
mkdir -p checkpoints/mrl_redimnet
wget https://github.com/sappho192/redimnet-mrl/releases/download/1.0.1/best_2025-12-10_07-20.pt \
     -O checkpoints/mrl_redimnet/best.pt

# 3. Use immediately!
# See "30-Second Example" in README.md
```

**Performance**: 7.2% average EER, validated on 500 VoxCeleb pairs.

---

## 🎯 Full Training Guide

**Want to train your own model?** Follow the 3-step guide below.

> [!IMPORTANT]
> **Training Strategy**: Use projection-only training (frozen backbone) for best results.
> This approach is validated and achieves 7.2% average EER on VoxCeleb test.

---

## Prerequisites Check ✓

Before starting, verify you have:

```bash
# 1. GPU with 16GB VRAM
nvidia-smi  # Should show RTX 5060 Ti

# 2. Python 3.8+
python --version  # Should be 3.8 or higher

# 3. 100GB+ free space
df -h ~  # Check available space
```

---

## Step 1: Install Dependencies (5 minutes)

```bash
cd ~/repo/redimnet-mrl

# Install Python packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchaudio; print(f'Torchaudio {torchaudio.__version__}')"
python -c "print('✅ Installation successful!')"
```

**Expected output**:
```
PyTorch 2.x.x
Torchaudio 2.x.x
✅ Installation successful!
```

---

## Step 2: Download Data (2-4 hours)

### Option A: Automated Download

```bash
./quick_start.sh
# This script will:
# - Download VoxCeleb2 (~50GB)
# - Verify data integrity
# - Update config paths
# - Test data loading
```

### Option B: Manual Download

```bash
# Download VoxCeleb2
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac.zip
unzip vox2_dev_aac.zip -d ~/data/voxceleb2

# Update config
vim config_5060ti.yaml
# Change: train_dataset: '~/data/voxceleb2/dev/aac'

# Test data loading
python -c "
from dataset import VoxCelebDataset
ds = VoxCelebDataset('~/data/voxceleb2/dev/aac')
print(f'✅ Loaded {len(ds)} utterances')
"
```

---

## Step 3: Start Training (2 days)

```bash
# Start training with projection-only approach (validated)
python train.py --config config_5060ti.yaml

# In another terminal: Monitor progress
tensorboard --logdir logs/mrl_redimnet_5060ti
# Open browser: http://localhost:6006
```

**What happens**:
```
Loading pretrained ReDimNet-b2 (ptn, vox2)...
✅ Successfully loaded pretrained model

Epoch 1/30 [Projection-Only Training: Backbone Frozen]
  Train Loss: 12.93
  Val EER: 3.80%  # This is the metric that matters!
  GPU Memory: 6.2GB / 16.0GB
  Time: ~90 minutes per epoch

Epoch 5/30
  Train Loss: 11.65
  Val EER: 4.00%

Epoch 14/30
  Train Loss: 11.55
  Val EER: 4.20%
  ✅ Best model saved (lowest EER)

Epoch 30/30
  Train Loss: 11.50
  Val EER: 4.30%
  ✅ Training complete!

Final Results:
  Best EER: 4.20% (Epoch 14)
  64D EER: 9.6%
  128D EER: 7.6%
  192D EER: 6.0%
  256D EER: 5.6%
```

**Checkpoints saved to**: `checkpoints/mrl_redimnet_5060ti/best.pt`

> [!NOTE]
> **Why backbone stays frozen**: Backbone fine-tuning degrades performance by 50%.
> See [validation reports](../docs/report/) for detailed analysis.

---

## Step 4: Use Your Trained Model

```python
from pretrained import create_mrl_from_pretrained
import torch

# Load your trained model
model = torch.load('checkpoints/mrl_redimnet_5060ti/best.pt')

# Or load from checkpoint
checkpoint = torch.load('checkpoints/mrl_redimnet_5060ti/best.pt')
model = create_mrl_from_pretrained('b2', 'ptn', 'vox2')
model.load_state_dict(checkpoint['model_state_dict'])

# Extract embeddings
audio = torch.randn(1, 1, 48000)

emb_64d = model(audio, target_dim=64)    # Fast
emb_256d = model(audio, target_dim=256)  # Accurate

print(f"Fast: {emb_64d.shape}")      # [1, 64]
print(f"Accurate: {emb_256d.shape}") # [1, 256]
```

---

## What You Get

After training (projection-only, 30 epochs), you'll have **one model** that provides:

| Dimension | EER | Speed | Accuracy | Use Case |
|-----------|-----|-------|----------|----------|
| 64D | ~9.6% | 2x faster | 90.4% | Real-time processing |
| 128D | ~7.6% | 1.5x faster | 92.4% | Mobile apps |
| 192D | ~6.0% | 1.2x faster | 94.0% | Balanced |
| 256D | ~5.6% | Baseline | 94.4% | Maximum accuracy |

**All from the same model!** 🎉

> [!NOTE]
> Performance validated on 500 real VoxCeleb pairs.
> See [checkpoint comparison report](../docs/report/2025-12-13_CHECKPOINT_COMPARISON_REAL_AUDIO.md) for details.

---

## Troubleshooting

### Can't download VoxCeleb2?

**Alternative**: Use your own speaker dataset
- Organize as: `data/speaker_id/utterance.wav`
- Minimum: 500 speakers, 30 utterances each
- Update config paths

### Out of memory during training?

**Solution**: Lower batch size
```yaml
training:
  batch_size: 32  # Down from 48
```

### Training too slow?

**Solution**: Check GPU utilization
```bash
nvidia-smi  # Should show 80-100% GPU usage
```

If low utilization:
- Increase `num_workers` in config
- Use SSD instead of HDD for data
- Enable `compile: true` (PyTorch 2.0+)

---

## Quick Reference

| Task | Command |
|------|---------|
| **Install** | `pip install -r requirements.txt` |
| **Download data** | `./quick_start.sh` |
| **Train** | `python train.py --config config_5060ti.yaml` |
| **Monitor** | `tensorboard --logdir logs/` |
| **Test model** | `python example_pretrained.py` |
| **Resume training** | `python train.py --config config.yaml --resume checkpoints/latest.pt` |

---

## Documentation Index

**Start here**:
1. 📖 [README.md](README.md) - Overview and features
2. 💻 [INSTALLATION.md](INSTALLATION.md) - Detailed installation
3. 📊 [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md) - Your RTX 5060 Ti specs
4. 💾 [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) - Dataset download

**Advanced**:
5. 🔥 [PRETRAINED_GUIDE.md](PRETRAINED_GUIDE.md) - Using pretrained models
6. 🧬 [LORA_SURVEY.md](LORA_SURVEY.md) - Parameter-efficient training
7. 🔬 [CROSS_MODEL_DISTILLATION_ANALYSIS.md](CROSS_MODEL_DISTILLATION_ANALYSIS.md) - Model fusion

**Development**:
8. 🤝 [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
9. 📝 [CHANGELOG.md](CHANGELOG.md) - Version history

---

## Timeline

| Task | Duration | When |
|------|----------|------|
| Install dependencies | 5 minutes | Now |
| Download VoxCeleb2 | 2-4 hours | Today |
| Training Stage 1 | 18 hours | Day 1 |
| Training Stage 2 | 6 days | Days 2-7 |
| **Total** | **~7 days** | **Week 1** |

---

## Support

**Questions?**
- Check [README.md](README.md) first
- Review [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md) for memory issues
- Open GitHub issue for bugs

**Ready to start?**
```bash
./quick_start.sh
```

**Happy training!** 🎯
