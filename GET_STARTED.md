# Get Started with ReDimNet-MRL

**Welcome!** This is your **3-step guide** to training multi-resolution speaker embeddings.

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

## Step 3: Start Training (7 days)

```bash
# Start training with your GPU-optimized config
python train.py --config config_5060ti.yaml

# In another terminal: Monitor progress
tensorboard --logdir logs/mrl_redimnet_5060ti
# Open browser: http://localhost:6006
```

**What happens**:
```
Loading pretrained ReDimNet-b2 (ft_lm, vox2)...
✅ Successfully loaded pretrained model

Epoch 1/100 [Stage 1: Backbone Frozen]
  Train Loss: 2.145
  GPU Memory: 6.2GB / 16.0GB
  Time: 4.5 hours

Epoch 5/100
  Val Loss: 1.234
  ✅ Saved best model

Epoch 6/100 [Stage 2: Unfreezing Backbone]
  Train Loss: 1.123
  GPU Memory: 7.1GB / 16.0GB

...

Epoch 100/100
  Val Loss: 0.456
  ✅ Training complete!
```

**Checkpoints saved to**: `checkpoints/mrl_redimnet_5060ti/best.pt`

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

After training, you'll have **one model** that provides:

| Dimension | EER | Speed | Use Case |
|-----------|-----|-------|----------|
| 64D | ~1.2% | 2x faster | Real-time processing |
| 128D | ~1.0% | 1.5x faster | Mobile apps |
| 192D | ~0.9% | 1.2x faster | Balanced |
| 256D | ~0.85% | Baseline | Maximum accuracy |

**All from the same model!** 🎉

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
10. 🔄 [MIGRATION_NOTES.md](MIGRATION_NOTES.md) - Migration details

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
