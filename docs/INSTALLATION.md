# Installation Guide

Complete installation instructions for ReDimNet-MRL.

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL2)
- **Python**: 3.12+ (pinned for compatibility with latest PyTorch and torchcodec)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (16GB recommended)
- **Storage**: 100GB free space

### Check Your System

```bash
# Python version
python --version  # Should be 3.12+

# NVIDIA GPU
nvidia-smi  # Should show your GPU

# Free disk space
df -h ~  # Should have 100GB+ free
```

---

## Installation Steps

### Option 1: Quick Install with uv (Recommended)

```bash
# 1. Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
cd ~/repo
git clone https://github.com/yourusername/redimnet-mrl.git
cd redimnet-mrl

# 3. Sync dependencies (automatically uses Python 3.12 from .python-version)
uv sync

# 4. Set up Weights & Biases (optional, for experiment tracking)
echo "WANDB_API_KEY=your_api_key_here" > .env

# 5. Verify installation
uv run python -c "import torch; print(f'PyTorch {torch.__version__}')"
uv run python -c "import torchaudio; print(f'Torchaudio {torchaudio.__version__}')"
uv run python -c "print('✅ Installation successful!')"

# 6. Test model loading
uv run python example_pretrained.py
```

### Option 1b: Traditional pip Install

```bash
# 1. Ensure Python 3.12+
python --version

# 2. Clone repository
cd ~/repo
git clone https://github.com/yourusername/redimnet-mrl.git
cd redimnet-mrl

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up Weights & Biases (optional)
echo "WANDB_API_KEY=your_api_key_here" > .env

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchaudio; print(f'Torchaudio {torchaudio.__version__}')"
python -c "print('✅ Installation successful!')"

# 6. Test model loading
python example_pretrained.py
```

### Option 2: Development Install

```bash
# Clone and install in editable mode
git clone https://github.com/yourusername/redimnet-mrl.git
cd redimnet-mrl

pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Option 3: Conda Environment

```bash
# Create conda environment with Python 3.12
conda create -n mrl python=3.12
conda activate mrl

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install pyyaml tqdm tensorboard wandb python-dotenv scipy torchcodec

# Clone repository
git clone https://github.com/yourusername/redimnet-mrl.git
cd redimnet-mrl

# Set up W&B (optional)
echo "WANDB_API_KEY=your_api_key_here" > .env
```

---

## Dependency Details

### Core Dependencies

```
torch>=2.9.0          # Deep learning framework
torchaudio>=2.9.0     # Audio processing
torchcodec>=0.1.0     # Audio codec support (for .m4a files)
scipy>=1.14.0         # Scientific computing (required by ReDimNet)
numpy>=1.20.0         # Numerical computing
pyyaml>=6.0           # Configuration files
tqdm>=4.60.0          # Progress bars
tensorboard>=2.8.0    # Training visualization
```

### Experiment Tracking & Monitoring

```
wandb>=0.12.0         # Weights & Biases experiment tracking
python-dotenv>=1.0.0  # Load WANDB_API_KEY from .env file
```

**Setup for W&B**:
```bash
# Create .env file with your API key
echo "WANDB_API_KEY=your_api_key_here" > .env

# Enable in config.yaml
logging:
  wandb: true
  wandb_project: 'mrl-speaker-recognition'
```

### Development Dependencies

```
pytest>=7.0.0         # Testing
black>=22.0.0         # Code formatting
```

---

## Verify Installation

### Test 1: Import Package

```python
from redimnet_mrl import (
    ReDimNetMRL,
    MatryoshkaProjection,
    AAMSoftmax,
    create_mrl_from_pretrained,
)
print("✅ All imports successful!")
```

### Test 2: Load Pretrained Model

```python
from redimnet_mrl import load_pretrained_redimnet

model = load_pretrained_redimnet('b2', 'ptn', 'vox2')
print(f"✅ Loaded pretrained model: {model.__class__.__name__}")
```

### Test 3: Create MRL Model

```python
import torch
from redimnet_mrl import create_mrl_from_pretrained

model = create_mrl_from_pretrained(
    model_name='b2',
    train_type='ptn',
    embed_dim=256,
    mrl_dims=[64, 128, 192, 256]
)

# Test inference
audio = torch.randn(1, 1, 48000)
emb = model(audio, target_dim=128)
print(f"✅ Embedding shape: {emb.shape}")
```

---

## Troubleshooting

### Issue: "No module named 'redimnet'"

**Problem**: Can't import ReDimNet from original repository

**Solution**: The package uses `torch.hub` to load pretrained models:
```python
# This is handled automatically by pretrained.py
model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', ...)
```

No need to install the original ReDimNet separately!

### Issue: "CUDA out of memory"

**Problem**: GPU doesn't have enough VRAM

**Solution**:
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 16  # Down from 32
   ```

2. Enable gradient accumulation:
   ```yaml
   training:
     batch_size: 16
     accumulation_steps: 2  # Effective batch = 32
   ```

### Issue: "torch.hub download failed"

**Problem**: Can't download pretrained models

**Solution**:
1. Check internet connection
2. Set cache directory:
   ```python
   torch.hub.set_dir('~/.cache/torch/hub')
   ```
3. Try manual download:
   ```bash
   git clone https://github.com/IDRnD/ReDimNet.git ~/.cache/torch/hub/IDRnD_ReDimNet_main
   ```

### Issue: "ImportError: No module named 'redimnet.layers'"

**Problem**: Original ReDimNet not in path

**Solution**: The package should handle this automatically via `torch.hub`. If issues persist:
```python
# In model.py, the path is already configured:
sys.path.insert(0, str(Path(__file__).parent.parent / "RD-1376"))
```

For standalone usage, pretrained models are loaded via `torch.hub` which includes all dependencies.

### Issue: "torchaudio backend not available" or ".m4a files not loading"

**Problem**: Audio loading fails or .m4a files not supported

**Solution**: Install torchcodec for .m4a support:
```bash
pip install torchcodec>=0.1.0
```

If issues persist:
```bash
# Linux
sudo apt-get install sox libsox-fmt-all

# macOS
brew install sox

# Or use soundfile backend
pip install soundfile
```

**Note**: The project uses torchcodec backend for loading .m4a files from VoxCeleb2, which is now handled automatically.

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install ffmpeg sox libsox-fmt-all

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install package
pip install -r requirements.txt
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install ffmpeg sox

# Install PyTorch (CPU or MPS)
pip install torch torchvision torchaudio

# Install package
pip install -r requirements.txt
```

### Windows (WSL2 recommended)

```bash
# Use Windows Subsystem for Linux 2
# Then follow Linux instructions above

# Or native Windows:
# Install PyTorch from: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Next Steps

After installation:

1. **Read documentation**: Start with [README.md](README.md)
2. **Check GPU requirements**: See [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md)
3. **Download data**: Follow [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)
4. **Start training**: Run `./quick_start.sh` or `python train.py`

---

## Uninstallation

```bash
# If installed with pip install -e
pip uninstall redimnet-mrl

# Remove repository
rm -rf ~/repo/redimnet-mrl

# Clean pip cache (optional)
pip cache purge
```

---

For more help, see:
- [README.md](README.md) - Main documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup
- Open an issue on GitHub
