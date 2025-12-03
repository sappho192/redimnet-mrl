# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-02

### Added

#### Core Features
- **MatryoshkaProjection module**: Multi-resolution projection head
- **ReDimNetMRL model**: MRL-enabled wrapper around ReDimNet
- **Pretrained model support**: Load and convert official ReDimNet models (b0-b6)
- **Two-stage training**: Automatic backbone freezing/unfreezing
- **Multi-resolution training**: Simultaneous training at [64, 128, 192, 256] dimensions

#### Loss Functions
- **MatryoshkaLoss**: Multi-dimension loss wrapper
- **AAMSoftmax (ArcFace)**: Speaker verification loss
- **SubCenterAAMSoftmax**: Advanced variant with sub-centers
- **TripletLoss**: Alternative metric learning approach

#### Data Pipeline
- **VoxCelebDataset**: Complete dataset loader with augmentation
- **PairedSpeakerDataset**: For verification tasks
- **Audio augmentation**: Noise injection, volume perturbation
- **Flexible DataLoader**: Configurable workers, pin memory

#### Training Infrastructure
- **Complete training script**: train.py with full training loop
- **Configuration system**: YAML-based configuration
- **Checkpoint management**: Save/load/resume training
- **TensorBoard integration**: Real-time monitoring
- **Weights & Biases support**: Optional experiment tracking
- **Mixed precision training**: Automatic memory optimization
- **Gradient accumulation**: For memory-constrained GPUs

#### Pretrained Model Integration
- **torch.hub loading**: Automatic download of official models
- **Weight transfer**: Backbone initialization from pretrained
- **Model variant support**: b0, b1, b2, b3, b4, b5, b6, M
- **Training type selection**: ptn, ft_lm, ft_mix

#### Documentation
- **README.md**: Comprehensive overview and quick start
- **PRETRAINED_GUIDE.md**: Using pretrained ReDimNet models
- **DATA_REQUIREMENTS.md**: Dataset download and preparation
- **GPU_REQUIREMENTS.md**: Memory analysis and optimization
- **LORA_SURVEY.md**: LoRA feasibility analysis
- **CROSS_MODEL_DISTILLATION_ANALYSIS.md**: Model ensemble strategies

#### Utilities
- **quick_start.sh**: Automated setup script
- **example_pretrained.py**: Usage examples
- **GPU-specific configs**: Optimized for different hardware

### Technical Details

- **Total code**: ~1,850 lines of Python
- **Total documentation**: ~3,200 lines of Markdown
- **Model variants supported**: 7 (b0-b6, M)
- **MRL dimensions**: Configurable, default [64, 128, 192, 256]
- **Pretrained models**: Compatible with official IDRnD/ReDimNet

### Performance Targets

| Dimension | Target EER | Speed | Memory |
|-----------|-----------|-------|--------|
| 256D | 0.8-0.9% | 1.0x | 1.0x |
| 192D | 0.85-0.95% | 1.2x | 0.75x |
| 128D | 0.95-1.1% | 1.5x | 0.50x |
| 64D | 1.1-1.4% | 2.0x | 0.25x |

### Hardware Support

- **Minimum**: 12GB VRAM (RTX 3060)
- **Recommended**: 16GB VRAM (RTX 5060 Ti) ✅
- **Optimal**: 24GB+ VRAM (RTX 3090, A100)

### Known Limitations

- Requires VoxCeleb2 dataset (~50GB)
- Training takes 7-14 days on single GPU
- No distributed training support yet (planned)
- Evaluation tools not yet implemented (planned)

### Dependencies

- PyTorch >= 2.0.0
- Torchaudio >= 2.0.0
- Python >= 3.8

**Note**: See version 0.1.1 for updated dependency requirements (PyTorch 2.9+, Python 3.12+)

## [0.1.1] - 2025-12-03

### Added

#### Experiment Tracking
- **Weights & Biases (wandb) integration**: Comprehensive experiment tracking with real-time metrics
  - Training and validation loss logging
  - Multi-dimension MRL loss tracking (64D, 128D, 192D, 256D)
  - Learning rate scheduling visualization
  - Best model tracking in wandb summary
  - Dashboard URL printed at training startup
- **python-dotenv support**: Load `WANDB_API_KEY` from `.env` file automatically
- **wandb configuration options**:
  - `wandb_project`: Set project name
  - `wandb_run_name`: Optional run name
  - `wandb_tags`: Tag experiments for organization
  - `wandb_watch_model`: Optional gradient/parameter logging

#### Environment & Dependencies
- **uv package manager support**: Modern Python package management
  - `pyproject.toml`: Project configuration
  - `uv.lock`: Dependency lockfile for reproducibility
  - Faster dependency resolution and installation
- **Python 3.12**: Pinned via `.python-version` for compatibility
- **scipy >= 1.14.0**: Required for ReDimNet compatibility
- **torchcodec >= 0.1.0**: Native .m4a audio file support (no manual conversion needed)
- **Updated PyTorch**: torch >= 2.9.0, torchaudio >= 2.9.0

#### Audio Processing
- **Automatic .m4a support**: Load VoxCeleb2 AAC files directly via torchcodec backend
- **Removed deprecated torchaudio.info()**: Improved compatibility with torchaudio 2.9.1
- **Skip duration validation**: Faster file list building during dataset initialization

### Changed

#### Training Data Configuration
- **Validation dataset fix**: Use VoxCeleb1 dev instead of VoxCeleb2 for validation
  - Follows standard speaker recognition practices
  - Validation uses **different speakers** (1,251) than training (5,994)
  - Prevents overfitting and provides true generalization metrics
  - Configuration updated: `val_dataset: '/path/to/voxceleb1/dev/wav'`

#### Model Architecture
- **Fixed ReDimNet-b2 configuration**: Corrected channel multiplier from C=12 to C=16
  - Matches official pretrained model architecture
  - Ensures proper weight transfer from pretrained models
- **Enhanced pretrained model loading**:
  - Extract architecture directly from loaded pretrained weights
  - Improved error messages for weight transfer failures
  - Added `stages_setup` parameter to model creation

#### Loss Functions
- **AAMSoftmax variable dimension support**: Dynamic weight matrix slicing for MRL
  - Supports all MRL dimensions (64D, 128D, 192D, 256D) without dimension mismatches
  - Enables true multi-resolution training

#### Package Management
- **Migrated quick_start.sh to uv**: From pip to modern uv package manager
  - Commands changed: `uv sync`, `uv run python`
  - Faster and more reliable dependency installation
- **Git housekeeping**: Added training artifacts to `.gitignore`
  - `checkpoints/`, `logs/`, `*.backup`, `training*.log`
  - `wandb/` directory for experiment tracking data

### Fixed

- **Pretrained model loading**: Re-enabled `use_pretrained: true` in default config
- **Dataset compatibility**: Fixed issues with VoxCeleb2 .m4a files
- **MRL training**: Resolved dimension mismatch errors in loss computation
- **Stage 1 training**: Successfully loads pretrained ReDimNet-b2 (5.07M params) with frozen backbone

### Documentation

#### Updated
- **README.md**:
  - Added wandb installation and setup instructions
  - Updated installation steps to include `python-dotenv`
  - Clarified dataset requirements (VoxCeleb1 for validation)
  - Added wandb configuration section
  - Updated monitoring instructions (TensorBoard + wandb)
- **DATA_REQUIREMENTS.md**:
  - Emphasized importance of separate validation dataset
  - Added automatic .m4a file support information
  - Removed manual audio conversion requirements
  - Added wandb setup instructions
  - Clarified why VoxCeleb1 is used for validation
- **INSTALLATION.md**:
  - Added uv package manager installation instructions
  - Updated Python requirement to 3.12+
  - Added wandb and python-dotenv to dependencies
  - Added scipy and torchcodec to core dependencies
  - Updated dependency versions (torch >= 2.9.0)
  - Added .env file setup instructions

### Training Status (as of 2025-12-03)

- ✅ Successfully loads pretrained ReDimNet-b2 (5.07M params)
- ✅ Stage 1 active: Frozen backbone, training MRL projection (264K trainable params)
- ✅ Training speed: ~4.79 it/s (~59 min/epoch on single GPU)
- ✅ Dataset: VoxCeleb2 (1.09M files, 5,994 speakers)
- ✅ Validation: VoxCeleb1 dev (1,251 different speakers)
- ✅ Experiment tracking: wandb logging active

### Dependencies Update

```
Core:
- torch >= 2.9.0 (was >= 2.0.0)
- torchaudio >= 2.9.0 (was >= 2.0.0)
- scipy >= 1.14.0 (new)
- torchcodec >= 0.1.0 (new)

Monitoring:
- wandb >= 0.12.0 (new)
- python-dotenv >= 1.0.0 (new)

Python:
- Python >= 3.12 (was >= 3.8)
```

---

## [Unreleased]

### Planned Features

- **Evaluation tools**: Multi-dimension EER evaluation
- **Benchmark suite**: Comprehensive performance testing
- **LoRA support**: Parameter-efficient fine-tuning (research)
- **Distributed training**: Multi-GPU support
- **Model zoo**: Pre-trained MRL checkpoints
- **Deployment tools**: ONNX export, TorchScript
- **Additional datasets**: LibriSpeech, Common Voice support

---

[0.1.1]: https://github.com/yourusername/redimnet-mrl/releases/tag/v0.1.1
[0.1.0]: https://github.com/yourusername/redimnet-mrl/releases/tag/v0.1.0
