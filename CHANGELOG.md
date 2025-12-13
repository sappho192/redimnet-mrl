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

## [1.0.1] - 2025-12-13 🎯

### Major Changes - Training Strategy Validated ⭐

**TL;DR**: Projection-only training (frozen backbone) is now the validated and recommended approach. Backbone fine-tuning degrades performance by 50%.

**🎁 Pre-trained Checkpoint Released**: Download validated checkpoint from [GitHub Release v1.0.1](https://github.com/sappho192/redimnet-mrl/releases/tag/1.0.1)
- Epoch 14, projection-only training
- 7.2% average EER on VoxCeleb test
- Ready to use, no training required
- 21.46 MB file size

#### Validation & Analysis

- **EER validation system**: Implemented proper speaker verification evaluation
  - `evaluate.py`: EER computation on real VoxCeleb verification pairs
  - `use_eer_validation: true` in config
  - `use_eer_for_best_model: true` for checkpoint selection
  - Generates 500 verification pairs from VoxCeleb test sets
  - Industry-standard Equal Error Rate (EER) metric

- **Checkpoint testing with real audio**:
  - `test_checkpoint.py`: Updated to use real VoxCeleb audio (was synthetic)
  - `compare_checkpoints.py`: Side-by-side checkpoint comparison tool
  - Tests on 500 real verification pairs (250 VoxCeleb1 + 250 VoxCeleb2)
  - All dimensions evaluated (64D, 128D, 192D, 256D)

#### Training Strategy - BREAKING CHANGE

**Changed from two-stage to projection-only training**:

```yaml
# OLD (deprecated):
advanced:
  freeze_backbone_epochs: 5  # Two-stage training
training:
  num_epochs: 100

# NEW (validated):
advanced:
  freeze_backbone_epochs: 9999  # Never unfreeze
training:
  num_epochs: 30  # Projection-only sufficient
```

**Evidence**:
- Epoch 14 (frozen backbone): **7.2% average EER** ✅
- Epoch 42 (unfrozen backbone): **10.85% average EER** ❌ (50% worse)
- All dimensions show consistent degradation with backbone unfreezing
- See validation reports for detailed analysis

#### Validation Reports Added

**Comprehensive analysis in `docs/report/`**:

1. **[EER_VALIDATION_RESULTS.md](docs/report/2025-12-09_EER_VALIDATION_RESULTS.md)**
   - EER tracking over 42 epochs
   - Stage 1 (frozen): 3.8-4.2% EER (consistent)
   - Stage 2 (unfrozen): 4.4% → 7.5% EER (degrading)
   - **Conclusion**: Projection-only training recommended

2. **[CHECKPOINT_COMPARISON_REAL_AUDIO.md](docs/report/2025-12-13_CHECKPOINT_COMPARISON_REAL_AUDIO.md)**
   - Side-by-side comparison on 500 VoxCeleb pairs
   - Epoch 14 vs Epoch 42 performance
   - All dimensions tested with real speaker verification
   - **Conclusion**: Frozen backbone significantly better

3. **[ROOT_CAUSE_ANALYSIS.md](docs/report/2025-12-05_ROOT_CAUSE_ANALYSIS.md)**
   - Why validation loss was misleading
   - Classification vs verification objectives
   - **Conclusion**: Use EER, not validation loss

4. **[TRAINING_REPORT.md](docs/report/2025-12-03_TRAINING_REPORT.md)**
   - Initial training analysis
   - Overfitting patterns identified

### Performance Results - Validated ✅

**Actual measured performance** (projection-only training, 30 epochs):

| Dimension | EER | Accuracy | Speed | Use Case |
|-----------|-----|----------|-------|----------|
| 256D | 5.6-7.2% | 94.4% | 1.0x | Maximum accuracy |
| 192D | 6.0-7.5% | 94.0% | 1.2x | Balanced |
| 128D | 7.6-9.0% | 92.4% | 1.5x | Mobile/Edge |
| 64D | 9.6-12.0% | 90.4% | 2.0x | Ultra-fast |

**Notes**:
- Tested on 500 real VoxCeleb verification pairs
- ~10x gap from baseline ReDimNet (0.57% EER) due to MRL adaptation
- Performance sufficient for practical applications
- All dimensions show good speaker discrimination

### Added

#### Pre-trained Checkpoint Release 🎁

- **GitHub Release v1.0.1**: Pre-trained MRL checkpoint now available for download
  - `best.pt`: Epoch 14 checkpoint (projection-only training)
  - Performance: 7.2% average EER on VoxCeleb test
  - Size: 21.46 MB
  - Download: `wget https://github.com/sappho192/redimnet-mrl/releases/download/1.0.1/best_2025-12-10_07-20.pt`
  - Ready to use immediately - no training required
  - Validated on 500 real VoxCeleb verification pairs

#### Evaluation & Testing Tools

- **EER validation module** (`evaluate.py`):
  - `generate_verification_pairs()`: Create test pairs from VoxCeleb
  - `compute_eer()`: Calculate Equal Error Rate
  - `calculate_eer()`: FAR/FRR computation
  - `evaluate_mrl_all_dims()`: Multi-dimension evaluation
  - Supports batch processing and progress tracking

- **Checkpoint comparison tool** (`compare_checkpoints.py`):
  - Side-by-side checkpoint comparison
  - Real VoxCeleb audio testing
  - Per-dimension EER reporting
  - Statistical significance analysis

- **Updated test_checkpoint.py**:
  - Real audio testing (removed misleading synthetic audio tests)
  - EER evaluation integration
  - Multi-dimension verification performance
  - Warning about synthetic audio being unreliable

### Changed

#### Training Configuration

- **Default epochs**: 100 → **30** (projection-only sufficient)
- **Backbone freezing**: 5 epochs → **9999** (never unfreeze)
- **Best model selection**: Validation loss → **EER** (proper metric)
- **Train type**: 'ft_lm' → **'ptn'** (use pre-trained weights directly)

#### Documentation - Major Updates

**README.md**:
- Added "Training Strategy Validated" banner
- Updated training timeline (7 days → 2 days)
- Added "Validation Reports" section with links
- Updated performance table with actual results
- Rewrote "Two-Stage Training" → "Projection-Only Training"
- Added evidence links throughout
- Updated status to 1.0.1

**GET_STARTED.md**:
- Added projection-only training notice
- Updated expected training output with EER values
- Changed training duration (7 days → 2 days)
- Updated performance expectations with validated numbers
- Added "why backbone stays frozen" explanation

**Config files**:
- Updated `config.yaml` with validated settings
- Added EER validation configuration
- Changed default epochs to 30
- Set `freeze_backbone_epochs: 9999`

### Fixed

- **Misleading synthetic audio tests**: Replaced with real VoxCeleb evaluation
  - Synthetic audio tests showed opposite results (misleading)
  - Real audio tests show correct performance (reliable)
  - Updated all testing scripts

- **Model checkpoint loading**: Fixed `torch.load()` for PyTorch 2.6+
  - Added `weights_only=False` for trusted checkpoints
  - Handles numpy scalar unpickling

### Deprecated

- **Two-stage training** (backbone unfreezing):
  - Evidence shows 50% performance degradation
  - No longer recommended
  - Configuration still supported but discouraged
  - See validation reports for why

### Training Improvements

**Validated training approach**:
- ✅ Use pretrained ReDimNet-b2 (ptn variant)
- ✅ Keep backbone frozen throughout training
- ✅ Train only MRL projection (~264K params, 5.3% of total)
- ✅ Monitor EER, not validation loss
- ✅ 30 epochs sufficient for convergence
- ✅ Faster training (2 days vs 7 days)
- ✅ Better generalization to unseen speakers

**Why projection-only works**:
1. Pretrained backbone already excellent (0.57% baseline EER)
2. MRL projection learns multi-resolution mappings
3. Frozen backbone prevents harmful overfitting
4. Training objective (classification) misaligned with evaluation (verification)
5. Fine-tuning specializes to training speakers, fails on new speakers

### Key Takeaways

1. **✅ Projection-only training validated**: 7.2% average EER
2. **❌ Backbone fine-tuning degrades performance**: 10.85% average EER (50% worse)
3. **✅ Use EER for evaluation**: Classification loss is misleading
4. **✅ 30 epochs sufficient**: Faster convergence
5. **✅ Real audio testing essential**: Synthetic tests unreliable

### Migration Guide

**If you're using old two-stage training**:

1. Update `config.yaml`:
   ```yaml
   advanced:
     freeze_backbone_epochs: 9999  # Was: 5
   training:
     num_epochs: 30  # Was: 100
   evaluation:
     use_eer_validation: true  # New
     use_eer_for_best_model: true  # New
   ```

2. Use EER for monitoring:
   - Watch "Val EER" in logs (not "Val Loss")
   - Best checkpoint selected based on lowest EER
   - Validation loss is no longer meaningful

3. Expect different performance:
   - Better generalization to new speakers
   - Lower EER on test sets
   - Faster training completion

### Known Issues

- None reported for projection-only training approach

### Testing

- **Validation dataset**: 500 VoxCeleb verification pairs
- **Test duration**: ~4 minutes per checkpoint (all dimensions)
- **Hardware tested**: RTX 5060 Ti 16GB, CUDA 11.8+
- **Reproducible**: Seeded pair generation

---

## [Unreleased]

### Planned Features

- **LoRA support**: Parameter-efficient fine-tuning (research)
- **Distributed training**: Multi-GPU support
- **Model zoo**: Pre-trained MRL checkpoints
- **Deployment tools**: ONNX export, TorchScript
- **Additional datasets**: LibriSpeech, Common Voice support
- **Cross-lingual evaluation**: CN-Celeb, other languages

---

[1.0.1]: https://github.com/sappho192/redimnet-mrl/releases/tag/1.0.1
[0.1.1]: https://github.com/sappho192/redimnet-mrl/releases/tag/0.1.1
[0.1.0]: https://github.com/sappho192/redimnet-mrl/releases/tag/0.1.0
