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

[0.1.0]: https://github.com/yourusername/redimnet-mrl/releases/tag/v0.1.0
