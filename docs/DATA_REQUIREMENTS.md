# Data Requirements for MRL Training

**Strategy**: Option 3 - Standard MRL with Pretrained b2
**Training Mode**: Two-stage fine-tuning (recommended)

---

## Quick Answer

**Minimum viable**: Same dataset the pretrained model was trained on (VoxCeleb2)
**Recommended**: VoxCeleb2 for training + VoxCeleb1 for validation
**Optional**: Additional datasets for domain robustness

---

## 1. Core Dataset Requirements

### Training Data: VoxCeleb2

**Dataset**: VoxCeleb2 Development Set
**Link**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html

**Statistics**:
- **Speakers**: 5,994
- **Utterances**: 1,092,009
- **Duration**: ~2,442 hours
- **Languages**: Multiple (English-dominant)
- **Recording**: YouTube videos (in-the-wild)

**Directory Structure**:
```
voxceleb2/dev/aac/
├── id00012/
│   ├── 21Uxsk56VDQ/
│   │   ├── 00001.m4a
│   │   ├── 00002.m4a
│   │   └── ...
│   └── ...
├── id00013/
└── ...
```

**File Format**:
- Audio: .m4a (AAC format) or .wav
- Sample rate: Variable (need to resample to 16kHz)
- Channels: Mono or stereo (convert to mono)

**Download Size**: ~36GB (compressed)
**Extracted Size**: ~50GB

---

### Validation Data: VoxCeleb1

**Dataset**: VoxCeleb1 Development/Test Set
**Link**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

**Statistics**:
- **Speakers**: 1,251
- **Utterances**: 153,516
- **Duration**: ~352 hours

**Used for**:
- Monitoring training progress (validation loss)
- Early stopping
- Hyperparameter tuning

**Directory Structure**: Similar to VoxCeleb2

---

### Test Data: VoxCeleb1 Trials

**Files Needed**:
```
voxceleb1_test/
├── veri_test2.txt        # Official test pairs (37,720 pairs)
└── wav/
    └── id10270/...       # Test audio files
```

**Test Pairs Format** (`veri_test2.txt`):
```
1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
0 id10309/0cYFdtyWVds/00001.wav id10296/q-8fGPszYYI/00001.wav
...
```
- `1` = Same speaker
- `0` = Different speaker

**Used for**: Final EER evaluation

---

## 2. Data Preparation Steps

### Step 1: Download Datasets

```bash
# VoxCeleb2 (Training)
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac.zip
unzip vox2_dev_aac.zip

# VoxCeleb1 (Validation/Test)
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav.zip
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
unzip vox1_dev_wav.zip
unzip vox1_test_wav.zip

# Test pairs
wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
```

### Step 2: Convert Audio Format (if needed)

If you have .m4a files, convert to .wav:

```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg          # macOS

# Convert script
find voxceleb2/dev/aac -name "*.m4a" | while read file; do
    output="${file%.m4a}.wav"
    ffmpeg -i "$file" -ar 16000 -ac 1 "$output"
done
```

**Important**:
- Sample rate: 16000 Hz (16kHz)
- Channels: 1 (mono)
- Format: WAV or FLAC (lossless)

### Step 3: Organize Directory Structure

**Recommended structure**:
```
/data/
├── voxceleb2/
│   └── dev/
│       └── aac/
│           ├── id00012/
│           ├── id00013/
│           └── ...
├── voxceleb1/
│   ├── dev/
│   │   └── wav/
│   │       ├── id10270/
│   │       └── ...
│   └── test/
│       └── wav/
│           ├── id10270/
│           └── ...
└── test_pairs/
    └── veri_test2.txt
```

### Step 4: Update Config

Edit `mrl/config.yaml`:

```yaml
data:
  # Training data
  train_dataset: '/data/voxceleb2/dev/aac'

  # Validation data
  val_dataset: '/data/voxceleb1/dev/wav'

  # Test data
  test_dataset: '/data/voxceleb1/test/wav'

evaluation:
  test_pairs: '/data/test_pairs/veri_test2.txt'
```

---

## 3. Dataset Size Requirements

### Minimum Requirements

**For successful MRL training**:
- ✅ **Training**: At least 1,000 speakers with 50+ utterances each
- ✅ **Validation**: At least 100 speakers
- ✅ **Total audio**: At least 100 hours

**Why VoxCeleb2 is ideal**:
- ✅ 5,994 speakers (excellent diversity)
- ✅ ~180 utterances per speaker (sufficient)
- ✅ 2,442 hours (more than enough)

### Can You Use Less Data?

**Scenario 1: Smaller subset of VoxCeleb2**

If storage is limited, you can use a subset:

```python
# Use 50% of speakers (still 3,000 speakers)
import random
all_speakers = list(Path('/data/voxceleb2/dev/aac').iterdir())
subset = random.sample(all_speakers, len(all_speakers) // 2)

# Create subset
for speaker in subset:
    # Copy or symlink to new location
    ...
```

**Impact**:
- Training time: 50% faster ✅
- Performance: ~2-5% EER degradation ⚠️
- Still viable for most use cases

**Scenario 2: Your own dataset**

Minimum for acceptable results:
- **Speakers**: ≥500
- **Utterances per speaker**: ≥30
- **Total duration**: ≥50 hours
- **Audio quality**: Similar to VoxCeleb (in-the-wild)

**Expected performance**:
- If < 500 speakers: Significant overfitting risk
- If < 50 hours: May not converge well

---

## 4. Data During Different Training Stages

### Stage 1: Projection Head Training (Epochs 1-5)

**Data needed**: ✅ Full VoxCeleb2

**Why**: Even though backbone is frozen, the projection head needs diverse data to learn multi-resolution representations.

**Batch size**: 64 (default)
**Epochs**: 5
**Training time**: ~1 day on single GPU

### Stage 2: Full Model Fine-tuning (Epochs 6-100)

**Data needed**: ✅ Full VoxCeleb2

**Why**: Fine-tuning the entire model requires comprehensive speaker coverage.

**Batch size**: 64
**Epochs**: 50-100
**Training time**: ~1 week on 4 GPUs

---

## 5. Optional: Additional Datasets for Robustness

### CN-Celeb (Chinese Speakers)

**Purpose**: Cross-language robustness

**Statistics**:
- Speakers: 3,000
- Duration: ~274 hours
- Language: Mandarin Chinese

**How to use**:
```yaml
data:
  train_dataset: '/data/voxceleb2/dev/aac'
  additional_datasets:
    - '/data/cnceleb/data'

training:
  dataset_mixing_ratio: 0.8  # 80% VoxCeleb2, 20% CN-Celeb
```

### VoxBlink2 (Larger Scale)

**Purpose**: Maximum performance

**Statistics**:
- Speakers: 10,000+
- Duration: Much larger than VoxCeleb2

**Note**: Requires significant storage and training time

---

## 6. Data Augmentation (Built-in)

**Our implementation includes**:

```yaml
data:
  augmentation: true
  noise_snr_range: [20, 40]  # Add Gaussian noise
  volume_range: [-3, 3]       # Volume perturbation
```

**What this does**:
- Adds Gaussian noise (SNR: 20-40 dB)
- Perturbs volume (±3 dB)
- Improves robustness

**Effective data size**: 2-3x larger due to augmentation

---

## 7. No New Data Scenario

**Question**: Can I train MRL without downloading any new data?

**Answer**: ⚠️ **Not really**, but here's what you can do:

### Option A: Use Pretrained Model As-Is

```python
# No training, just use pretrained model
from mrl import load_pretrained_redimnet

model = load_pretrained_redimnet('b2', 'ft_lm', 'vox2')

# Use directly (not MRL, just pretrained)
embedding = model(audio)  # 192D embedding
```

**Pros**: ✅ Zero data needed
**Cons**: ❌ Not multi-resolution

### Option B: Synthetic Data Training (Experimental)

```python
# Generate synthetic audio for proof-of-concept
import torch

# Create dummy dataset
class SyntheticSpeakerDataset:
    def __getitem__(self, idx):
        # Random audio
        audio = torch.randn(1, 48000)
        # Random speaker label
        label = idx % 100  # 100 synthetic speakers
        return audio, label
```

**Use case**:
- Testing implementation
- Debugging training loop
- NOT for production (meaningless embeddings)

---

## 8. Storage Requirements

### Disk Space

| Dataset | Compressed | Extracted | Format |
|---------|-----------|-----------|--------|
| VoxCeleb2 Dev | 36 GB | 50 GB | m4a/wav |
| VoxCeleb1 Dev | 7 GB | 10 GB | wav |
| VoxCeleb1 Test | 2 GB | 3 GB | wav |
| **Total** | **45 GB** | **63 GB** | - |

**Recommendation**: Have at least **100 GB free space** (for processed files, checkpoints, logs)

### RAM Requirements

**During training**:
- Batch loading: ~4 GB
- Model parameters: ~1 GB
- Gradients: ~1 GB
- **Total**: ≥8 GB RAM recommended

**During data preprocessing**:
- Audio loading/resampling: ~2 GB
- Can be done in batches

---

## 9. Data Loading Performance

### Optimization Tips

**Config settings**:
```yaml
data:
  num_workers: 8           # Parallel data loading
  pin_memory: true         # Faster GPU transfer
  prefetch_factor: 2       # Prefetch batches
```

**Expected loading speed**:
- With 8 workers: ~0.5-1 second per batch
- Bottleneck usually: Disk I/O, not CPU

**SSD vs HDD**:
- SSD: ✅ Recommended (3-5x faster)
- HDD: ⚠️ Slower but acceptable

---

## 10. Quick Start: Minimal Setup

**Fastest way to get started**:

```bash
# 1. Download VoxCeleb2 dev set only (50GB)
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac.zip
unzip vox2_dev_aac.zip

# 2. Use VoxCeleb2 for both train AND validation (quick test)
# Update config.yaml
data:
  train_dataset: '/data/voxceleb2/dev/aac'
  val_dataset: '/data/voxceleb2/dev/aac'  # Same as train (not ideal but works)

# 3. Start training
cd mrl
python train.py --config config.yaml
```

**Training time**:
- Stage 1 (5 epochs): ~1 day
- Stage 2 (20 epochs): ~4 days
- **Total**: ~5 days on single GPU

---

## 11. Data Checklist

Before starting training, verify:

- [ ] VoxCeleb2 downloaded and extracted
- [ ] Audio files are 16kHz, mono
- [ ] Directory paths updated in `config.yaml`
- [ ] Can load a batch: `python -c "from mrl.dataset import create_dataloader; loader = create_dataloader('/data/voxceleb2/dev/aac', batch_size=4); print(next(iter(loader))[0].shape)"`
- [ ] At least 100GB free disk space
- [ ] GPU available: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 12. Summary

### What You Need

**Essential**:
- ✅ VoxCeleb2 (training): 50GB
- ✅ VoxCeleb1 (validation): 10GB
- ✅ Test pairs (evaluation): <1MB

**Total**: ~60GB storage, ~1 week training time

### What You DON'T Need

- ❌ Any additional labeled data
- ❌ Speaker labels beyond what's in VoxCeleb
- ❌ Transcriptions or text data
- ❌ Video data (audio only)

### Alternatives If You Don't Have VoxCeleb

**Option 1**: Use your own speaker dataset
- Minimum: 500 speakers, 30 utterances each, 50+ hours
- Format: 16kHz mono audio
- Organize same way as VoxCeleb

**Option 2**: Use other public datasets
- LibriSpeech (for English)
- Common Voice (multi-language)
- VoxPopuli (European languages)

**Convert to VoxCeleb format**:
```bash
# Example structure
my_dataset/
├── speaker_001/
│   ├── utt_001.wav
│   ├── utt_002.wav
│   └── ...
├── speaker_002/
└── ...
```

---

## Next Steps

1. **Download data**: VoxCeleb2 (~1 hour)
2. **Verify setup**: Run dataset test script
3. **Update config**: Edit paths in `config.yaml`
4. **Start training**: `python mrl/train.py --config mrl/config.yaml`

See [README.md](README.md) for complete training guide.

---

## References

- **VoxCeleb**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- **Dataset paper**: Nagrani et al., "VoxCeleb: Large-scale speaker verification in the wild" (2020)
- **Download instructions**: https://github.com/clovaai/voxceleb_trainer
