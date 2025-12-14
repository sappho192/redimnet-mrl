# Diarization Configuration Usage Guide

## Overview

The `diar_config.yaml` file contains optimized hyperparameters for ReDimNet-MRL speaker diarization based on threshold optimization testing (December 14, 2025).

## Quick Start

### 1. Use Default Configuration

The simplest way to use the optimized parameters:

```bash
python diarization/main.py --audio sample.wav
```

This automatically loads `diarization/diar_config.yaml` with the best hyperparameters:
- coarse_threshold: 0.60
- refined_threshold: 0.40
- boundary_threshold: 0.70
- method: hierarchical_mrl

### 2. Use Custom Configuration File

```bash
python diarization/main.py --audio sample.wav --diar-config my_config.yaml
```

### 3. Override Specific Parameters

Command-line arguments override config file values:

```bash
# Override coarse threshold
python diarization/main.py --audio sample.wav --coarse-threshold 0.65

# Override multiple parameters
python diarization/main.py --audio sample.wav \
  --coarse-threshold 0.65 \
  --refined-threshold 0.50

# Use different clustering method
python diarization/main.py --audio sample.wav \
  --clustering-method pyannote_default
```

## Configuration File Structure

```yaml
clustering:
  method: hierarchical_mrl
  coarse_threshold: 0.60
  refined_threshold: 0.40
  boundary_threshold: 0.70
  min_cluster_size: 2

embedding:
  dimension: 256
  extract_all_dims: true
  checkpoint_path: ../checkpoints/best.pt
  config_path: ../config.yaml
```

## Alternative Scenarios

The config file includes presets for different use cases:

### Few Speakers (2-4 speakers)

Use higher thresholds to avoid over-segmentation:

```yaml
# In diar_config.yaml under few_speakers:
coarse_threshold: 0.65
refined_threshold: 0.50
boundary_threshold: 0.70
```

Command line:
```bash
python diarization/main.py --audio sample.wav \
  --coarse-threshold 0.65 \
  --refined-threshold 0.50
```

### Many Speakers (5-10+ speakers)

Use lower thresholds to detect more clusters:

```yaml
# In diar_config.yaml under many_speakers:
coarse_threshold: 0.55
refined_threshold: 0.35
boundary_threshold: 0.70
```

Command line:
```bash
python diarization/main.py --audio sample.wav \
  --coarse-threshold 0.55 \
  --refined-threshold 0.35
```

### High Accuracy Priority

Optimize for best DER:

```yaml
# In diar_config.yaml under high_accuracy:
coarse_threshold: 0.60
refined_threshold: 0.40
boundary_threshold: 0.65
min_cluster_size: 1
```

### High Speed Priority

Faster clustering with higher thresholds:

```yaml
# In diar_config.yaml under high_speed:
coarse_threshold: 0.70
refined_threshold: 0.55
boundary_threshold: 0.75
min_cluster_size: 3
```

## Tuning Guidelines

### Threshold Effects

**Coarse Threshold** (Stage 1 - 64D clustering):
- **Lower (0.5-0.6)**: More initial clusters, better for many speakers
- **Higher (0.7-0.9)**: Fewer initial clusters, faster but may under-segment

**Refined Threshold** (Stage 2 - 192D sub-clustering):
- **Lower (0.3-0.4)**: More sub-cluster splits, finer granularity
- **Higher (0.5-0.7)**: Fewer splits, preserve coarse clustering

**Boundary Threshold** (Stage 3 - 256D verification):
- **Lower (0.6-0.65)**: More samples reassigned, more conservative
- **Higher (0.75-0.8)**: Fewer reassignments, trust Stage 2 more

### General Rules

1. **Too many speakers detected?**
   - Increase coarse_threshold (+0.05 to +0.10)
   - Increase refined_threshold (+0.05 to +0.10)

2. **Too few speakers detected?**
   - Decrease coarse_threshold (-0.05 to -0.10)
   - Decrease refined_threshold (-0.05 to -0.10)

3. **Adjust coarse_threshold first** (biggest impact)
4. **Fine-tune refined_threshold second**
5. **Keep boundary_threshold around 0.7** (usually doesn't need much tuning)

## Validation Testing

Test results from December 14, 2025:

| Coarse | Refined | Coarse Clusters | Final Speakers | DER | Time |
|--------|---------|-----------------|----------------|-----|------|
| 0.60 | 0.40 | 14 | 13 | 76.77% | 45.6s |
| 0.65 | 0.50 | 26 | 24 | 86.38% | 45.6s |
| 0.70 | 0.55 | 46 | 41 | 89.57% | 45.6s |
| 0.80 | 0.65 | 151 | 111 | 102.34% | 46.4s |
| 0.90 | 0.75 | 465 | 232 | 108.79% | 48.9s |

**Best configuration**: coarse=0.60, refined=0.40 (lowest DER)

## Scripts That Use Configuration

All major diarization scripts now support `diar_config.yaml`:

1. **main.py** - Main diarization script
   - Loads config automatically
   - Command-line args override config values

2. **test_diarization.py** - Testing script
   - Uses config for hierarchical clustering parameters

3. **Future scripts** - Easy to integrate:
   ```python
   import yaml
   from pathlib import Path

   def load_diar_config(config_path='diarization/diar_config.yaml'):
       with open(config_path, 'r') as f:
           config = yaml.safe_load(f)
       return config

   config = load_diar_config()
   coarse_threshold = config['clustering']['coarse_threshold']
   ```

## Best Practices

1. **Start with defaults** - The default config (0.60/0.40/0.70) is optimized for general use

2. **Test with ground truth** - If you have RTTM annotations:
   ```bash
   python diarization/test_diarization.py
   ```

3. **Iterate systematically**:
   - Adjust coarse_threshold first
   - Observe speaker count changes
   - Fine-tune refined_threshold
   - Validate with DER

4. **Document your changes** - Keep notes on what works for your specific audio type

5. **Create custom configs** - For different domains (meetings, calls, broadcasts):
   ```bash
   cp diar_config.yaml meeting_config.yaml
   # Edit meeting_config.yaml for your needs
   python diarization/main.py --audio meeting.wav --diar-config meeting_config.yaml
   ```

## Troubleshooting

**Config not loading?**
- Check file exists: `ls diarization/diar_config.yaml`
- Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('diarization/diar_config.yaml'))"`

**Parameters not applied?**
- Command-line args override config (this is intentional)
- Check script output to see which values are being used

**Unexpected results?**
- Verify config values: `cat diarization/diar_config.yaml | grep threshold`
- Run with `--verbose` for detailed logging
- Compare with default values (0.60/0.40/0.70)

## Examples

### Example 1: Interview Audio (2 speakers)
```bash
python diarization/main.py --audio interview.wav \
  --coarse-threshold 0.70 \
  --refined-threshold 0.55
```

### Example 2: Meeting Audio (5-8 speakers)
```bash
python diarization/main.py --audio meeting.wav \
  --coarse-threshold 0.55 \
  --refined-threshold 0.35
```

### Example 3: Broadcast (many speakers, need accuracy)
```bash
python diarization/main.py --audio broadcast.wav \
  --coarse-threshold 0.60 \
  --refined-threshold 0.40 \
  --boundary-threshold 0.65
```

### Example 4: Real-time (need speed)
```bash
python diarization/main.py --audio sample.wav \
  --coarse-threshold 0.75 \
  --refined-threshold 0.60
```

## See Also

- `diar_config.yaml` - Configuration file with full documentation
- `README.md` - Main documentation
- `docs/report/2025-12-14_speaker_diarization_implementation.md` - Technical details
