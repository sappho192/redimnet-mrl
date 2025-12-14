# Input Directory Usage Guide

This document describes how to use the updated diarization scripts that now accept input directories instead of hardcoded file paths.

## Updated Scripts

### 1. test_diarization.py

The main testing script has been updated to accept either a single audio file or an entire directory of audio files.

#### Usage Examples

**Single audio file:**
```bash
# Without reference RTTM
uv run python diarization/test_diarization.py --audio sample.wav

# With reference RTTM
uv run python diarization/test_diarization.py --audio sample.wav --reference sample.rttm

# Test only hierarchical method
uv run python diarization/test_diarization.py --audio sample.wav --method hierarchical

# Test only standard method
uv run python diarization/test_diarization.py --audio sample.wav --method standard
```

**Process entire directory:**
```bash
# Process all audio files in a directory
uv run python diarization/test_diarization.py --input-dir diarization/input/

# The script will automatically look for matching RTTM files with these patterns:
#   - audio.flac -> audio.rttm
#   - audio.flac -> audio_annotated.rttm
```

#### Command-line Arguments

- `--audio`: Single audio file to process (mutually exclusive with --input-dir)
- `--input-dir`: Directory containing audio files to process (mutually exclusive with --audio)
- `--reference`: Reference RTTM file (for single audio file only)
- `--method`: Which clustering method to test (`standard`, `hierarchical`, or `both`)
  - Default: `both`
- `--device`: Device for inference (`cpu`, `cuda`, `cuda:0`, etc.)
  - Default: `cuda` if available, else `cpu`

#### Supported Audio Formats

When using `--input-dir`, the script automatically detects files with these extensions:
- `.wav` / `.WAV`
- `.flac` / `.FLAC`
- `.mp3` / `.MP3`
- `.m4a` / `.M4A`

#### Reference RTTM Auto-Detection

When processing a directory with `--input-dir`, the script automatically looks for reference RTTM files using these patterns:

1. Same name with `.rttm` extension: `audio.flac` → `audio.rttm`
2. Name with `_annotated` suffix: `audio.flac` → `audio_annotated.rttm`

If a matching RTTM file is found, it will be used for DER computation and visualization.

### 2. scripts/test_debug_embeddings.py

Debug script for understanding the embedding and clustering flow.

#### Usage Examples

```bash
# Debug with audio file
uv run python diarization/scripts/test_debug_embeddings.py --audio sample.wav

# Debug with specific device
uv run python diarization/scripts/test_debug_embeddings.py --audio sample.wav --device cpu
```

#### Command-line Arguments

- `--audio`: Input audio file (required)
- `--device`: Device for inference (default: cuda if available, else cpu)

## Migration from Hardcoded Paths

### Before (Hardcoded)

```python
# Old approach - hardcoded paths
AUDIO_FILE = "diarization/input/example_audio.flac"
REFERENCE_RTTM = "diarization/input/example_audio_annotated.rttm"
```

### After (Command-line Arguments)

```bash
# New approach - flexible command-line arguments
uv run python diarization/test_diarization.py \
  --audio diarization/input/example_audio.flac \
  --reference diarization/input/example_audio_annotated.rttm

# Or process entire directory
uv run python diarization/test_diarization.py \
  --input-dir diarization/input/
```

## Output Files

All scripts save their outputs to:
- **Diarization results (RTTM)**: `diarization/output/`
- **Visualizations**: `diarization/visualization/`

The output filenames are automatically generated based on the input audio filename and method name.

## Examples

### Process all files in a directory

```bash
# Process all audio files with automatic RTTM detection
uv run python diarization/test_diarization.py --input-dir diarization/input/

# Expected directory structure:
# diarization/input/
#   ├── audio1.wav
#   ├── audio1.rttm          # Auto-detected
#   ├── audio2.flac
#   ├── audio2_annotated.rttm  # Auto-detected
#   └── audio3.mp3           # Processed without reference
```

### Test single file with both methods

```bash
# Compare standard vs hierarchical clustering
uv run python diarization/test_diarization.py \
  --audio sample.wav \
  --reference sample.rttm \
  --method both
```

### Test with hierarchical method only

```bash
# Faster - only test hierarchical clustering
uv run python diarization/test_diarization.py \
  --audio sample.wav \
  --method hierarchical
```

## Backward Compatibility

The scripts no longer support hardcoded file paths. All input must be specified via command-line arguments. This provides:

1. **Flexibility**: Process any audio file or directory without code changes
2. **Automation**: Easily integrate into scripts and pipelines
3. **Batch processing**: Process multiple files in one command
4. **CI/CD friendly**: No need to modify code for different datasets

## Integration with Other Scripts

All other diarization scripts (`main.py`, `compare_clustering_methods.py`, `compare_dimensions.py`) already accept command-line arguments and are compatible with this approach.

```bash
# Main diarization script
uv run python diarization/main.py --audio sample.wav

# Compare clustering methods
uv run python diarization/compare_clustering_methods.py --audio sample.wav

# Compare embedding dimensions
uv run python diarization/compare_dimensions.py --audio sample.wav
```
