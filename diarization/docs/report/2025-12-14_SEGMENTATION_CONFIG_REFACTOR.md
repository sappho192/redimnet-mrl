# Segmentation Configuration Centralization

## Summary

All diarization scripts now properly reference segmentation parameters from `diar_config.yaml` instead of using hardcoded defaults.

## Changes Made

### 1. Added `apply_segmentation_params()` Helper Function

A reusable helper function was added to apply segmentation parameters from config to any pipeline.

**Location**: Added to 3 files:
- `compare_clustering_methods.py` (lines 63-105)
- `compare_dimensions.py` (lines 67-109)
- `test_diarization.py` (lines 80-122)

**Function signature**:
```python
def apply_segmentation_params(pipeline, config=None, verbose=False):
    """
    Apply segmentation parameters from config to pipeline.

    Args:
        pipeline: pyannote SpeakerDiarization pipeline
        config: Configuration dict (will load default if None)
        verbose: Print parameter info
    """
```

**Features**:
- Automatically loads `diar_config.yaml` if no config provided
- Detects powerset vs binary model type
- Applies appropriate parameters based on model type
- Optionally prints applied parameters when verbose=True

### 2. Updated All Diarization Functions

#### `compare_clustering_methods.py`

**`run_diarization_standard()`**:
- Added call to `apply_segmentation_params(pipeline, verbose=verbose)` after pipeline creation
- Location: Line 157

**`run_diarization_hierarchical()`**:
- Added call to `apply_segmentation_params(pipeline, config=diar_config, verbose=verbose)` after pipeline creation
- Location: Line 240
- Changed verbose message from "Thresholds:" to "Clustering thresholds:" for clarity (line 253)

#### `compare_dimensions.py`

**`run_diarization_with_dimension()`**:
- Added call to `apply_segmentation_params(pipeline, verbose=verbose)` after pipeline creation
- Location: Line 160

#### `test_diarization.py`

**`test_diarization_full()`**:
- Added config loading: `diar_config = load_diar_config()`
- Added call to `apply_segmentation_params(pipeline, config=diar_config, verbose=True)` after embedding override
- Location: Lines 375-379

### 3. Configuration File (`diar_config.yaml`)

**Already contains** (no changes needed):
```yaml
pipeline:
  segmentation:
    threshold: 0.5              # Only for binary models
    min_duration_on: 0.0        # Post-processing filter
    min_duration_off: 0.0       # Main parameter (all models)
```

These values are now referenced by all scripts.

## Benefits

### 1. **Single Source of Truth**
All segmentation parameters defined in one place (`diar_config.yaml`)

### 2. **Consistency**
All scripts use the same parameter values by default

### 3. **Easy Tuning**
Change values in `diar_config.yaml` once, affects all scripts

### 4. **Maintainability**
No scattered hardcoded defaults throughout codebase

### 5. **Visibility**
Verbose mode shows which parameters are being applied

## Usage

### Changing Default Segmentation Parameters

Edit `diar_config.yaml`:
```yaml
pipeline:
  segmentation:
    min_duration_off: 0.1  # Changed from 0.0
```

All scripts will automatically use the new value:
- `main.py` ✓
- `compare_clustering_methods.py` ✓
- `compare_dimensions.py` ✓
- `test_diarization.py` ✓

### Verification

All scripts will print segmentation parameters when run with verbose mode:

```bash
# main.py shows segmentation parameters by default
python diarization/main.py --audio sample.wav

# Other scripts show with verbose flag
python diarization/compare_clustering_methods.py --audio sample.wav --verbose
python diarization/compare_dimensions.py --audio sample.wav --verbose
```

Expected output:
```
Segmentation: min_duration_off=0.0s (powerset model)
```

Or for binary models:
```
Segmentation: threshold=0.5
Segmentation: min_duration_off=0.0s
```

## Testing

To verify the changes work correctly:

```bash
# Test 1: Verify config is loaded
python -c "
from pathlib import Path
import yaml
config = yaml.safe_load(open('diarization/diar_config.yaml'))
print('Segmentation config:', config['pipeline']['segmentation'])
"

# Test 2: Run diarization with default config
uv run python diarization/main.py --audio diarization/input/sample.wav

# Test 3: Override config value from command line
uv run python diarization/main.py --audio diarization/input/sample.wav \
    --segmentation-min-duration-off 0.2
```

## Files Modified

1. ✅ `diarization/compare_clustering_methods.py`
   - Added `apply_segmentation_params()` helper function
   - Updated `run_diarization_standard()` to use config
   - Updated `run_diarization_hierarchical()` to use config

2. ✅ `diarization/compare_dimensions.py`
   - Added `apply_segmentation_params()` helper function
   - Updated `run_diarization_with_dimension()` to use config

3. ✅ `diarization/test_diarization.py`
   - Added `apply_segmentation_params()` helper function
   - Updated `test_diarization_full()` to use config

4. ℹ️ `diarization/main.py`
   - Already properly uses config (no changes needed)

5. ℹ️ `diarization/diar_config.yaml`
   - Already contains segmentation parameters (no changes needed)

## Implementation Details

### Model Type Detection

The helper function automatically detects model type:

```python
is_powerset = pipeline._segmentation.model.specifications.powerset
```

### Parameter Application

- **Powerset models** (speaker-diarization-community-1):
  - Only `min_duration_off` is applied
  - `threshold` is silently ignored (not available)

- **Binary models** (older segmentation models):
  - Both `threshold` and `min_duration_off` are applied

### Fallback Defaults

If config is missing or parameters not specified:
- `threshold`: 0.5
- `min_duration_on`: 0.0
- `min_duration_off`: 0.0

## Future Work

Possible enhancements:
1. Create a shared utility module (`diarization/utils.py`) to avoid duplicating `apply_segmentation_params()` in 3 files
2. Add validation for parameter ranges in the helper function
3. Add warnings if parameters outside recommended ranges

## Related Documentation

- VAD Configuration Guide: `diarization/docs/VAD_CONFIGURATION.md`
- VAD Implementation Summary: `diarization/VAD_CONFIG_SUMMARY.md`
- Config Usage Guide: `diarization/docs/CONFIG_USAGE.md`
