# VAD (Voice Activity Detection) Configuration

## Overview

The pyannote segmentation models have different configurable parameters depending on their architecture type. Understanding these differences is important for properly configuring your diarization pipeline.

## Model Types

### Powerset Models (Multi-Label)
- **Example**: `pyannote/speaker-diarization-community-1` (default)
- **VAD Approach**: Learned by the neural network, not threshold-based
- **Configurable Parameters**: Only `min_duration_off`
- **VAD Threshold**: Not available - the model learns to detect speech

### Binary/Multilabel Models
- **Example**: Older segmentation models like `pyannote/segmentation-3.0`
- **VAD Approach**: Threshold-based binary classification
- **Configurable Parameters**: `threshold`, `min_duration_off`
- **VAD Threshold**: Available and tunable

## Available Parameters

### `min_duration_off` (All Models)

**Purpose**: Minimum duration of silence between speech segments

**Effect**:
- Silences shorter than this threshold will merge adjacent segments
- Higher values = fewer, longer segments
- Lower values = more, shorter segments with brief pauses preserved

**Range**: 0.0 - 1.0 seconds

**Default**: 0.0 (no merging)

**When to adjust**:
- Increase (e.g., 0.1-0.3) if you have many short segments with brief pauses
- Keep at 0.0 for maximum temporal resolution

### `threshold` (Binary Models Only)

**Purpose**: VAD threshold for speech detection

**Effect**:
- Lower values = more speech detected (may include non-speech)
- Higher values = more conservative (may miss some speech)

**Range**: 0.1 - 0.9

**Default**: 0.5

**When to adjust**:
- Decrease (e.g., 0.3-0.4) if missing speech in noisy conditions
- Increase (e.g., 0.6-0.7) if detecting too much non-speech

**Note**: NOT available for powerset models like `speaker-diarization-community-1`

### `min_duration_on` (Post-processing Only)

**Purpose**: Minimum duration of speech segments

**Effect**: Segments shorter than this will be filtered out

**Range**: 0.0+ seconds

**Default**: 0.0 (no filtering)

**Note**: Applied during post-processing, not real-time inference

## Configuration

### Via Config File (`diar_config.yaml`)

```yaml
pipeline:
  segmentation_model: pyannote/speaker-diarization-community-1
  segmentation:
    threshold: 0.5             # Only for binary models
    min_duration_on: 0.0       # Post-processing filter
    min_duration_off: 0.0      # Main parameter for powerset models
```

### Via Command-Line Arguments

```bash
# For powerset models (speaker-diarization-community-1)
python diarization/main.py --audio audio.wav \
    --segmentation-min-duration-off 0.1

# For binary models (if using older segmentation)
python diarization/main.py --audio audio.wav \
    --segmentation-threshold 0.4 \
    --segmentation-min-duration-off 0.1
```

## How the System Detects Model Type

The code automatically detects whether the model is powerset or binary:

```python
if pipeline._segmentation.model.specifications.powerset:
    # Powerset model - only min_duration_off available
    pipeline.segmentation.min_duration_off = value
else:
    # Binary model - threshold and min_duration_off available
    pipeline.segmentation.threshold = value
    pipeline.segmentation.min_duration_off = value
```

When you run diarization, you'll see output like:

```
Segmentation parameters:
  Model type: powerset (multi-label)
  Min duration off: 0.0s
  [INFO] Powerset models don't have threshold parameter
  [INFO] VAD is learned by the model, not controlled by threshold
```

## Testing Your Configuration

### Quick Test

Run the parameter detection test to verify everything works:

```bash
uv run python diarization/test_param_detection.py
```

Expected output:
```
✓ All tests passed!

Summary:
  - Model type: Powerset
  - Tunable parameters: min_duration_off
  - VAD threshold: Not available (learned by model)
  - Main control: min_duration_off (merges segments)
```

### Full Diarization Test

Test with actual audio:

```bash
# Test default parameters
uv run python diarization/main.py --audio input/sample.wav

# Test with custom min_duration_off
uv run python diarization/main.py --audio input/sample.wav \
    --segmentation-min-duration-off 0.2
```

## Common Use Cases

### Case 1: Too Many Short Segments

**Problem**: Getting many very short segments with brief pauses

**Solution**: Increase `min_duration_off`

```bash
python main.py --audio audio.wav --segmentation-min-duration-off 0.2
```

### Case 2: Missing Speech in Noisy Audio (Binary Models Only)

**Problem**: Speech not being detected in noisy conditions

**Solution**: Lower the threshold (binary models only)

```bash
# Only works with binary models, not speaker-diarization-community-1
python main.py --audio audio.wav --segmentation-threshold 0.3
```

**Note**: For powerset models, you cannot adjust VAD sensitivity. Consider:
- Using a different segmentation model (binary type)
- Audio preprocessing (noise reduction)
- Fine-tuning the segmentation model

### Case 3: Filtering Very Short Segments

**Problem**: Want to remove segments shorter than 0.5 seconds

**Solution**: Use `min_duration_on`

```bash
python main.py --audio audio.wav --segmentation-min-duration-on 0.5
```

## Limitations

### Powerset Models (speaker-diarization-community-1)

**Cannot adjust**:
- VAD threshold (learned by model)
- Speech detection sensitivity
- False positive/negative tradeoff

**Can adjust**:
- `min_duration_off`: Segment merging
- `min_duration_on`: Post-processing filtering

**Workarounds if you need threshold control**:
1. Use a binary segmentation model instead
2. Fine-tune the powerset model on your data
3. Apply preprocessing to audio (e.g., noise reduction)

### Binary Models

**Cannot adjust**:
- The underlying neural network behavior
- Model-specific biases

**Can adjust**:
- `threshold`: VAD sensitivity
- `min_duration_off`: Segment merging
- `min_duration_on`: Post-processing filtering

## FAQ

### Q: Why can't I change the VAD threshold for speaker-diarization-community-1?

**A**: This model uses powerset architecture where VAD is learned end-to-end by the neural network, not determined by a threshold. The model outputs speaker probabilities directly without a binary VAD step.

### Q: How do I know which parameters are available for my model?

**A**: Run the test script or check the output when running diarization:

```bash
uv run python diarization/test_param_detection.py
```

The script will tell you the model type and available parameters.

### Q: Can I use a different segmentation model with threshold support?

**A**: Yes! Use the `--segmentation` argument:

```bash
python main.py --audio audio.wav \
    --segmentation "pyannote/segmentation-3.0" \
    --segmentation-threshold 0.4
```

Note: You may need to accept different terms of use for other models.

### Q: What's the recommended `min_duration_off` value?

**A**:
- Default (0.0): Maximum resolution, preserves all pauses
- Conservative (0.1-0.2): Merges very short pauses, good for most use cases
- Aggressive (0.3-0.5): Merges longer pauses, good for very fragmented speech

Start with 0.1 and adjust based on your results.

## Implementation Details

### Code Location

- Main implementation: `diarization/main.py:455-485`
- Parameter detection: Checks `pipeline._segmentation.model.specifications.powerset`
- Parameter application: Sets `pipeline.segmentation.min_duration_off` (and `threshold` for binary models)

### Parameter Flow

1. Load from config file (`diar_config.yaml`)
2. Override with command-line arguments (if provided)
3. Detect model type (powerset vs binary)
4. Apply appropriate parameters to pipeline
5. Display applied parameters in output

## Related Files

- Configuration: `diarization/diar_config.yaml`
- Main script: `diarization/main.py`
- Tests: `diarization/test_param_detection.py`, `diarization/test_vad_config.py`
- This document: `diarization/docs/VAD_CONFIGURATION.md`
