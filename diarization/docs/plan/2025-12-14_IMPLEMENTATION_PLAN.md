# Implementation Plan: Speaker Diarization with ReDimNet-MRL

## Overview

Create a speaker diarization system that integrates ReDimNet-MRL embeddings into the pyannote.audio pipeline architecture. The implementation will replace pyannote's default embedding model with ReDimNet-MRL while using pyannote's segmentation component. Additionally, we will implement a novel **Hierarchical Multi-Resolution Clustering** approach that leverages MRL's multi-dimensional embeddings for improved speed-accuracy tradeoff.

## Architecture

### Standard Pipeline
```
Audio File
    ↓
Pyannote Segmentation Model (detect speech regions)
    ↓
ReDimNet-MRL Embedding Extraction (single dimension)
    ↓
Pyannote Clustering (group speakers)
    ↓
Diarization Output (RTTM format)
```

### Hierarchical MRL Pipeline (Recommended)
```
Audio File
    ↓
Pyannote Segmentation (detect speech regions)
    ↓
ReDimNet-MRL Multi-Dimension Embeddings (64D, 192D, 256D simultaneously)
    ↓
Hierarchical MRL Clustering (3-stage progressive refinement)
    ├─ Stage 1 (64D): Fast coarse separation of distinct speakers
    ├─ Stage 2 (192D): Refined sub-clustering within coarse groups
    └─ Stage 3 (256D): Boundary verification for uncertain samples
    ↓
Diarization Output (RTTM format)
```

## Key Innovation: Hierarchical Multi-Resolution Clustering

### Motivation
Traditional clustering applies a single algorithm at one embedding dimension, resulting in O(n²) complexity on high-dimensional spaces. MRL's hierarchical structure (64D ⊂ 128D ⊂ 192D ⊂ 256D) enables **progressive refinement**:
- Use low dimensions (64D) for fast, coarse separation
- Use medium dimensions (192D) for balanced refinement
- Use high dimensions (256D) only for boundary verification

### Expected Performance Gains
- **Speed**: ~1.5-1.7x faster (most computation in 64D space)
- **Accuracy**: ~1% DER increase (acceptable for practical use)
- **Scalability**: Larger gains with more speakers/segments

## Project Structure

```
redimnet-mrl/
├── diarization/
│   ├── PLAN.md                              # This file
│   ├── README.md                            # User documentation
│   ├── __init__.py                          # Package initialization
│   │
│   ├── redimnet_wrapper.py                  # Pyannote interface adapter
│   ├── hierarchical_mrl_clustering.py       # Novel clustering algorithm
│   ├── checkpoint_utils.py                  # Model loading utilities
│   │
│   ├── main.py                              # Complete diarization example
│   ├── compare_dimensions.py                # Compare 64D/128D/192D/256D
│   ├── compare_clustering_methods.py        # Hierarchical vs baseline
│   ├── visualize_clustering_stages.py       # Visualize 3-stage clustering
│   │
│   └── test_wrapper.py                      # Unit tests
│
├── model.py                                 # ReDimNetMRL model
├── pretrained.py                            # Model loading functions
├── config.yaml                              # Training configuration
├── checkpoints/
│   └── mrl_redimnet/
│       └── best.pt                          # Best trained checkpoint
└── ...
```

## Key Integration Points

### Pyannote Embedding Interface Requirements
All embedding models must implement (from `BaseInference`):
- `sample_rate: int` - Audio sample rate (16000 for ReDimNet-MRL)
- `dimension: int` - Embedding dimension (64/128/192/256)
- `metric: str` - Distance metric ("cosine")
- `min_num_samples: int` - Minimum audio samples needed
- `__call__(waveforms, masks) -> np.ndarray` - Extract embeddings
- `to(device)` - Device management

### ReDimNet-MRL Model Interface
- Input: `(batch_size, 1, num_samples)` at 16kHz
- Output: `(batch_size, dimension)` tensor or `Dict[int, Tensor]` for multi-dim
- Supports multiple dimensions via `target_dim` parameter or `return_all_dims=True`
- Best checkpoint: `../checkpoints/mrl_redimnet/best.pt`

## Files to Create

### 1. `redimnet_wrapper.py` (Priority: HIGH)
**Purpose:** Wrapper class that adapts ReDimNet-MRL to pyannote interface

**Key Components:**
```python
class ReDimNetMRLSpeakerEmbedding(BaseInference):
    """ReDimNet-MRL wrapper compatible with pyannote.audio pipelines"""

    def __init__(
        self,
        checkpoint_path,
        config_path,
        embedding_dim=256,
        extract_all_dims=False,  # For hierarchical clustering
        device=None
    ):
        # Load trained MRL model from checkpoint
        # Set target embedding dimension
        # Move to device

    @cached_property
    def sample_rate(self) -> int:
        return 16000

    @cached_property
    def dimension(self) -> int:
        return self.embedding_dim

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        # Binary search to find minimum viable audio length
        # Similar to PyannoteAudioPretrainedSpeakerEmbedding

    def __call__(self, waveforms, masks=None):
        # waveforms: (B, 1, num_samples) torch.Tensor
        # masks: (B, num_frames) torch.Tensor or None
        #
        # Returns:
        #   If extract_all_dims=False: (B, dimension) numpy array
        #   If extract_all_dims=True: Dict[int, np.ndarray] with keys [64, 192, 256]

        with torch.inference_mode():
            waveforms = waveforms.to(self.device)

            if masks is not None:
                # Upsample masks from frame-level to sample-level
                masks = masks.to(self.device)
                upsampled_masks = F.interpolate(
                    masks.unsqueeze(1),  # (B, 1, num_frames)
                    size=waveforms.shape[-1],
                    mode='nearest'
                )  # (B, 1, num_samples)
                waveforms = waveforms * upsampled_masks

            if self.extract_all_dims:
                # For hierarchical clustering: extract multiple dimensions
                embeddings_dict = self.model(waveforms, return_all_dims=True)
                # Convert to numpy: {64: np.ndarray, 192: np.ndarray, 256: np.ndarray}
                return {dim: emb.cpu().numpy() for dim, emb in embeddings_dict.items()}
            else:
                # Standard mode: single dimension
                embeddings = self.model(waveforms, target_dim=self.embedding_dim)
                return embeddings.cpu().numpy()

    def to(self, device):
        # Move model to device
```

**Critical Implementation Details:**
- **Mask Handling**: Upsample frame-level masks to sample-level using `F.interpolate()`
- **Multi-dimension Support**: Return dict when `extract_all_dims=True` for hierarchical clustering
- **Device Management**: Ensure all tensors on same device

### 2. `hierarchical_mrl_clustering.py` (Priority: HIGH)
**Purpose:** Novel 3-stage clustering algorithm leveraging MRL embeddings

**Key Components:**
```python
class HierarchicalMRLClustering:
    """
    Multi-stage clustering using multiple resolution embeddings.

    Exploits the hierarchical nature of MRL embeddings (64D ⊂ 192D ⊂ 256D)
    to achieve better speed-accuracy tradeoff than single-dimension clustering.

    Algorithm:
        Stage 1 (64D): Fast coarse separation
            - Use agglomerative clustering with loose threshold (0.6)
            - Quickly separate clearly distinct speakers
            - O(n²) on 64-dimensional space (4x smaller than 256D)

        Stage 2 (192D): Refined sub-clustering
            - Within each coarse cluster, apply tighter clustering (threshold 0.4)
            - Split clusters that contain multiple speakers
            - O(k * m²) where k=num_coarse_clusters, m=avg_cluster_size << n

        Stage 3 (256D): Boundary verification
            - Compute cluster centroids
            - Identify low-confidence samples (similarity < 0.7 to own centroid)
            - Reassign boundary samples to nearest centroid
            - Only processes boundary samples, not all n samples

    Performance:
        - Computational complexity: O(n²/4) + O(Σm²) + O(b*k) where b=boundary samples
        - Expected speedup: 1.5-1.7x compared to single 256D clustering
        - Expected accuracy: -0.5~1% DER (minimal degradation)
    """

    def __init__(
        self,
        coarse_threshold: float = 0.6,    # Stage 1: loose grouping
        refined_threshold: float = 0.4,   # Stage 2: tighter clustering
        boundary_threshold: float = 0.7,  # Stage 3: confidence threshold
        min_cluster_size: int = 2,
    ):
        pass

    def __call__(
        self,
        embeddings_dict: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform hierarchical clustering on multi-resolution embeddings.

        Args:
            embeddings_dict: Dictionary mapping dimension to embeddings
                {64: (N, 64), 192: (N, 192), 256: (N, 256)}

        Returns:
            labels: (N,) array of cluster labels
            metadata: Dictionary with intermediate results and statistics
                - labels_coarse: Stage 1 output
                - labels_refined: Stage 2 output
                - stats_coarse: Stage 1 statistics
                - stats_refined: Stage 2 statistics
                - stats_final: Stage 3 statistics
        """
        pass

    def _stage1_coarse_clustering(self, embeddings: np.ndarray):
        """Stage 1: Fast coarse separation with 64D embeddings."""
        pass

    def _stage2_refined_clustering(self, embeddings: np.ndarray, coarse_labels: np.ndarray):
        """Stage 2: Refined sub-clustering within coarse groups."""
        pass

    def _stage3_boundary_verification(self, embeddings: np.ndarray, refined_labels: np.ndarray):
        """Stage 3: Verify and reassign boundary samples using 256D."""
        pass


class PyannoteStyleClustering:
    """Wrapper to make HierarchicalMRLClustering compatible with pyannote pipelines."""

    def __init__(self, method: str = "hierarchical_mrl", **kwargs):
        pass

    def __call__(self, embeddings, **kwargs):
        """
        Args:
            embeddings: Can be either:
                - Dict[int, np.ndarray] for hierarchical MRL
                - np.ndarray for single-dimension clustering
        """
        pass
```

**Algorithm Details:**
- **Stage 1**: Agglomerative clustering on 64D with threshold 0.6
- **Stage 2**: For each coarse cluster, sub-cluster with threshold 0.4 on 192D
- **Stage 3**: Identify boundary samples (cosine similarity < 0.7) and reassign using 256D

### 3. `checkpoint_utils.py` (Priority: HIGH)
**Purpose:** Utilities for loading trained MRL checkpoints

**Functions:**
```python
def load_mrl_checkpoint(checkpoint_path, config_path, device='cpu'):
    """
    Load trained MRL model from checkpoint.

    Steps:
    1. Load config YAML to get model architecture params
    2. Create model using create_mrl_from_pretrained() or ReDimNetMRL()
    3. Load trained weights from checkpoint
    4. Set to eval mode
    5. Return initialized model
    """

def get_default_checkpoint_path():
    """Return path to best trained checkpoint"""
    return "../checkpoints/mrl_redimnet/best.pt"

def get_default_config_path():
    """Return path to training config"""
    return "../config.yaml"
```

**Critical Files to Reference:**
- `../pretrained.py` (lines 86-135): `create_mrl_from_pretrained()` function
- `../config.yaml`: Model architecture configuration
- `../checkpoints/mrl_redimnet/best.pt`: Best trained checkpoint

### 4. `main.py` (Priority: HIGH)
**Purpose:** Complete diarization example script

**Features:**
- Command-line argument parsing
- Audio file loading
- Pipeline construction with ReDimNet-MRL
- Support for both standard and hierarchical clustering
- Results saving in RTTM format
- Optional visualization

**Usage:**
```bash
# Standard clustering (single 256D)
python main.py --audio sample.wav --clustering-method pyannote_default --device cuda

# Hierarchical MRL clustering (recommended)
python main.py --audio sample.wav --clustering-method hierarchical_mrl --device cuda

# Custom thresholds
python main.py --audio sample.wav \
    --clustering-method hierarchical_mrl \
    --coarse-threshold 0.6 \
    --refined-threshold 0.4 \
    --boundary-threshold 0.7
```

**Key Implementation:**
```python
from pyannote.audio.pipelines import SpeakerDiarization
from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding
from hierarchical_mrl_clustering import PyannoteStyleClustering

# Parse arguments
parser = argparse.ArgumentParser(description='Speaker Diarization with ReDimNet-MRL')
parser.add_argument('--audio', required=True, help='Input audio file')
parser.add_argument('--output', default='output.rttm', help='Output RTTM file')
parser.add_argument('--clustering-method', default='hierarchical_mrl',
                    choices=['pyannote_default', 'hierarchical_mrl'])
parser.add_argument('--embedding-dim', type=int, default=256)
parser.add_argument('--coarse-threshold', type=float, default=0.6)
parser.add_argument('--refined-threshold', type=float, default=0.4)
parser.add_argument('--boundary-threshold', type=float, default=0.7)
parser.add_argument('--device', default='cuda')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

# Create embedding model
embedding_model = ReDimNetMRLSpeakerEmbedding(
    checkpoint_path=get_default_checkpoint_path(),
    config_path=get_default_config_path(),
    embedding_dim=args.embedding_dim,
    extract_all_dims=(args.clustering_method == 'hierarchical_mrl'),
    device=torch.device(args.device)
)

# Create pipeline
pipeline = SpeakerDiarization(
    segmentation="pyannote/segmentation-3.0",
    embedding=None,  # Will be overridden
)
pipeline._embedding = embedding_model

# Override clustering if using hierarchical MRL
if args.clustering_method == 'hierarchical_mrl':
    clustering = PyannoteStyleClustering(
        method='hierarchical_mrl',
        coarse_threshold=args.coarse_threshold,
        refined_threshold=args.refined_threshold,
        boundary_threshold=args.boundary_threshold,
    )
    pipeline._clustering = clustering

# Apply to audio
print(f"Processing: {args.audio}")
diarization = pipeline(args.audio)

# Save results (RTTM format)
with open(args.output, "w") as f:
    diarization.write_rttm(f)
print(f"Results saved to: {args.output}")

# Print statistics
print(f"\nSpeaker statistics:")
for speaker in diarization.labels():
    duration = sum(segment.duration for segment in diarization.label_timeline(speaker))
    print(f"  {speaker}: {duration:.1f}s")
```

### 5. `compare_dimensions.py` (Priority: MEDIUM)
**Purpose:** Compare performance across different MRL dimensions (64D, 128D, 192D, 256D)

**Features:**
- Run diarization with each dimension independently
- Measure inference time for each dimension
- Compare memory usage
- Generate comparison plots

**Expected Output:**
```
Dimension | Inference Time | Memory | Speakers Detected | Speedup
----------|----------------|--------|-------------------|--------
64D       | 0.12s         | 512MB  | 2                 | 2.0x
128D      | 0.15s         | 548MB  | 2                 | 1.6x
192D      | 0.18s         | 584MB  | 2                 | 1.3x
256D      | 0.24s         | 620MB  | 2                 | 1.0x
```

### 6. `compare_clustering_methods.py` (Priority: MEDIUM)
**Purpose:** Compare standard clustering vs hierarchical MRL clustering

**Features:**
- Run both clustering methods on same audio
- Measure performance metrics (DER, inference time, memory)
- Generate comparison tables and plots
- Support ground truth RTTM for DER calculation

**Expected Output:**
```
Method                  | Inference Time | DER    | Speakers | Speedup
------------------------|----------------|--------|----------|--------
Pyannote Default (256D) | 10.0s         | 12.5%  | 3        | 1.0x
Hierarchical MRL        | 6.5s          | 13.2%  | 3        | 1.5x
```

**Usage:**
```bash
python compare_clustering_methods.py \
    --audio sample.wav \
    --reference-rttm ground_truth.rttm \
    --output comparison_results.png
```

### 7. `visualize_clustering_stages.py` (Priority: LOW)
**Purpose:** Visualize the hierarchical clustering process

**Features:**
- Show how clusters evolve through 3 stages
- Use t-SNE to project embeddings to 2D
- Color-code clusters at each stage
- Highlight reassigned samples

**Output:**
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  64D Input  │ Stage 1:64D │ Stage 2:192D│ Stage 3:256D│
│  (no labels)│  (coarse)   │  (refined)  │   (final)   │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 8. `README.md` (Priority: MEDIUM)
**Purpose:** Documentation and usage guide

**Sections:**
- Overview of speaker diarization and ReDimNet-MRL
- Installation instructions
- Quick start guide
- Usage examples (standard vs hierarchical clustering)
- Performance benchmarks
- Troubleshooting
- References

### 9. `test_wrapper.py` (Priority: LOW)
**Purpose:** Unit tests for wrapper class

**Test Cases:**
- Model loading from checkpoint
- Property access (sample_rate, dimension, metric)
- Inference with different batch sizes
- Mask handling (with/without masks)
- Device switching (CPU ↔ CUDA)
- Output format validation (shape, dtype)
- Multi-dimension extraction (`extract_all_dims=True`)

## Implementation Steps

### Step 1: Setup Project Structure
- [x] Create `diarization/` directory
- [ ] Create `__init__.py`
- [ ] Add dependencies to main project

### Step 2: Implement Checkpoint Loading (`checkpoint_utils.py`)
- [ ] Write `load_mrl_checkpoint()` function
- [ ] Handle config loading from YAML
- [ ] Use `create_mrl_from_pretrained()` from `../pretrained.py`
- [ ] Load trained weights from checkpoint
- [ ] Add error handling for missing files

**Critical Files to Reference:**
- `/home/user/redimnet-mrl/pretrained.py` (lines 86-135)
- `/home/user/redimnet-mrl/config.yaml`

### Step 3: Implement Wrapper Class (`redimnet_wrapper.py`)
- [ ] Create `ReDimNetMRLSpeakerEmbedding` class
- [ ] Inherit from `pyannote.audio.core.inference.BaseInference`
- [ ] Implement all required properties and methods
- [ ] Handle mask upsampling and application
- [ ] Add proper device management
- [ ] Support multi-dimension extraction for hierarchical clustering

**Critical Dependencies:**
- Requires pyannote.audio installed
- May need to reference pyannote source code for interface details

### Step 4: Implement Hierarchical Clustering (`hierarchical_mrl_clustering.py`)
- [ ] Create `HierarchicalMRLClustering` class
- [ ] Implement Stage 1: Coarse clustering with 64D
- [ ] Implement Stage 2: Refined clustering with 192D
- [ ] Implement Stage 3: Boundary verification with 256D
- [ ] Create `PyannoteStyleClustering` wrapper for pipeline compatibility
- [ ] Add logging for each stage

### Step 5: Create Basic Example Script (`main.py`)
- [ ] Add command-line argument parsing (argparse)
- [ ] Load audio file using pyannote's `Audio` class
- [ ] Create `SpeakerDiarization` pipeline
- [ ] Replace embedding model with ReDimNet-MRL wrapper
- [ ] Support both clustering methods (standard vs hierarchical)
- [ ] Apply pipeline to audio
- [ ] Save results in RTTM format
- [ ] Print speaker statistics

### Step 6: Test with Sample Audio
- [ ] Find or create test audio file with multiple speakers
- [ ] Run basic diarization example (standard clustering)
- [ ] Run hierarchical MRL clustering
- [ ] Verify output format (RTTM)
- [ ] Check for errors and edge cases

### Step 7: Add Visualization Support (`main.py`)
- [ ] Add `--visualize` flag
- [ ] Generate matplotlib plot of speaker segments
- [ ] Color-code different speakers
- [ ] Save as PNG/PDF

### Step 8: Implement Dimension Comparison (`compare_dimensions.py`)
- [ ] Loop over dimensions [64, 128, 192, 256]
- [ ] Run diarization for each dimension
- [ ] Measure inference time and memory
- [ ] Generate comparison table and plots

### Step 9: Implement Clustering Comparison (`compare_clustering_methods.py`)
- [ ] Run both clustering methods on same audio
- [ ] Measure inference time, memory, DER (if ground truth available)
- [ ] Generate comparison tables
- [ ] Create plots showing performance trade-offs

### Step 10: Add Clustering Visualization (`visualize_clustering_stages.py`)
- [ ] Extract intermediate clustering results from metadata
- [ ] Use t-SNE to project embeddings to 2D
- [ ] Create 4-panel plot (input → stage1 → stage2 → stage3)
- [ ] Highlight reassigned samples

### Step 11: Write Documentation (`README.md`)
- [ ] Installation instructions
- [ ] Quick start guide
- [ ] Detailed usage examples
- [ ] Performance benchmarks
- [ ] Troubleshooting tips
- [ ] References to papers and repositories

### Step 12: Add Unit Tests (`test_wrapper.py`)
- [ ] Test model loading
- [ ] Test inference with dummy data
- [ ] Test mask handling
- [ ] Test device management
- [ ] Test property accessors
- [ ] Test multi-dimension extraction

### Step 13: Polish and Optimize
- [ ] Add progress bars (tqdm)
- [ ] Improve error messages
- [ ] Add logging
- [ ] Optimize batch processing
- [ ] Profile performance bottlenecks
- [ ] Add type hints

## Key Technical Challenges & Solutions

### Challenge 1: Mask Temporal Resolution Mismatch
**Problem:** Pyannote masks are frame-level (e.g., 10ms), waveforms are sample-level (0.0625ms at 16kHz)

**Solution:** Use `F.interpolate()` with `mode='nearest'` to upsample masks to match waveform length

### Challenge 2: Model Weight Loading
**Problem:** Need to load both architecture config and trained weights

**Solution:**
1. Load YAML config to get model parameters
2. Create model using `create_mrl_from_pretrained()` or `ReDimNetMRL()`
3. Load checkpoint with `torch.load()`
4. Load state dict: `model.load_state_dict(checkpoint['model_state_dict'])`

### Challenge 3: Minimum Samples Computation
**Problem:** `min_num_samples` property requires binary search

**Solution:** Use `@cached_property` decorator and implement binary search to find minimum audio length that produces valid embeddings

### Challenge 4: Device Management
**Problem:** Model and inputs must be on same device

**Solution:** Implement `to(device)` method and ensure all tensors moved to `self.device` in `__call__()`

### Challenge 5: Pyannote Pipeline Integration
**Problem:** Pyannote pipelines expect specific interfaces and data formats

**Solution:**
- Carefully inherit from `BaseInference`
- Return numpy arrays (not torch tensors) from `__call__()`
- For hierarchical clustering, override `pipeline._clustering` with custom implementation

### Challenge 6: Multi-dimension Embedding Extraction
**Problem:** Standard pyannote interface expects single embedding array, but hierarchical clustering needs multiple dimensions

**Solution:**
- Add `extract_all_dims` parameter to wrapper
- When `True`, return `Dict[int, np.ndarray]` instead of `np.ndarray`
- Custom clustering handles dict format

## Dependencies

Add to main project's `pyproject.toml` or `requirements.txt`:
```toml
dependencies = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "pyannote-audio>=3.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]
```

**Note:** Pyannote.audio may require additional dependencies or authentication for pretrained models.

## Expected Results

After implementation, users should be able to:

### 1. Run basic diarization (standard clustering):
```bash
python diarization/main.py --audio sample.wav --clustering-method pyannote_default
```

Output:
```
Loading ReDimNet-MRL model (256D)...
Processing audio: sample.wav
Duration: 120.5 seconds
Extracting embeddings...
Clustering speakers...
Found 3 speakers

Results saved to: output.rttm

Speaker statistics:
  SPEAKER_00: 45.2s (37.5%)
  SPEAKER_01: 38.7s (32.1%)
  SPEAKER_02: 36.6s (30.4%)
```

### 2. Run hierarchical MRL clustering (recommended):
```bash
python diarization/main.py --audio sample.wav --clustering-method hierarchical_mrl
```

Output:
```
Loading ReDimNet-MRL model (multi-dimension mode)...
Processing audio: sample.wav
Duration: 120.5 seconds

Extracting embeddings at multiple resolutions...
  64D embeddings: [N, 64]
  192D embeddings: [N, 192]
  256D embeddings: [N, 256]

Hierarchical MRL Clustering:
  Stage 1: Coarse clustering (64D)...
    → 5 coarse clusters
  Stage 2: Refined clustering (192D)...
    → 3 refined clusters
  Stage 3: Boundary verification (256D)...
    → 3 final speakers
    → 12 samples reassigned

Results saved to: output.rttm

Speaker statistics:
  SPEAKER_00: 45.8s (38.0%)
  SPEAKER_01: 38.2s (31.7%)
  SPEAKER_02: 36.5s (30.3%)

Inference time: 6.5s (1.54x faster than standard)
```

### 3. Compare dimensions:
```bash
python diarization/compare_dimensions.py --audio sample.wav
```

### 4. Compare clustering methods:
```bash
python diarization/compare_clustering_methods.py \
    --audio sample.wav \
    --reference-rttm ground_truth.rttm
```

### 5. Get output in standard RTTM format:
```
SPEAKER sample 1 0.00 5.23 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER sample 1 5.50 8.91 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER sample 1 9.15 12.33 <NA> <NA> SPEAKER_00 <NA> <NA>
```

## Performance Targets

### Hierarchical MRL Clustering vs Standard (256D)
| Metric | Standard | Hierarchical MRL | Change |
|--------|----------|------------------|--------|
| Inference Time | 10.0s | 6.5s | **-35%** ✓ |
| DER | 12.5% | 13.2% | +0.7% (acceptable) |
| Memory | 620MB | 650MB | +30MB (3 embeddings) |
| Speakers Detected | 3 | 3 | Same |

### Dimension Comparison (Standard Clustering Only)
| Dimension | EER (Verification) | Inference Time | Speedup |
|-----------|-------------------|----------------|---------|
| 64D | 9.6% | 5.0s | 2.0x |
| 128D | 7.6% | 6.3s | 1.6x |
| 192D | 6.0% | 7.7s | 1.3x |
| 256D | 5.6% | 10.0s | 1.0x |

**Note:** EER is for speaker verification (1-to-1), not diarization (clustering). Diarization performance (DER) will be measured during testing.

## Critical Files Reference

### From ReDimNet-MRL (this repository):
- `/home/user/redimnet-mrl/pretrained.py`
  - Lines 86-212: `create_mrl_from_pretrained()` function
  - Use for model initialization
- `/home/user/redimnet-mrl/model.py`
  - Lines 90-240: `ReDimNetMRL` class
  - Lines 168-202: `forward()` method with `target_dim` and `return_all_dims` parameters
- `/home/user/redimnet-mrl/config.yaml`
  - Model architecture configuration
- `/home/user/redimnet-mrl/checkpoints/mrl_redimnet/best.pt`
  - Best trained checkpoint (22.5MB, 7.2% EER on VoxCeleb1)

### From Pyannote.audio (external):
Reference implementations (may need to view source code or docs):
- `pyannote.audio.pipelines.speaker_verification.PyannoteAudioPretrainedSpeakerEmbedding`
  - Reference for wrapper class structure and interface
- `pyannote.audio.pipelines.speaker_diarization.SpeakerDiarization`
  - Shows how embeddings are extracted with masks
- `pyannote.audio.core.inference.BaseInference`
  - Base class to inherit from

## Success Criteria

- [ ] ReDimNet-MRL wrapper class implements full pyannote interface
- [ ] Basic diarization example runs successfully (standard clustering)
- [ ] Hierarchical MRL clustering runs successfully
- [ ] Output matches RTTM format specification
- [ ] All four MRL dimensions (64, 128, 192, 256) work correctly
- [ ] Hierarchical clustering shows measurable speedup (>1.3x)
- [ ] Hierarchical clustering maintains acceptable accuracy (DER increase <2%)
- [ ] Comparison scripts show performance trade-offs
- [ ] Visualization shows 3-stage clustering process
- [ ] Documentation covers installation and usage
- [ ] Code is well-commented and maintainable
- [ ] Unit tests pass

## Future Enhancements

### Phase 2 (After initial implementation):
- [ ] Support for online/streaming diarization
- [ ] Integration with other segmentation models
- [ ] Speaker overlap detection
- [ ] Fine-tuning on diarization-specific datasets
- [ ] Export to other formats (Audacity labels, JSON, etc.)
- [ ] Web interface for easy usage

### Phase 3 (Research extensions):
- [ ] Adaptive threshold selection for hierarchical clustering
- [ ] Learned clustering (replace hand-crafted thresholds)
- [ ] Multi-modal diarization (audio + video)
- [ ] Cross-language evaluation
- [ ] Benchmark on standard datasets (AMI, CALLHOME, DIHARD)

## References

1. **ReDimNet**: "Rethinking Dimensionality in Speaker Embeddings" (original paper)
2. **Matryoshka Representation Learning**: Kusupati et al., "Matryoshka Representation Learning"
3. **Pyannote.audio**: Bredin et al., "pyannote.audio: neural building blocks for speaker diarization"
4. **Speaker Diarization**: "Who Spoke When?" - overview of diarization systems

## Notes

- This plan assumes pyannote.audio 3.0+ is available and properly configured
- Pretrained pyannote models may require authentication (HuggingFace token)
- Hierarchical clustering thresholds (0.6, 0.4, 0.7) are initial estimates and should be tuned on validation data
- The 1.5x speedup target is conservative; actual gains may be higher with more speakers/segments
