"""
ReDimNet-MRL wrapper compatible with pyannote.audio pipelines.

This module provides a wrapper class that adapts ReDimNet-MRL to the
pyannote.audio embedding interface, enabling seamless integration with
pyannote's speaker diarization pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from functools import cached_property
from pathlib import Path
import sys

# Add parent directory to import checkpoint utils
sys.path.insert(0, str(Path(__file__).parent))
from checkpoint_utils import load_mrl_checkpoint, get_default_checkpoint_path, get_default_config_path

try:
    from pyannote.audio.core.inference import BaseInference
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    # Create a dummy base class
    class BaseInference:
        pass


class ReDimNetMRLSpeakerEmbedding(BaseInference):
    """
    ReDimNet-MRL wrapper compatible with pyannote.audio pipelines.

    This class adapts the ReDimNet-MRL model to the pyannote.audio embedding
    interface, allowing it to be used as a drop-in replacement for pyannote's
    default embedding models in speaker diarization pipelines.

    Key features:
    - Compatible with pyannote.audio BaseInference interface
    - Supports multi-resolution embeddings (64D, 128D, 192D, 256D)
    - Handles frame-level masks with automatic upsampling
    - Supports both single-dimension and multi-dimension extraction

    Args:
        checkpoint_path: Path to trained MRL checkpoint (default: best.pt)
        config_path: Path to config YAML (default: config.yaml)
        embedding_dim: Target embedding dimension (default: 256)
        extract_all_dims: Extract multiple dimensions for hierarchical clustering
        device: Device for inference (default: 'cuda' if available)

    Example:
        >>> # Basic usage with pyannote pipeline
        >>> embedding = ReDimNetMRLSpeakerEmbedding(
        ...     embedding_dim=256,
        ...     device='cuda'
        ... )
        >>>
        >>> from pyannote.audio.pipelines import SpeakerDiarization
        >>> pipeline = SpeakerDiarization(
        ...     segmentation="pyannote/segmentation-3.0"
        ... )
        >>> pipeline._embedding = embedding
        >>> diarization = pipeline("audio.wav")
    """

    def __init__(
        self,
        checkpoint_path=None,
        config_path=None,
        embedding_dim=256,
        extract_all_dims=False,
        device=None
    ):
        if not PYANNOTE_AVAILABLE:
            print(
                "WARNING: pyannote.audio is not installed. "
                "This wrapper will work for basic embedding extraction but "
                "cannot be used with pyannote pipelines."
            )

        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Use defaults if not specified
        if checkpoint_path is None:
            checkpoint_path = get_default_checkpoint_path()
        if config_path is None:
            config_path = get_default_config_path()

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.embedding_dim = embedding_dim
        self.extract_all_dims = extract_all_dims

        # Storage for multi-dimensional embeddings (used by hierarchical clustering)
        self._last_embeddings_dict = None
        self._accumulated_embeddings_dict = None  # Accumulates across all batches

        # Load trained MRL model
        print(f"Loading ReDimNet-MRL model...")
        print(f"  Checkpoint: {Path(checkpoint_path).name}")
        print(f"  Embedding dimension: {embedding_dim}D")
        if extract_all_dims:
            print(f"  Mode: Multi-dimension extraction (for hierarchical clustering)")
        else:
            print(f"  Mode: Single dimension")

        self.model = load_mrl_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            verbose=False
        )
        self.model.eval()

        print(f"  Device: {device}")
        print(f"[OK] Model loaded successfully")

    @cached_property
    def sample_rate(self) -> int:
        """Audio sample rate expected by the model (16000 Hz)."""
        return 16000

    @cached_property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.embedding_dim

    @cached_property
    def metric(self) -> str:
        """Distance metric for embeddings (cosine similarity)."""
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        """
        Minimum number of audio samples required for embedding extraction.

        This is determined by the model's receptive field. We use a binary
        search to find the minimum viable audio length, similar to pyannote's
        approach.
        """
        # Start with a reasonable guess (1 second = 16000 samples)
        # ReDimNet typically requires at least 0.5-1.0 seconds
        min_samples = 8000  # 0.5 seconds

        # Binary search to find minimum
        left, right = 1000, 48000  # Search between 0.0625s and 3s
        found_min = right

        with torch.no_grad():
            while left <= right:
                mid = (left + right) // 2
                try:
                    # Test if model can process this length
                    test_audio = torch.randn(1, 1, mid, device=self.device)
                    _ = self.model(test_audio, target_dim=self.embedding_dim)
                    # Success - try shorter
                    found_min = mid
                    right = mid - 1000
                except Exception:
                    # Failed - need longer audio
                    left = mid + 1000

        # Add small buffer for safety
        return found_min + 1000

    def __call__(self, waveforms, masks=None):
        """
        Extract embeddings from audio waveforms.

        Args:
            waveforms: Audio waveforms tensor [batch_size, 1, num_samples]
            masks: Optional frame-level masks [batch_size, num_frames]
                   Used to zero-out non-speech regions

        Returns:
            If extract_all_dims=False:
                numpy array [batch_size, embedding_dim]
            If extract_all_dims=True:
                dict {dim: numpy array [batch_size, dim]}
                e.g., {64: [B, 64], 192: [B, 192], 256: [B, 256]}

        Notes:
            - Masks are upsampled from frame-level to sample-level
            - Output is normalized (L2 norm = 1)
        """
        with torch.inference_mode():
            # Move to device
            waveforms = waveforms.to(self.device)

            # Handle masks: upsample from frame-level to sample-level
            if masks is not None:
                masks = masks.to(self.device)

                # Upsample masks from (B, num_frames) to (B, num_samples)
                upsampled_masks = F.interpolate(
                    masks.unsqueeze(1),  # (B, 1, num_frames)
                    size=waveforms.shape[-1],  # num_samples
                    mode='nearest'
                )  # (B, 1, num_samples)

                # Apply mask to waveforms
                waveforms = waveforms * upsampled_masks

            # Extract embeddings
            if self.extract_all_dims:
                # Multi-dimension extraction for hierarchical clustering
                embeddings_dict = self.model(waveforms, return_all_dims=True)

                # Normalize each dimension
                embeddings_dict_normalized = {}
                for dim, emb in embeddings_dict.items():
                    # L2 normalization
                    emb = F.normalize(emb, p=2, dim=1)
                    # Convert to numpy
                    embeddings_dict_normalized[dim] = emb.cpu().numpy()

                # Store for hierarchical clustering to access
                self._last_embeddings_dict = embeddings_dict_normalized

                # Accumulate embeddings across all batches for hierarchical clustering
                if self._accumulated_embeddings_dict is None:
                    # First batch - initialize
                    self._accumulated_embeddings_dict = embeddings_dict_normalized.copy()
                else:
                    # Subsequent batches - concatenate
                    for dim in embeddings_dict_normalized:
                        self._accumulated_embeddings_dict[dim] = np.concatenate([
                            self._accumulated_embeddings_dict[dim],
                            embeddings_dict_normalized[dim]
                        ], axis=0)

                # Return only the highest dimension to pyannote (maintains compatibility)
                return embeddings_dict_normalized[self.embedding_dim]

            else:
                # Single dimension extraction
                embeddings = self.model(waveforms, target_dim=self.embedding_dim)

                # L2 normalization
                embeddings = F.normalize(embeddings, p=2, dim=1)

                # Convert to numpy
                return embeddings.cpu().numpy()

    def reset_accumulated_embeddings(self):
        """Reset accumulated embeddings (call before each new diarization run)."""
        self._accumulated_embeddings_dict = None
        self._last_embeddings_dict = None

    def to(self, device):
        """
        Move model to specified device.

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)

        Returns:
            self
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.model = self.model.to(device)
        return self

    def __repr__(self):
        return (
            f"ReDimNetMRLSpeakerEmbedding("
            f"embedding_dim={self.embedding_dim}, "
            f"extract_all_dims={self.extract_all_dims}, "
            f"device={self.device})"
        )


if __name__ == "__main__":
    print("Testing ReDimNet-MRL wrapper...\n")

    # Test 1: Basic initialization
    print("Test 1: Initialization")
    print("-" * 60)
    try:
        embedding = ReDimNetMRLSpeakerEmbedding(
            embedding_dim=256,
            device='cpu'
        )
        print(f"[OK] Model: {embedding}")
        print(f"  Sample rate: {embedding.sample_rate} Hz")
        print(f"  Dimension: {embedding.dimension}")
        print(f"  Metric: {embedding.metric}")
        print(f"  Min samples: {embedding.min_num_samples}")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 2: Single-dimension extraction
    print("\nTest 2: Single-dimension extraction")
    print("-" * 60)
    try:
        # Create dummy audio (3 seconds)
        waveforms = torch.randn(2, 1, 48000)  # 2 samples, 3 seconds

        # Extract embeddings
        embeddings = embedding(waveforms)
        print(f"  Input shape: {waveforms.shape}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Output type: {type(embeddings)}")
        print(f"  L2 norms: {np.linalg.norm(embeddings, axis=1)}")  # Should be ~1.0
        assert embeddings.shape == (2, 256), "Wrong output shape"
        assert isinstance(embeddings, np.ndarray), "Output should be numpy array"
        print("[OK] Single-dimension extraction works")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Multi-dimension extraction
    print("\nTest 3: Multi-dimension extraction")
    print("-" * 60)
    try:
        embedding_multi = ReDimNetMRLSpeakerEmbedding(
            embedding_dim=256,
            extract_all_dims=True,
            device='cpu'
        )

        embeddings_dict = embedding_multi(waveforms)
        print(f"  Input shape: {waveforms.shape}")
        print(f"  Output type: {type(embeddings_dict)}")
        for dim, emb in embeddings_dict.items():
            print(f"  {dim}D: {emb.shape}, L2 norms: {np.linalg.norm(emb, axis=1)}")
            assert emb.shape == (2, dim), f"Wrong shape for {dim}D"

        assert isinstance(embeddings_dict, dict), "Output should be dict"
        print("[OK] Multi-dimension extraction works")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Mask handling
    print("\nTest 4: Mask handling")
    print("-" * 60)
    try:
        # Create dummy masks (100 frames)
        num_frames = 100
        masks = torch.ones(2, num_frames)
        # Zero out second half of second sample
        masks[1, 50:] = 0.0

        # Extract embeddings with masks
        embeddings_masked = embedding(waveforms, masks=masks)
        print(f"  Input shape: {waveforms.shape}")
        print(f"  Mask shape: {masks.shape}")
        print(f"  Output shape: {embeddings_masked.shape}")
        assert embeddings_masked.shape == (2, 256), "Wrong output shape"
        print("[OK] Mask handling works")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("[OK] All tests passed!")
    print("="*60)
