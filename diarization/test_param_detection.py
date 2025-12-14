#!/usr/bin/env python3
"""
Quick test to verify parameter detection works correctly.
"""

import os
import sys
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

# Load .env
def _load_dotenv(dotenv_path: Path) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    except Exception:
        pass

    try:
        if not dotenv_path.exists() or not dotenv_path.is_file():
            return

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if not key:
                continue
            if key in os.environ:
                continue
            os.environ[key] = value
    except Exception:
        return

_load_dotenv(Path(__file__).parent / ".env")

from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding
from checkpoint_utils import get_default_checkpoint_path, get_default_config_path

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("ERROR: pyannote.audio is not installed")
    sys.exit(1)


def test_parameter_detection():
    """Test if parameter detection logic works correctly."""

    print("="*70)
    print("Testing Segmentation Parameter Detection")
    print("="*70)

    # Load embedding model
    print("\n[1/4] Loading ReDimNet-MRL...")
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=get_default_checkpoint_path(),
        config_path=get_default_config_path(),
        embedding_dim=256,
        extract_all_dims=False,
        device=torch.device('cpu')
    )
    print("   ✓ Model loaded")

    # Load pipeline
    print("\n[2/4] Loading pyannote pipeline...")
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("   ✗ ERROR: HF_TOKEN not found")
        return False

    import functools
    original_load = torch.load

    @functools.wraps(original_load)
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=hf_token
        )
    finally:
        torch.load = original_load

    pipeline._embedding = embedding_model
    print("   ✓ Pipeline loaded")

    # Test parameter detection
    print("\n[3/4] Detecting model type and available parameters...")

    # Check if model uses powerset
    is_powerset = False
    if hasattr(pipeline, '_segmentation') and hasattr(pipeline._segmentation, 'model'):
        is_powerset = pipeline._segmentation.model.specifications.powerset
        print(f"   Model type: {'powerset (multi-label)' if is_powerset else 'binary'}")
        print(f"   ✓ Detection successful")
    else:
        print("   ✗ Could not detect model type")
        return False

    # Check available parameters
    print("\n[4/4] Checking available parameters...")
    if hasattr(pipeline, 'segmentation'):
        available_params = []
        if hasattr(pipeline.segmentation, 'threshold'):
            available_params.append('threshold')
        if hasattr(pipeline.segmentation, 'min_duration_on'):
            available_params.append('min_duration_on')
        if hasattr(pipeline.segmentation, 'min_duration_off'):
            available_params.append('min_duration_off')

        print(f"   Available parameters: {', '.join(available_params) if available_params else 'none'}")

        # Verify expectations
        if is_powerset:
            if 'min_duration_off' in available_params and 'threshold' not in available_params:
                print("   ✓ Powerset model has correct parameters")
            else:
                print("   ✗ Unexpected parameters for powerset model")
                return False
        else:
            if 'threshold' in available_params and 'min_duration_off' in available_params:
                print("   ✓ Binary model has correct parameters")
            else:
                print("   ✗ Unexpected parameters for binary model")
                return False

        # Test setting a parameter
        if 'min_duration_off' in available_params:
            original_value = pipeline.segmentation.min_duration_off
            test_value = 0.123
            pipeline.segmentation.min_duration_off = test_value
            if pipeline.segmentation.min_duration_off == test_value:
                print(f"   ✓ Successfully set min_duration_off: {original_value} → {test_value}")
                pipeline.segmentation.min_duration_off = original_value  # Restore
            else:
                print("   ✗ Failed to set parameter")
                return False

    else:
        print("   ✗ segmentation attribute not found")
        return False

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    print("\nSummary:")
    print(f"  - Model type: {'Powerset' if is_powerset else 'Binary'}")
    print(f"  - Tunable parameters: {', '.join(available_params)}")
    if is_powerset:
        print(f"  - VAD threshold: Not available (learned by model)")
        print(f"  - Main control: min_duration_off (merges segments)")
    else:
        print(f"  - VAD threshold: Available")
        print(f"  - Main control: threshold (speech detection)")

    return True


if __name__ == "__main__":
    success = test_parameter_detection()
    sys.exit(0 if success else 1)
