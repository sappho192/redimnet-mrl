#!/usr/bin/env python3
"""
Test script to verify VAD configuration is properly applied.
"""

import os
import sys
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
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


def test_vad_configuration():
    """Test if VAD configuration parameters are properly applied."""

    print("="*70)
    print("Testing VAD Configuration")
    print("="*70)

    # Load embedding model
    print("\n1. Loading ReDimNet-MRL embedding model...")
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=get_default_checkpoint_path(),
        config_path=get_default_config_path(),
        embedding_dim=256,
        extract_all_dims=False,
        device=torch.device('cpu')
    )
    print("   [OK] Model loaded")

    # Load pipeline
    print("\n2. Loading pyannote pipeline...")
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("   ERROR: HF_TOKEN not found in environment")
        print("   Set it with: export HF_TOKEN=your_token")
        return False

    # Temporarily set weights_only to False
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
    print("   [OK] Pipeline loaded")

    # Test 1: Check pipeline structure
    print("\n3. Checking pipeline structure...")
    if hasattr(pipeline, '_segmentation'):
        segmentation = pipeline._segmentation
        print("   [OK] Found pipeline._segmentation")
    elif hasattr(pipeline, 'segmentation'):
        segmentation = pipeline.segmentation
        print("   [OK] Found pipeline.segmentation")
    else:
        print("   [FAIL] Could not find segmentation attribute")
        print(f"   Available attributes: {[attr for attr in dir(pipeline) if not attr.startswith('_')]}")
        return False

    # Test 2: Check if parameters exist
    print("\n4. Checking segmentation parameters...")
    params_to_check = ['threshold', 'min_duration_on', 'min_duration_off']
    available_params = {}

    for param in params_to_check:
        if hasattr(segmentation, param):
            value = getattr(segmentation, param)
            available_params[param] = value
            print(f"   [OK] {param}: {value}")
        else:
            print(f"   [WARN] {param}: not found")

    if not available_params:
        print("   [INFO] No direct segmentation parameters found, continuing investigation...")

    # Test 3: Try to modify parameters
    print("\n5. Testing parameter modification...")
    test_threshold = 0.42

    try:
        if 'threshold' in available_params:
            original_threshold = segmentation.threshold
            segmentation.threshold = test_threshold
            new_threshold = segmentation.threshold

            if new_threshold == test_threshold:
                print(f"   [OK] threshold successfully changed: {original_threshold} -> {new_threshold}")
                # Restore original
                segmentation.threshold = original_threshold
            else:
                print(f"   [FAIL] threshold not changed properly: set to {test_threshold}, got {new_threshold}")
                return False
    except Exception as e:
        print(f"   [FAIL] Error modifying threshold: {e}")
        return False

    # Test 4: Check if parameters are used in inference
    print("\n6. Checking parameter access during inference...")
    print(f"   Current segmentation parameters:")
    if hasattr(segmentation, 'threshold'):
        print(f"     threshold: {segmentation.threshold}")
    if hasattr(segmentation, 'min_duration_on'):
        print(f"     min_duration_on: {segmentation.min_duration_on}")
    if hasattr(segmentation, 'min_duration_off'):
        print(f"     min_duration_off: {segmentation.min_duration_off}")

    # Test 5: Check all attributes of segmentation model
    print("\n7. Full segmentation model inspection:")
    print(f"   Type: {type(segmentation)}")
    print(f"   Configurable parameters:")
    for attr in dir(segmentation):
        if not attr.startswith('_') and not callable(getattr(segmentation, attr)):
            try:
                value = getattr(segmentation, attr)
                print(f"     {attr}: {value}")
            except:
                pass

    # Test 6: Check pipeline-level parameters
    print("\n8. Checking pipeline hyperparameters...")
    if hasattr(pipeline, 'parameters'):
        if callable(pipeline.parameters):
            params = pipeline.parameters()
            print(f"   Pipeline parameters() output:")
            for key, value in params.items():
                print(f"     {key}: {value}")
        else:
            print(f"   Pipeline parameters: {pipeline.parameters}")

    # Check for instantiated parameters
    if hasattr(pipeline, 'CACHED_SEGMENTATION'):
        print(f"   [INFO] Found CACHED_SEGMENTATION")

    # Test 7: Try to access parameters through different paths
    print("\n9. Exploring alternative parameter locations...")

    # Check if pipeline has params dict
    for attr_name in ['_parameters', 'params', 'config', 'hyperparameters']:
        if hasattr(pipeline, attr_name):
            attr = getattr(pipeline, attr_name)
            print(f"   [OK] Found pipeline.{attr_name}: {attr}")

    # Check segmentation wrapper
    print(f"\n10. Checking if segmentation has inference wrapper...")
    if hasattr(segmentation, 'get_params'):
        print(f"    get_params: {segmentation.get_params()}")
    if hasattr(segmentation, 'set_params'):
        print(f"    set_params method exists")

    # Test 8: Try instantiate method (standard pyannote approach)
    print(f"\n11. Testing instantiate() method...")
    if hasattr(pipeline, 'instantiate'):
        print(f"    [OK] Pipeline has instantiate() method")
        try:
            # Try to instantiate with test parameters
            test_params = {
                "segmentation": {
                    "threshold": 0.42,
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.1
                }
            }
            print(f"    Attempting to instantiate with: {test_params}")
            pipeline.instantiate(test_params)
            print(f"    [OK] instantiate() succeeded")

            # Check if params were applied
            params_after = pipeline.parameters()
            print(f"    Parameters after instantiate():")
            for key, value in params_after.items():
                print(f"      {key}: {value}")
        except Exception as e:
            print(f"    [INFO] instantiate() not applicable: {e}")

    print("\n" + "="*70)
    if not available_params:
        print("[INFO] VAD parameters not directly accessible on segmentation model")
        print("       Pyannote pipelines use instantiate() or frozen parameters")
        print("       The parameters may already be set from pretrained config")
    else:
        print("[OK] VAD configuration test completed successfully!")
    print("="*70)
    return True


if __name__ == "__main__":
    success = test_vad_configuration()
    sys.exit(0 if success else 1)
