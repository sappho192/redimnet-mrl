#!/usr/bin/env python3
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent))

# Load HF token
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key == 'HF_TOKEN':
                    os.environ['HF_TOKEN'] = value

import torch
from pyannote.audio import Pipeline

# Patch torch.load
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
        token=os.environ.get('HF_TOKEN')
    )
finally:
    torch.load = original_load

print("Pipeline attributes containing 'clust' (including private):")
for attr in dir(pipeline):
    if 'clust' in attr.lower():
        val = getattr(pipeline, attr)
        print(f"  {attr}: {type(val).__name__} = {val if not callable(val) else '(callable)'}")

print("\nPipeline attributes containing 'embed':")
for attr in dir(pipeline):
    if 'embed' in attr.lower():
        val = getattr(pipeline, attr)
        print(f"  {attr}: {type(val).__name__}")

print("\nAll private attributes starting with '_':")
for attr in sorted(dir(pipeline)):
    if attr.startswith('_') and not attr.startswith('__'):
        try:
            val = getattr(pipeline, attr)
            if not callable(val):
                print(f"  {attr}: {type(val).__name__}")
        except:
            pass

print("\nContents of _inferences:")
if hasattr(pipeline, '_inferences'):
    for key, val in pipeline._inferences.items():
        print(f"  {key}: {type(val).__name__}")

print("\nContents of _instantiated:")
if hasattr(pipeline, '_instantiated'):
    for key, val in pipeline._instantiated.items():
        print(f"  {key}: {type(val).__name__} - {val}")
