#!/usr/bin/env python3
import sys
from pathlib import Path
import os
import inspect

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

print(f"Pipeline type: {type(pipeline)}")
print(f"Pipeline class: {pipeline.__class__.__name__}")
print(f"Pipeline module: {pipeline.__class__.__module__}")

print("\nPipeline MRO (Method Resolution Order):")
for cls in pipeline.__class__.__mro__[:5]:
    print(f"  {cls}")

print("\nChecking for apply() method:")
if hasattr(pipeline, 'apply'):
    print("  Found apply() method")
    sig = inspect.signature(pipeline.apply)
    print(f"  Signature: {sig}")

print("\nChecking for __call__() implementation:")
if hasattr(pipeline, '__call__'):
    try:
        source = inspect.getsource(pipeline.__call__)
        # Print first 50 lines
        lines = source.split('\n')[:50]
        for line in lines:
            print(f"  {line}")
    except:
        print("  Could not get source")

print("\nLooking for 'cluster' in method names:")
for attr in dir(pipeline):
    if 'cluster' in attr.lower() or 'kluster' in attr.lower():
        val = getattr(pipeline, attr)
        print(f"  {attr}: {type(val).__name__}")
        if attr == 'clustering' and not callable(val):
            print(f"    Value: {val}")
        if callable(val) and not attr.startswith('__'):
            try:
                sig = inspect.signature(val)
                print(f"    Signature: {sig}")
            except:
                pass

print("\nChecking if 'clustering' attribute exists:")
if hasattr(pipeline, 'clustering'):
    clustering = pipeline.clustering
    print(f"  Type: {type(clustering)}")
    print(f"  Callable: {callable(clustering)}")
    if hasattr(clustering, '__call__'):
        try:
            sig = inspect.signature(clustering)
            print(f"  Signature: {sig}")
        except:
            pass
