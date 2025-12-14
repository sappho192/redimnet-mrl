#!/usr/bin/env python3
"""
Compare diarization performance across different MRL dimensions.

This script runs speaker diarization with different embedding dimensions
(64D, 128D, 192D, 256D) and compares their performance in terms of:
- Inference time
- Memory usage
- Number of speakers detected
- Diarization quality (if ground truth is provided)

Usage:
    python compare_dimensions.py --audio sample.wav
    python compare_dimensions.py --audio sample.wav --reference-rttm ground_truth.rttm
"""

import argparse
import sys
import time
from pathlib import Path
import warnings
import yaml
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from checkpoint_utils import get_default_checkpoint_path, get_default_config_path
from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding

try:
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.metrics.diarization import DiarizationErrorRate
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("ERROR: pyannote.audio is not installed")
    sys.exit(1)


def load_diar_config(config_path=None):
    """Load diarization configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'diar_config.yaml'

    if not Path(config_path).exists():
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load diarization config: {e}")
        return {}


def measure_memory():
    """Measure current GPU memory usage (if available)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return 0


def run_diarization_with_dimension(
    audio_path,
    dimension,
    segmentation,
    device,
    checkpoint_path=None,
    config_path=None,
    verbose=False
):
    """
    Run diarization with a specific embedding dimension.

    Args:
        audio_path: Path to audio file
        dimension: Embedding dimension (64, 128, 192, 256)
        segmentation: Pyannote segmentation model
        device: Device for inference
        checkpoint_path: Path to model checkpoint (optional)
        config_path: Path to model config (optional)
        verbose: Print detailed info

    Returns:
        dict with results (diarization, inference_time, memory_mb)
    """
    if verbose:
        print(f"\nRunning diarization with {dimension}D embeddings...")

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create embedding model
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=checkpoint_path or get_default_checkpoint_path(),
        config_path=config_path or get_default_config_path(),
        embedding_dim=dimension,
        extract_all_dims=False,
        device=device
    )

    # Create pipeline
    pipeline = SpeakerDiarization(
        segmentation=segmentation,
        embedding=None,
    )
    pipeline._embedding = embedding_model

    # Run diarization
    start_time = time.time()
    diarization = pipeline(str(audio_path))
    inference_time = time.time() - start_time

    # Measure memory
    memory_mb = measure_memory()

    if verbose:
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Memory: {memory_mb:.0f} MB")
        print(f"  Speakers: {len(diarization.labels())}")

    return {
        'diarization': diarization,
        'inference_time': inference_time,
        'memory_mb': memory_mb,
        'n_speakers': len(diarization.labels()),
    }


def compute_der(reference_rttm, hypothesis):
    """
    Compute Diarization Error Rate.

    Args:
        reference_rttm: Path to reference RTTM file
        hypothesis: pyannote Annotation object

    Returns:
        DER as percentage
    """
    try:
        from pyannote.database.util import load_rttm
        reference = load_rttm(reference_rttm)[next(iter(load_rttm(reference_rttm)))]

        metric = DiarizationErrorRate()
        der = metric(reference, hypothesis)
        return der * 100
    except Exception as e:
        print(f"WARNING: Failed to compute DER: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Compare diarization performance across MRL dimensions'
    )
    parser.add_argument('--audio', required=True, help='Input audio file')
    parser.add_argument(
        '--reference-rttm',
        default=None,
        help='Reference RTTM for DER computation (optional)'
    )
    parser.add_argument(
        '--diar-config',
        default=str(Path(__file__).parent / 'diar_config.yaml'),
        help='Path to diarization config YAML (default: diar_config.yaml)'
    )
    parser.add_argument(
        '--dimensions',
        nargs='+',
        type=int,
        default=[64, 128, 192, 256],
        help='Dimensions to compare (default: 64 128 192 256)'
    )
    parser.add_argument(
        '--segmentation',
        default='pyannote/speaker-diarization-community-1',
        help='Pyannote segmentation model'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference'
    )
    parser.add_argument(
        '--output',
        default='dimension_comparison.txt',
        help='Output file for results'
    )
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load diarization config
    diar_config = load_diar_config(args.diar_config)

    # Get checkpoint and config paths from config file if available
    checkpoint_path = None
    config_path = None
    if diar_config:
        embedding_config = diar_config.get('embedding', {})
        if 'checkpoint_path' in embedding_config:
            checkpoint_path = str(Path(__file__).parent / embedding_config['checkpoint_path'])
        if 'config_path' in embedding_config:
            config_path = str(Path(__file__).parent / embedding_config['config_path'])

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print("="*70)
    print("MRL Dimension Comparison")
    print("="*70)
    print(f"Audio: {audio_path.name}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Device: {args.device}")
    if args.reference_rttm:
        print(f"Reference: {args.reference_rttm}")
    if Path(args.diar_config).exists():
        print(f"Config: {Path(args.diar_config).name}")
    print()

    # Run diarization for each dimension
    results = {}
    device = torch.device(args.device)

    for dim in args.dimensions:
        try:
            result = run_diarization_with_dimension(
                audio_path,
                dim,
                args.segmentation,
                device,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                verbose=args.verbose
            )
            results[dim] = result

            # Compute DER if reference provided
            if args.reference_rttm:
                der = compute_der(args.reference_rttm, result['diarization'])
                results[dim]['der'] = der

        except Exception as e:
            print(f"ERROR: Failed for {dim}D: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

    # Print comparison table
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)

    # Determine columns
    has_der = args.reference_rttm and any('der' in r for r in results.values())

    # Print header
    if has_der:
        print(f"{'Dim':>6} | {'Time (s)':>10} | {'Memory (MB)':>12} | {'Speakers':>9} | {'DER (%)':>10} | {'Speedup':>8}")
        print("-" * 70)
    else:
        print(f"{'Dim':>6} | {'Time (s)':>10} | {'Memory (MB)':>12} | {'Speakers':>9} | {'Speedup':>8}")
        print("-" * 70)

    # Find baseline (256D) for speedup calculation
    baseline_time = results.get(256, {}).get('inference_time', None)

    # Print rows
    for dim in sorted(args.dimensions):
        if dim not in results:
            continue

        result = results[dim]
        inference_time = result['inference_time']
        memory_mb = result['memory_mb']
        n_speakers = result['n_speakers']

        # Calculate speedup
        if baseline_time and baseline_time > 0:
            speedup = baseline_time / inference_time
        else:
            speedup = 1.0

        if has_der:
            der = result.get('der', None)
            der_str = f"{der:8.1f}%" if der is not None else "N/A".center(10)
            print(
                f"{dim:>5}D | {inference_time:10.2f} | {memory_mb:12.0f} | "
                f"{n_speakers:9d} | {der_str} | {speedup:7.2f}x"
            )
        else:
            print(
                f"{dim:>5}D | {inference_time:10.2f} | {memory_mb:12.0f} | "
                f"{n_speakers:9d} | {speedup:7.2f}x"
            )

    print("="*70)

    # Save results to file
    with open(args.output, 'w') as f:
        f.write("MRL Dimension Comparison Results\n")
        f.write("="*70 + "\n")
        f.write(f"Audio: {audio_path.name}\n")
        f.write(f"Dimensions: {args.dimensions}\n")
        if args.reference_rttm:
            f.write(f"Reference: {args.reference_rttm}\n")
        f.write("\n")

        # Write table
        if has_der:
            f.write(f"{'Dim':>6} | {'Time (s)':>10} | {'Memory (MB)':>12} | {'Speakers':>9} | {'DER (%)':>10} | {'Speedup':>8}\n")
        else:
            f.write(f"{'Dim':>6} | {'Time (s)':>10} | {'Memory (MB)':>12} | {'Speakers':>9} | {'Speedup':>8}\n")
        f.write("-" * 70 + "\n")

        for dim in sorted(args.dimensions):
            if dim not in results:
                continue

            result = results[dim]
            inference_time = result['inference_time']
            memory_mb = result['memory_mb']
            n_speakers = result['n_speakers']

            if baseline_time and baseline_time > 0:
                speedup = baseline_time / inference_time
            else:
                speedup = 1.0

            if has_der:
                der = result.get('der', None)
                der_str = f"{der:8.1f}%" if der is not None else "N/A".center(10)
                f.write(
                    f"{dim:>5}D | {inference_time:10.2f} | {memory_mb:12.0f} | "
                    f"{n_speakers:9d} | {der_str} | {speedup:7.2f}x\n"
                )
            else:
                f.write(
                    f"{dim:>5}D | {inference_time:10.2f} | {memory_mb:12.0f} | "
                    f"{n_speakers:9d} | {speedup:7.2f}x\n"
                )

    print(f"\nResults saved to: {args.output}")
    print("\n[OK] Comparison completed successfully!")


if __name__ == "__main__":
    main()
