#!/usr/bin/env python3
"""
Compare standard vs hierarchical MRL clustering methods.

This script runs speaker diarization with two clustering methods:
1. Standard clustering: Single 256D embedding clustering
2. Hierarchical MRL: 3-stage progressive refinement (64D → 192D → 256D)

Compares:
- Inference time
- Memory usage
- Number of speakers detected
- Diarization Error Rate (if ground truth provided)

Usage:
    python compare_clustering_methods.py --audio sample.wav
    python compare_clustering_methods.py --audio sample.wav --reference-rttm ground_truth.rttm
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

from checkpoint_utils import get_default_checkpoint_path, get_default_config_path
from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding
from hierarchical_mrl_clustering import PyannoteStyleClustering

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


def apply_segmentation_params(pipeline, config=None, verbose=False):
    """
    Apply segmentation parameters from config to pipeline.

    Args:
        pipeline: pyannote SpeakerDiarization pipeline
        config: Configuration dict (will load default if None)
        verbose: Print parameter info
    """
    if config is None:
        config = load_diar_config()

    pipeline_config = config.get('pipeline', {})
    segmentation_config = pipeline_config.get('segmentation', {})

    # Get parameters with defaults
    threshold = segmentation_config.get('threshold', 0.5)
    min_duration_on = segmentation_config.get('min_duration_on', 0.0)
    min_duration_off = segmentation_config.get('min_duration_off', 0.0)

    # Check if model uses powerset (determines available parameters)
    is_powerset = False
    if hasattr(pipeline, '_segmentation') and hasattr(pipeline._segmentation, 'model'):
        is_powerset = pipeline._segmentation.model.specifications.powerset

    # Apply parameters based on model type
    if hasattr(pipeline, 'segmentation'):
        if is_powerset:
            # Powerset models only have min_duration_off
            if hasattr(pipeline.segmentation, 'min_duration_off'):
                pipeline.segmentation.min_duration_off = min_duration_off
                if verbose:
                    print(f"  Segmentation: min_duration_off={min_duration_off}s (powerset model)")
        else:
            # Non-powerset models have threshold and min_duration_off
            if hasattr(pipeline.segmentation, 'threshold'):
                pipeline.segmentation.threshold = threshold
                if verbose:
                    print(f"  Segmentation: threshold={threshold}")
            if hasattr(pipeline.segmentation, 'min_duration_off'):
                pipeline.segmentation.min_duration_off = min_duration_off
                if verbose:
                    print(f"  Segmentation: min_duration_off={min_duration_off}s")


def measure_memory():
    """Measure current GPU memory usage (if available)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return 0


def run_diarization_standard(
    audio_path,
    segmentation,
    device,
    verbose=False
):
    """
    Run diarization with standard clustering (single 256D).

    Args:
        audio_path: Path to audio file
        segmentation: Pyannote segmentation model
        device: Device for inference
        verbose: Print detailed info

    Returns:
        dict with results
    """
    if verbose:
        print("\nRunning standard clustering (256D)...")

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create embedding model (single dimension)
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=get_default_checkpoint_path(),
        config_path=get_default_config_path(),
        embedding_dim=256,
        extract_all_dims=False,
        device=device
    )

    # Create pipeline with default clustering
    pipeline = SpeakerDiarization(
        segmentation=segmentation,
        embedding=None,
    )
    pipeline._embedding = embedding_model

    # Apply segmentation parameters from config
    apply_segmentation_params(pipeline, verbose=verbose)

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
        'method': 'standard',
        'diarization': diarization,
        'inference_time': inference_time,
        'memory_mb': memory_mb,
        'n_speakers': len(diarization.labels()),
    }


def run_diarization_hierarchical(
    audio_path,
    segmentation,
    device,
    coarse_threshold=None,
    refined_threshold=None,
    boundary_threshold=None,
    verbose=False
):
    """
    Run diarization with hierarchical MRL clustering.

    Args:
        audio_path: Path to audio file
        segmentation: Pyannote segmentation model
        device: Device for inference
        coarse_threshold: Stage 1 threshold (default: from config or 0.6)
        refined_threshold: Stage 2 threshold (default: from config or 0.4)
        boundary_threshold: Stage 3 threshold (default: from config or 0.7)
        verbose: Print detailed info

    Returns:
        dict with results
    """
    if verbose:
        print("\nRunning hierarchical MRL clustering...")

    # Load config for default thresholds
    diar_config = load_diar_config()
    clustering_config = diar_config.get('clustering', {})

    if coarse_threshold is None:
        coarse_threshold = clustering_config.get('coarse_threshold', 0.6)
    if refined_threshold is None:
        refined_threshold = clustering_config.get('refined_threshold', 0.4)
    if boundary_threshold is None:
        boundary_threshold = clustering_config.get('boundary_threshold', 0.7)

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create embedding model (multi-dimension)
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=get_default_checkpoint_path(),
        config_path=get_default_config_path(),
        embedding_dim=256,
        extract_all_dims=True,  # Extract multiple dimensions
        device=device
    )

    # Create pipeline
    pipeline = SpeakerDiarization(
        segmentation=segmentation,
        embedding=None,
    )
    pipeline._embedding = embedding_model

    # Apply segmentation parameters from config
    apply_segmentation_params(pipeline, config=diar_config, verbose=verbose)

    # Override clustering with hierarchical MRL
    clustering = PyannoteStyleClustering(
        method='hierarchical_mrl',
        embedding_model=embedding_model,
        coarse_threshold=coarse_threshold,
        refined_threshold=refined_threshold,
        boundary_threshold=boundary_threshold,
    )
    pipeline.clustering = clustering

    if verbose:
        print(f"  Clustering thresholds: coarse={coarse_threshold}, refined={refined_threshold}, boundary={boundary_threshold}")

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
        'method': 'hierarchical_mrl',
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
        description='Compare standard vs hierarchical MRL clustering'
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
        '--coarse-threshold',
        type=float,
        default=None,
        help='Hierarchical: Stage 1 threshold (default: from config or 0.6)'
    )
    parser.add_argument(
        '--refined-threshold',
        type=float,
        default=None,
        help='Hierarchical: Stage 2 threshold (default: from config or 0.4)'
    )
    parser.add_argument(
        '--boundary-threshold',
        type=float,
        default=None,
        help='Hierarchical: Stage 3 threshold (default: from config or 0.7)'
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
        default='clustering_comparison.txt',
        help='Output file for results'
    )
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print("="*70)
    print("Clustering Method Comparison")
    print("="*70)
    print(f"Audio: {audio_path.name}")
    print(f"Device: {args.device}")
    if args.reference_rttm:
        print(f"Reference: {args.reference_rttm}")
    print()

    device = torch.device(args.device)
    results = {}

    # Run standard clustering
    print("Method 1: Standard Clustering (single 256D)")
    print("-" * 70)
    try:
        result_standard = run_diarization_standard(
            audio_path,
            args.segmentation,
            device,
            verbose=args.verbose
        )
        results['standard'] = result_standard

        if args.reference_rttm:
            der = compute_der(args.reference_rttm, result_standard['diarization'])
            results['standard']['der'] = der

    except Exception as e:
        print(f"ERROR: Standard clustering failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    # Run hierarchical MRL clustering
    print("\nMethod 2: Hierarchical MRL Clustering (64D → 192D → 256D)")
    print("-" * 70)
    try:
        result_hierarchical = run_diarization_hierarchical(
            audio_path,
            args.segmentation,
            device,
            coarse_threshold=args.coarse_threshold,
            refined_threshold=args.refined_threshold,
            boundary_threshold=args.boundary_threshold,
            verbose=args.verbose
        )
        results['hierarchical'] = result_hierarchical

        if args.reference_rttm:
            der = compute_der(args.reference_rttm, result_hierarchical['diarization'])
            results['hierarchical']['der'] = der

    except Exception as e:
        print(f"ERROR: Hierarchical clustering failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    # Print comparison table
    if len(results) == 0:
        print("ERROR: No results to compare")
        sys.exit(1)

    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)

    has_der = args.reference_rttm and any('der' in r for r in results.values())

    # Print header
    if has_der:
        print(f"{'Method':<25} | {'Time (s)':>10} | {'Memory':>10} | {'Speakers':>9} | {'DER (%)':>10} | {'Speedup':>8}")
        print("-" * 90)
    else:
        print(f"{'Method':<25} | {'Time (s)':>10} | {'Memory':>10} | {'Speakers':>9} | {'Speedup':>8}")
        print("-" * 70)

    # Calculate speedup (baseline = standard)
    baseline_time = results.get('standard', {}).get('inference_time', None)

    # Print results
    for method_key in ['standard', 'hierarchical']:
        if method_key not in results:
            continue

        result = results[method_key]
        method_name = "Standard (256D)" if method_key == 'standard' else "Hierarchical MRL"
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
                f"{method_name:<25} | {inference_time:10.2f} | {memory_mb:8.0f}MB | "
                f"{n_speakers:9d} | {der_str} | {speedup:7.2f}x"
            )
        else:
            print(
                f"{method_name:<25} | {inference_time:10.2f} | {memory_mb:8.0f}MB | "
                f"{n_speakers:9d} | {speedup:7.2f}x"
            )

    if has_der:
        print("-" * 90)
    else:
        print("-" * 70)

    # Save results to file
    with open(args.output, 'w') as f:
        f.write("Clustering Method Comparison Results\n")
        f.write("="*70 + "\n")
        f.write(f"Audio: {audio_path.name}\n")
        if args.reference_rttm:
            f.write(f"Reference: {args.reference_rttm}\n")
        f.write("\n")

        if has_der:
            f.write(f"{'Method':<25} | {'Time (s)':>10} | {'Memory':>10} | {'Speakers':>9} | {'DER (%)':>10} | {'Speedup':>8}\n")
        else:
            f.write(f"{'Method':<25} | {'Time (s)':>10} | {'Memory':>10} | {'Speakers':>9} | {'Speedup':>8}\n")
        f.write("-" * 70 + "\n")

        for method_key in ['standard', 'hierarchical']:
            if method_key not in results:
                continue

            result = results[method_key]
            method_name = "Standard (256D)" if method_key == 'standard' else "Hierarchical MRL"
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
                    f"{method_name:<25} | {inference_time:10.2f} | {memory_mb:8.0f}MB | "
                    f"{n_speakers:9d} | {der_str} | {speedup:7.2f}x\n"
                )
            else:
                f.write(
                    f"{method_name:<25} | {inference_time:10.2f} | {memory_mb:8.0f}MB | "
                    f"{n_speakers:9d} | {speedup:7.2f}x\n"
                )

    print(f"\nResults saved to: {args.output}")

    # Print analysis
    if 'standard' in results and 'hierarchical' in results:
        print("\nAnalysis:")
        print("-" * 70)
        speedup = results['standard']['inference_time'] / results['hierarchical']['inference_time']
        print(f"  Speedup: {speedup:.2f}x")

        if has_der and 'der' in results['standard'] and 'der' in results['hierarchical']:
            der_diff = results['hierarchical']['der'] - results['standard']['der']
            print(f"  DER difference: {der_diff:+.2f}% (hierarchical - standard)")
            if abs(der_diff) < 1.0:
                print(f"  → Similar accuracy with {speedup:.2f}x speedup ✓")
            elif der_diff < 0:
                print(f"  → Better accuracy AND faster! ✓✓")
            else:
                print(f"  → Small accuracy tradeoff for speedup")

    print("\n[OK] Comparison completed successfully!")


if __name__ == "__main__":
    main()
