#!/usr/bin/env python3
"""
Test diarization with specific audio file and compare with ground truth.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os

sys.path.insert(0, str(Path(__file__).parent))

# Load HF token from .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key == 'HF_TOKEN':
                    os.environ['HF_TOKEN'] = value
                    print(f"Loaded HF_TOKEN from .env file")
                    # Login to Hugging Face
                    try:
                        from huggingface_hub import login
                        login(token=value)
                        print("Successfully authenticated with Hugging Face")
                    except Exception as e:
                        print(f"Warning: HF login failed: {e}")

import torch
import torch.serialization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

from checkpoint_utils import get_default_checkpoint_path, get_default_config_path
from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding
from hierarchical_mrl_clustering import PyannoteStyleClustering

try:
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.database.util import load_rttm
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    print("ERROR: pyannote.audio not available")
    sys.exit(1)

# Configuration
OUTPUT_DIR = Path("diarization/output")
VIS_DIR = Path("diarization/visualization")

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)


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

def load_rttm_manual(rttm_path, uri='audio'):
    """Manually parse RTTM file to create Annotation object."""
    annotation = Annotation(uri=uri)

    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            # RTTM format: type file channel start duration ortho speaker_type speaker_name conf1 conf2
            if parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]

                segment = Segment(start, start + duration)
                annotation[segment] = speaker

    return annotation

def compute_der(reference_rttm_path, hypothesis):
    """Compute DER between reference and hypothesis."""
    try:
        # Try pyannote's load_rttm first
        reference_dict = load_rttm(reference_rttm_path)

        if not reference_dict:
            # Fall back to manual parsing if load_rttm returns empty dict
            print("  Using manual RTTM parser...")
            reference = load_rttm_manual(reference_rttm_path, uri='audio')
        else:
            uri = list(reference_dict.keys())[0]
            reference = reference_dict[uri]

        # Compute DER
        metric = DiarizationErrorRate()

        # Create UEM (Unpartitioned Evaluation Map) for the entire duration
        from pyannote.core import Timeline
        extent = hypothesis.get_timeline().extent()
        uem = Timeline([extent])

        # Compute DER with detailed components
        der = metric(reference, hypothesis, uem=uem, detailed=True)

        # Get detailed components from the metric's internal state
        # In newer versions, components are stored differently
        try:
            components = metric.components_
            return {
                'der': der['diarization error rate'] * 100 if isinstance(der, dict) else der * 100,
                'confusion': components.get('confusion', 0) * 100,
                'false_alarm': components.get('false alarm', 0) * 100,
                'missed_detection': components.get('missed detection', 0) * 100,
            }
        except (AttributeError, KeyError):
            # Fall back to just returning overall DER
            return {
                'der': der * 100 if not isinstance(der, dict) else der.get('diarization error rate', 0) * 100,
                'confusion': 0,
                'false_alarm': 0,
                'missed_detection': 0,
            }
    except Exception as e:
        print(f"Warning: Could not compute DER: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_diarization_comparison(reference_rttm_path, hypothesis, output_file, title):
    """Create visualization comparing reference and hypothesis."""
    try:
        # Load reference
        reference_dict = load_rttm(reference_rttm_path)

        if not reference_dict:
            # Fall back to manual parsing
            reference = load_rttm_manual(reference_rttm_path, uri='audio')
        else:
            uri = list(reference_dict.keys())[0]
            reference = reference_dict[uri]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

        # Get unique speakers from both
        ref_speakers = sorted(reference.labels())
        hyp_speakers = sorted(hypothesis.labels())

        # Assign colors
        all_speakers = sorted(set(ref_speakers) | set(hyp_speakers))
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_speakers)))
        color_map = dict(zip(all_speakers, colors))

        # Plot reference
        ax1.set_title('Ground Truth (Human Annotation)', fontsize=12, fontweight='bold')
        for segment, _, speaker in reference.itertracks(yield_label=True):
            color = color_map.get(speaker, 'gray')
            ax1.barh(
                y=0,
                width=segment.duration,
                left=segment.start,
                height=0.8,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            # Add label for longer segments
            if segment.duration > 1.0:
                ax1.text(
                    segment.start + segment.duration / 2,
                    0,
                    speaker,
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold'
                )

        ax1.set_ylim(-0.5, 0.5)
        ax1.set_yticks([])
        ax1.set_ylabel('Speakers', fontsize=10)
        ax1.grid(True, axis='x', alpha=0.3)

        # Plot hypothesis
        ax2.set_title(f'Predicted ({title})', fontsize=12, fontweight='bold')
        for segment, _, speaker in hypothesis.itertracks(yield_label=True):
            # Try to match colors if speaker names are similar
            color = color_map.get(speaker, 'gray')
            ax2.barh(
                y=0,
                width=segment.duration,
                left=segment.start,
                height=0.8,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            # Add label for longer segments
            if segment.duration > 1.0:
                ax2.text(
                    segment.start + segment.duration / 2,
                    0,
                    speaker,
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold'
                )

        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.set_ylabel('Speakers', fontsize=10)
        ax2.set_xlabel('Time (seconds)', fontsize=10)
        ax2.grid(True, axis='x', alpha=0.3)

        # Get total duration
        total_duration = max(
            reference.get_timeline().extent().end,
            hypothesis.get_timeline().extent().end
        )
        ax2.set_xlim(0, total_duration)

        # Add legends
        ref_patches = [
            mpatches.Patch(color=color_map[s], label=f'{s}')
            for s in ref_speakers
        ]
        ax1.legend(handles=ref_patches, loc='upper right', fontsize=8, ncol=min(len(ref_speakers), 5))

        hyp_patches = [
            mpatches.Patch(color=color_map.get(s, 'gray'), label=f'{s}')
            for s in hyp_speakers
        ]
        ax2.legend(handles=hyp_patches, loc='upper right', fontsize=8, ncol=min(len(hyp_speakers), 5))

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved: {output_file}")
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
        import traceback
        traceback.print_exc()

def run_diarization(method_name, clustering_method, audio_file, reference_rttm=None, extract_all_dims=False):
    """Run diarization with specified method."""
    print(f"\n{'='*70}")
    print(f"Testing: {method_name}")
    print(f"{'='*70}")

    # Create embedding model
    print("Loading embedding model...")
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=get_default_checkpoint_path(),
        config_path=get_default_config_path(),
        embedding_dim=256,
        extract_all_dims=extract_all_dims,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print()

    # Create pipeline
    print("Creating diarization pipeline...")
    try:
        # Get HF token from environment
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print("WARNING: HF_TOKEN not found in environment")

        # Load pipeline from pretrained with token
        from pyannote.audio import Pipeline

        # Temporarily set weights_only to False for loading pyannote models
        import functools
        original_load = torch.load

        @functools.wraps(original_load)
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load

        try:
            # Try with token parameter (newer API)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=hf_token
            )
        except TypeError:
            # Fall back to use_auth_token (older API)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                use_auth_token=hf_token
            )
        finally:
            # Restore original torch.load
            torch.load = original_load

        # Override embedding
        pipeline._embedding = embedding_model

        # Load config
        diar_config = load_diar_config()

        # Apply segmentation parameters from config
        apply_segmentation_params(pipeline, config=diar_config, verbose=True)

        # Override clustering if hierarchical
        if clustering_method == 'hierarchical':
            # Load config for thresholds
            clustering_config = diar_config.get('clustering', {})
            coarse_threshold = clustering_config.get('coarse_threshold', 0.6)
            refined_threshold = clustering_config.get('refined_threshold', 0.4)
            boundary_threshold = clustering_config.get('boundary_threshold', 0.7)

            clustering = PyannoteStyleClustering(
                method='hierarchical_mrl',
                embedding_model=embedding_model,
                coarse_threshold=coarse_threshold,
                refined_threshold=refined_threshold,
                boundary_threshold=boundary_threshold,
            )
            pipeline.clustering = clustering
            print(f"  Using Hierarchical MRL Clustering")
            print(f"    Thresholds: coarse={coarse_threshold}, refined={refined_threshold}, boundary={boundary_threshold}")
        else:
            print("  Using Standard Clustering")

        print()

    except Exception as e:
        print(f"ERROR: Failed to create pipeline: {e}")
        print("\nMake sure HF_TOKEN is set:")
        print("  export HF_TOKEN=your_token")
        return None

    # Run diarization
    print(f"Processing audio: {audio_file}")
    import time
    start_time = time.time()

    try:
        diarization = pipeline(str(audio_file))
        inference_time = time.time() - start_time
        print(f"  Inference time: {inference_time:.2f}s")
        print()

    except Exception as e:
        print(f"ERROR: Diarization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get annotation object first
    if hasattr(diarization, 'speaker_diarization'):
        # DiarizeOutput object from newer pyannote versions
        annotation = diarization.speaker_diarization
    elif hasattr(diarization, 'annotation'):
        annotation = diarization.annotation
    elif hasattr(diarization, 'labels'):
        annotation = diarization
    elif isinstance(diarization, dict) and 'annotation' in diarization:
        annotation = diarization['annotation']
    else:
        print(f"Warning: Unknown diarization output type: {type(diarization)}")
        print(f"Available attributes: {dir(diarization)}")
        annotation = None

    # Save results
    output_file = OUTPUT_DIR / f"{Path(audio_file).stem}_{method_name.lower().replace(' ', '_')}.rttm"
    if annotation:
        with open(output_file, 'w') as f:
            annotation.write_rttm(f)
        print(f"Results saved: {output_file}")
    else:
        print(f"ERROR: Could not extract annotation from diarization output")

    # Print statistics
    if annotation:
        print("\nSpeaker Statistics:")
        print("-" * 70)
        total_duration = 0
        for speaker in sorted(annotation.labels()):
            duration = sum(
                segment.duration
                for segment in annotation.label_timeline(speaker)
            )
            total_duration += duration
            percentage = (duration / annotation.get_timeline().extent().end) * 100
            n_segments = len(list(annotation.label_timeline(speaker)))
            print(f"  {speaker:12s}: {duration:6.1f}s ({percentage:5.1f}%) - {n_segments} segments")

        print("-" * 70)
        print(f"  {'Total':12s}: {total_duration:6.1f}s")
        print(f"  {'Speakers':12s}: {len(annotation.labels())}")
        print(f"  {'Duration':12s}: {annotation.get_timeline().extent().end:.1f}s")

    # Compute DER
    der_results = None
    if reference_rttm and Path(reference_rttm).exists():
        print("\nComparing with ground truth...")
        der_results = compute_der(reference_rttm, annotation if annotation else diarization)

    if der_results:
        print("\nDiarization Error Rate (DER):")
        print("-" * 70)
        print(f"  Overall DER:        {der_results['der']:6.2f}%")
        print(f"  Confusion:          {der_results['confusion']:6.2f}%")
        print(f"  False Alarm:        {der_results['false_alarm']:6.2f}%")
        print(f"  Missed Detection:   {der_results['missed_detection']:6.2f}%")
        print("-" * 70)

    # Create visualization
    if reference_rttm and Path(reference_rttm).exists():
        print("\nGenerating visualization...")
        audio_stem = Path(audio_file).stem
        vis_file = VIS_DIR / f"{audio_stem}_{method_name.lower().replace(' ', '_')}.png"
        visualize_diarization_comparison(reference_rttm, annotation if annotation else diarization, vis_file, method_name)

    return {
        'diarization': annotation if annotation else diarization,
        'inference_time': inference_time,
        'der_results': der_results,
    }

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test speaker diarization with ReDimNet-MRL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with single audio file
  python test_diarization.py --audio sample.wav

  # Test with audio file and reference RTTM
  python test_diarization.py --audio sample.wav --reference sample.rttm

  # Test all audio files in a directory
  python test_diarization.py --input-dir diarization/input/

  # Test only hierarchical method
  python test_diarization.py --audio sample.wav --method hierarchical
        """
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--audio',
        help='Single audio file to process'
    )
    input_group.add_argument(
        '--input-dir',
        help='Directory containing audio files (processes all .wav, .flac, .mp3, .m4a files)'
    )

    parser.add_argument(
        '--reference',
        default=None,
        help='Reference RTTM file (for single audio file only)'
    )
    parser.add_argument(
        '--method',
        choices=['standard', 'hierarchical', 'both'],
        default='both',
        help='Which clustering method to test (default: both)'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (default: cuda if available, else cpu)'
    )

    args = parser.parse_args()

    print("="*70)
    print("Speaker Diarization Testing")
    print("="*70)
    print(f"Device: {args.device}")
    print()

    # Collect audio files to process
    audio_files = []
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"ERROR: Audio file not found: {audio_path}")
            return
        audio_files.append(audio_path)

        # Check reference if provided
        if args.reference:
            ref_path = Path(args.reference)
            if not ref_path.exists():
                print(f"WARNING: Reference RTTM not found: {ref_path}")
                args.reference = None

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Input directory not found: {input_dir}")
            return
        if not input_dir.is_dir():
            print(f"ERROR: {input_dir} is not a directory")
            return

        # Find all audio files
        audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
        for ext in audio_extensions:
            audio_files.extend(input_dir.glob(f'*{ext}'))
            audio_files.extend(input_dir.glob(f'*{ext.upper()}'))

        if not audio_files:
            print(f"ERROR: No audio files found in {input_dir}")
            print(f"Supported formats: {', '.join(audio_extensions)}")
            return

        audio_files = sorted(audio_files)
        print(f"Found {len(audio_files)} audio file(s) to process:")
        for af in audio_files:
            print(f"  - {af.name}")
        print()

    # Process each audio file
    for audio_file in audio_files:
        print(f"\n{'='*70}")
        print(f"Processing: {audio_file.name}")
        print(f"{'='*70}")

        # Find reference RTTM if in input-dir mode
        reference_rttm = None
        if args.input_dir:
            # Look for matching RTTM file
            rttm_path = audio_file.with_suffix('.rttm')
            if not rttm_path.exists():
                # Try with _annotated suffix
                rttm_path = audio_file.parent / f"{audio_file.stem}_annotated.rttm"
            if rttm_path.exists():
                reference_rttm = str(rttm_path)
                print(f"Found reference: {rttm_path.name}")
        elif args.reference:
            reference_rttm = args.reference

        # Test selected methods
        results = {}

        # Method 1: Hierarchical MRL
        if args.method in ['hierarchical', 'both']:
            result1 = run_diarization(
                "Hierarchical MRL",
                "hierarchical",
                audio_file,
                reference_rttm=reference_rttm,
                extract_all_dims=True
            )
            if result1:
                results['hierarchical'] = result1

        # Method 2: Standard clustering
        if args.method in ['standard', 'both']:
            result2 = run_diarization(
                "Standard 256D",
                "standard",
                audio_file,
                reference_rttm=reference_rttm,
                extract_all_dims=False
            )
            if result2:
                results['standard'] = result2

        # Print comparison
        if len(results) == 2:
            print("\n" + "="*70)
            print("Comparison Summary")
            print("="*70)
            print(f"{'Method':<25} | {'Time (s)':>10} | {'DER (%)':>10} | {'Speakers':>9} | {'Speedup':>8}")
            print("-" * 70)

            baseline_time = results['standard']['inference_time']

            for method_key in ['standard', 'hierarchical']:
                if method_key not in results:
                    continue

                result = results[method_key]
                method_name = "Standard (256D)" if method_key == 'standard' else "Hierarchical MRL"
                inference_time = result['inference_time']
                n_speakers = len(result['diarization'].labels())

                der = result['der_results']['der'] if result['der_results'] else float('nan')
                speedup = baseline_time / inference_time if baseline_time > 0 else 1.0

                print(
                    f"{method_name:<25} | {inference_time:10.2f} | {der:10.2f} | "
                    f"{n_speakers:9d} | {speedup:7.2f}x"
                )

            print("="*70)

        # Analysis
        if args.method == 'both' and 'hierarchical' in results and 'standard' in results:
            h_time = results['hierarchical']['inference_time']
            s_time = results['standard']['inference_time']
            speedup = s_time / h_time

            h_der = results['hierarchical']['der_results']['der'] if results['hierarchical']['der_results'] else 0
            s_der = results['standard']['der_results']['der'] if results['standard']['der_results'] else 0
            der_diff = h_der - s_der

            print("\nAnalysis:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  DER difference: {der_diff:+.2f}% (hierarchical - standard)")

            if abs(der_diff) < 2.0:
                print(f"  → Similar accuracy with {speedup:.2f}x speedup ✓")
            elif der_diff < 0:
                print(f"  → Better accuracy AND faster! ✓✓")
            else:
                print(f"  → Small accuracy tradeoff for speedup")

    print("\n" + "="*70)
    print("[OK] Testing completed!")
    print("="*70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    if VIS_DIR.exists() and any(VIS_DIR.iterdir()):
        print(f"Visualizations saved to: {VIS_DIR}")

if __name__ == "__main__":
    main()
