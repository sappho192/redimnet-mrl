#!/usr/bin/env python3
"""
Speaker Diarization with ReDimNet-MRL

Complete example script for performing speaker diarization using ReDimNet-MRL
embeddings integrated with the pyannote.audio pipeline.

Usage:
    # Standard clustering (single 256D)
    python main.py --audio sample.wav --clustering-method pyannote_default

    # Hierarchical MRL clustering (recommended)
    python main.py --audio sample.wav --clustering-method hierarchical_mrl

    # Custom thresholds
    python main.py --audio sample.wav \\
        --clustering-method hierarchical_mrl \\
        --coarse-threshold 0.6 \\
        --refined-threshold 0.4 \\
        --boundary-threshold 0.7

    # Custom segmentation parameters
    # Note: speaker-diarization-community-1 only supports min_duration_off
    python main.py --audio sample.wav \\
        --segmentation-min-duration-off 0.1  # Merge segments with <0.1s gaps
"""

import argparse
import os
import sys
import time
from pathlib import Path
import warnings
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def _load_dotenv(dotenv_path: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

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

import torch

# Import our modules
from checkpoint_utils import get_default_checkpoint_path, get_default_config_path
from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding
from hierarchical_mrl_clustering import PyannoteStyleClustering

# Try to import pyannote
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines import SpeakerDiarization
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("ERROR: pyannote.audio is not installed")
    print("Please install it with: pip install pyannote.audio")
    sys.exit(1)


def load_diar_config(config_path):
    """Load diarization configuration from YAML file."""
    if not config_path or not Path(config_path).exists():
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load diarization config from {config_path}: {e}")
        return {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Speaker Diarization with ReDimNet-MRL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python main.py --audio audio.wav

  # Use custom diarization config
  python main.py --audio audio.wav --diar-config my_config.yaml

  # Override config with command-line arguments
  python main.py --audio audio.wav --diar-config diar_config.yaml --coarse-threshold 0.65

  # Standard clustering (no hierarchical)
  python main.py --audio audio.wav --clustering-method pyannote_default

  # Custom segmentation (merge short silences)
  python main.py --audio audio.wav --segmentation-min-duration-off 0.1

  # Save visualization
  python main.py --audio audio.wav --visualize --vis-output diarization.png
        """
    )

    # Required arguments
    parser.add_argument(
        '--audio',
        required=True,
        help='Input audio file (WAV, MP3, etc.)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        default='output.rttm',
        help='Output RTTM file (default: output.rttm)'
    )

    # Configuration file
    parser.add_argument(
        '--diar-config',
        default=str(Path(__file__).parent / 'diar_config.yaml'),
        help='Path to diarization config YAML (default: diar_config.yaml)'
    )

    # Model arguments
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Path to MRL checkpoint (default: best.pt from training)'
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to model config YAML (default: config.yaml)'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=None,
        choices=[64, 128, 192, 256],
        help='Embedding dimension (default: from config or 256)'
    )

    # Clustering arguments
    parser.add_argument(
        '--clustering-method',
        default=None,
        choices=['pyannote_default', 'hierarchical_mrl'],
        help='Clustering method (default: from config or hierarchical_mrl)'
    )
    parser.add_argument(
        '--coarse-threshold',
        type=float,
        default=None,
        help='Stage 1 coarse clustering threshold (default: from config or 0.6)'
    )
    parser.add_argument(
        '--refined-threshold',
        type=float,
        default=None,
        help='Stage 2 refined clustering threshold (default: from config or 0.4)'
    )
    parser.add_argument(
        '--boundary-threshold',
        type=float,
        default=None,
        help='Stage 3 boundary verification threshold (default: from config or 0.7)'
    )

    # Segmentation arguments
    parser.add_argument(
        '--segmentation',
        default='pyannote/speaker-diarization-community-1',
        help='Pyannote segmentation model (default: pyannote/speaker-diarization-community-1)'
    )
    parser.add_argument(
        '--segmentation-threshold',
        type=float,
        default=None,
        help='VAD threshold for non-powerset models only (default: from config or 0.5). '
             'Ignored for powerset models like speaker-diarization-community-1'
    )
    parser.add_argument(
        '--segmentation-min-duration-on',
        type=float,
        default=None,
        help='Minimum speech segment duration in seconds (default: from config or 0.0). '
             'Applied during post-processing'
    )
    parser.add_argument(
        '--segmentation-min-duration-off',
        type=float,
        default=None,
        help='Minimum silence duration between segments in seconds (default: from config or 0.0). '
             'Main tunable parameter for powerset models'
    )

    # Hardware arguments
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (default: cuda if available, else cpu)'
    )

    # Visualization arguments
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of diarization results'
    )
    parser.add_argument(
        '--vis-output',
        default='diarization.png',
        help='Visualization output file (default: diarization.png)'
    )

    # Other arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )

    return parser.parse_args()


def visualize_diarization(diarization, audio_file, output_file):
    """
    Visualize diarization results.

    Args:
        diarization: pyannote Annotation object
        audio_file: Path to audio file
        output_file: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("WARNING: matplotlib not installed, skipping visualization")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))

    # Get unique speakers and assign colors
    speakers = list(diarization.labels())
    colors = plt.cm.Set3(range(len(speakers)))
    speaker_colors = dict(zip(speakers, colors))

    # Plot each speaker segment
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        ax.barh(
            y=0,
            width=segment.duration,
            left=segment.start,
            height=0.8,
            color=speaker_colors[speaker],
            edgecolor='black',
            linewidth=0.5
        )

        # Add speaker label in the middle of long segments
        if segment.duration > 2.0:
            ax.text(
                segment.start + segment.duration / 2,
                0,
                speaker,
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold'
            )

    # Formatting
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, diarization.get_timeline().extent().end)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Speaker Diarization: {Path(audio_file).name}', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    # Add legend
    patches = [
        mpatches.Patch(color=speaker_colors[speaker], label=speaker)
        for speaker in speakers
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=10)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved to: {output_file}")
    plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Load diarization config
    diar_config = load_diar_config(args.diar_config)

    # Apply config defaults (command-line args override config)
    if diar_config:
        clustering_config = diar_config.get('clustering', {})
        embedding_config = diar_config.get('embedding', {})
        pipeline_config = diar_config.get('pipeline', {})
        segmentation_config = pipeline_config.get('segmentation', {})

        # Apply clustering defaults
        if args.clustering_method is None:
            args.clustering_method = clustering_config.get('method', 'hierarchical_mrl')
        if args.coarse_threshold is None:
            args.coarse_threshold = clustering_config.get('coarse_threshold', 0.6)
        if args.refined_threshold is None:
            args.refined_threshold = clustering_config.get('refined_threshold', 0.4)
        if args.boundary_threshold is None:
            args.boundary_threshold = clustering_config.get('boundary_threshold', 0.7)

        # Apply embedding defaults
        if args.embedding_dim is None:
            args.embedding_dim = embedding_config.get('dimension', 256)

        # Apply segmentation defaults
        if args.segmentation_threshold is None:
            args.segmentation_threshold = segmentation_config.get('threshold', 0.5)
        if args.segmentation_min_duration_on is None:
            args.segmentation_min_duration_on = segmentation_config.get('min_duration_on', 0.0)
        if args.segmentation_min_duration_off is None:
            args.segmentation_min_duration_off = segmentation_config.get('min_duration_off', 0.0)

        # Apply checkpoint/config paths if not specified
        if args.checkpoint is None and 'checkpoint_path' in embedding_config:
            checkpoint_path = embedding_config['checkpoint_path']
            args.checkpoint = str(Path(__file__).parent / checkpoint_path)
        if args.config is None and 'config_path' in embedding_config:
            config_path = embedding_config['config_path']
            args.config = str(Path(__file__).parent / config_path)
    else:
        # Fallback defaults if no config loaded
        if args.clustering_method is None:
            args.clustering_method = 'hierarchical_mrl'
        if args.coarse_threshold is None:
            args.coarse_threshold = 0.6
        if args.refined_threshold is None:
            args.refined_threshold = 0.4
        if args.boundary_threshold is None:
            args.boundary_threshold = 0.7
        if args.embedding_dim is None:
            args.embedding_dim = 256
        if args.segmentation_threshold is None:
            args.segmentation_threshold = 0.5
        if args.segmentation_min_duration_on is None:
            args.segmentation_min_duration_on = 0.0
        if args.segmentation_min_duration_off is None:
            args.segmentation_min_duration_off = 0.0

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print("="*70)
    print("Speaker Diarization with ReDimNet-MRL")
    print("="*70)
    print(f"Audio: {audio_path.name}")
    if Path(args.diar_config).exists():
        print(f"Config: {Path(args.diar_config).name}")
    print(f"Clustering: {args.clustering_method}")
    print(f"Device: {args.device}")
    print()

    # Create embedding model
    print("Loading ReDimNet-MRL embedding model...")
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=args.checkpoint or get_default_checkpoint_path(),
        config_path=args.config or get_default_config_path(),
        embedding_dim=args.embedding_dim,
        extract_all_dims=(args.clustering_method == 'hierarchical_mrl'),
        device=torch.device(args.device)
    )
    print()

    # Create pyannote pipeline
    print("Creating diarization pipeline...")
    try:
        # Get HF token from environment
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print("WARNING: HF_TOKEN not found in environment")
            print("  The .env file should be loaded automatically from diarization/.env")
            print("  Or set it manually: export HF_TOKEN=your_token")

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
                args.segmentation,
                token=hf_token
            )
        except TypeError:
            # Fall back to use_auth_token (older API)
            pipeline = Pipeline.from_pretrained(
                args.segmentation,
                use_auth_token=hf_token
            )
        finally:
            # Restore original torch.load
            torch.load = original_load

        # Apply segmentation parameters
        # Note: pyannote pipelines have different parameters depending on model type
        # - Powerset models (like speaker-diarization-community-1): only min_duration_off
        # - Non-powerset models: threshold and min_duration_off
        print(f"  Segmentation parameters:")

        # Check if model uses powerset (determines available parameters)
        is_powerset = False
        if hasattr(pipeline, '_segmentation') and hasattr(pipeline._segmentation, 'model'):
            is_powerset = pipeline._segmentation.model.specifications.powerset
            print(f"    Model type: {'powerset (multi-label)' if is_powerset else 'binary'}")

        # Apply parameters based on model type
        if hasattr(pipeline, 'segmentation'):
            if is_powerset:
                # Powerset models only have min_duration_off
                if hasattr(pipeline.segmentation, 'min_duration_off'):
                    pipeline.segmentation.min_duration_off = args.segmentation_min_duration_off
                    print(f"    Min duration off: {args.segmentation_min_duration_off}s")
                print(f"    [INFO] Powerset models don't have threshold parameter")
                print(f"    [INFO] VAD is learned by the model, not controlled by threshold")
            else:
                # Non-powerset models have threshold and min_duration_off
                if hasattr(pipeline.segmentation, 'threshold'):
                    pipeline.segmentation.threshold = args.segmentation_threshold
                    print(f"    VAD threshold: {args.segmentation_threshold}")
                if hasattr(pipeline.segmentation, 'min_duration_off'):
                    pipeline.segmentation.min_duration_off = args.segmentation_min_duration_off
                    print(f"    Min duration off: {args.segmentation_min_duration_off}s")
        else:
            print("    WARNING: Could not access segmentation parameters")

        # Override embedding model
        pipeline._embedding = embedding_model

        # Override clustering if using hierarchical MRL
        if args.clustering_method == 'hierarchical_mrl':
            print(f"  Using Hierarchical MRL Clustering")
            print(f"    Coarse threshold: {args.coarse_threshold}")
            print(f"    Refined threshold: {args.refined_threshold}")
            print(f"    Boundary threshold: {args.boundary_threshold}")

            clustering = PyannoteStyleClustering(
                method='hierarchical_mrl',
                embedding_model=embedding_model,
                coarse_threshold=args.coarse_threshold,
                refined_threshold=args.refined_threshold,
                boundary_threshold=args.boundary_threshold,
            )
            pipeline.clustering = clustering
        else:
            print(f"  Using pyannote default clustering")

        print()

    except Exception as e:
        print(f"ERROR: Failed to create pipeline: {e}")
        print("\nNote: You may need to accept pyannote's terms of use and set HF token:")
        print("  1. Visit: https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("  2. Accept the terms")
        print("  3. Set token: export HF_TOKEN=your_token")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Apply diarization
    print(f"Processing audio file...")
    start_time = time.time()

    try:
        diarization = pipeline(str(audio_path))
        inference_time = time.time() - start_time

    except Exception as e:
        print(f"ERROR: Diarization failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    print(f"  Inference time: {inference_time:.2f}s")
    print()

    # Extract annotation object from diarization output
    # Handle different pyannote versions that return different types
    if hasattr(diarization, 'speaker_diarization'):
        # DiarizeOutput object from newer pyannote versions
        annotation = diarization.speaker_diarization
    elif hasattr(diarization, 'annotation'):
        annotation = diarization.annotation
    elif hasattr(diarization, 'labels'):
        # Already an Annotation object
        annotation = diarization
    elif isinstance(diarization, dict) and 'annotation' in diarization:
        annotation = diarization['annotation']
    else:
        print(f"Warning: Unknown diarization output type: {type(diarization)}")
        print(f"Available attributes: {dir(diarization)}")
        annotation = diarization  # Try to use it as-is

    # Save results
    print("Saving results...")
    with open(args.output, "w") as f:
        annotation.write_rttm(f)
    print(f"  Results saved to: {args.output}")

    # Print statistics
    print()
    print("Speaker Statistics:")
    print("-" * 70)
    total_duration = 0
    for speaker in sorted(annotation.labels()):
        duration = sum(
            segment.duration
            for segment in annotation.label_timeline(speaker)
        )
        total_duration += duration
        percentage = (duration / annotation.get_timeline().extent().end) * 100
        print(f"  {speaker:12s}: {duration:6.1f}s ({percentage:5.1f}%)")

    print("-" * 70)
    print(f"  {'Total':12s}: {total_duration:6.1f}s")
    print(f"  {'Speakers':12s}: {len(annotation.labels())}")

    # Visualize if requested
    if args.visualize:
        print()
        print("Generating visualization...")
        visualize_diarization(annotation, audio_path, args.vis_output)

    print()
    print("="*70)
    print("[OK] Diarization completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
