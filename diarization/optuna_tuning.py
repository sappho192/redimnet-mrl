#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for Hierarchical MRL Diarization

Performs multi-objective optimization to find Pareto-optimal hyperparameters
that balance:
1. DER (Diarization Error Rate) - accuracy metric
2. Inference time - speed metric
3. Speaker count MAE - speaker estimation accuracy

Usage:
    # Run with default settings (100 trials)
    nohup uv run python diarization/optuna_tuning.py &

    # Custom number of trials
    uv run python diarization/optuna_tuning.py --n-trials 50

    # Resume existing study
    uv run python diarization/optuna_tuning.py --resume
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import time
import traceback
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

# Setup path
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
                    from huggingface_hub import login
                    try:
                        login(token=value)
                        print("Authenticated with Hugging Face")
                    except Exception as e:
                        print(f"Warning: HF login failed: {e}")

import numpy as np
import yaml
import torch
import torch.serialization
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import (
    plot_pareto_front,
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

from checkpoint_utils import get_default_checkpoint_path, get_default_config_path
from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding
from hierarchical_mrl_clustering import PyannoteStyleClustering

try:
    from pyannote.audio import Pipeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.database.util import load_rttm
    from pyannote.core import Annotation, Segment, Timeline
except ImportError:
    print("ERROR: pyannote.audio not available")
    sys.exit(1)

# Configuration
CONFIG_DIR = Path(__file__).parent
INPUT_DIR = CONFIG_DIR / "input"
LOGS_DIR = CONFIG_DIR / "logs"
REPORT_DIR = CONFIG_DIR / "docs" / "report"
FIGURES_DIR = REPORT_DIR / "figures"
CHECKPOINT_FILE = CONFIG_DIR / "optuna_checkpoint.pkl"

# Create directories
LOGS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Global variables for shared resources
EMBEDDING_MODEL = None
PIPELINE_TEMPLATE = None
AUDIO_FILES_WITH_RTTM = []
TRIAL_COUNTER = 0
STUDY_START_TIME = None


def load_rttm_manual(rttm_path: Path, uri: str = 'audio') -> Annotation:
    """Manually parse RTTM file to create Annotation object."""
    annotation = Annotation(uri=uri)

    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 8 or parts[0] != 'SPEAKER':
                continue

            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]

            segment = Segment(start, start + duration)
            annotation[segment] = speaker

    return annotation


def get_ground_truth_speaker_count(rttm_path: Path) -> int:
    """Get the number of unique speakers from RTTM file."""
    try:
        reference_dict = load_rttm(str(rttm_path))
        if not reference_dict:
            reference = load_rttm_manual(rttm_path, uri='audio')
        else:
            uri = list(reference_dict.keys())[0]
            reference = reference_dict[uri]
        return len(reference.labels())
    except Exception as e:
        print(f"Warning: Could not get speaker count from {rttm_path}: {e}")
        return 0


def compute_der(reference_rttm_path: Path, hypothesis: Annotation) -> Optional[float]:
    """Compute DER between reference and hypothesis."""
    try:
        reference_dict = load_rttm(str(reference_rttm_path))

        if not reference_dict:
            reference = load_rttm_manual(reference_rttm_path, uri='audio')
        else:
            uri = list(reference_dict.keys())[0]
            reference = reference_dict[uri]

        metric = DiarizationErrorRate()
        extent = hypothesis.get_timeline().extent()
        uem = Timeline([extent])

        der = metric(reference, hypothesis, uem=uem, detailed=False)

        # Handle different return types
        if isinstance(der, dict):
            return der.get('diarization error rate', 0) * 100
        else:
            return der * 100
    except Exception as e:
        print(f"Warning: Could not compute DER: {e}")
        return None


def find_audio_files_with_rttm() -> List[Tuple[Path, Path]]:
    """Find all audio files with corresponding RTTM files."""
    audio_files = []

    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
    for ext in audio_extensions:
        audio_files.extend(INPUT_DIR.glob(f'*{ext}'))
        audio_files.extend(INPUT_DIR.glob(f'*{ext.upper()}'))

    files_with_rttm = []
    for audio_path in sorted(audio_files):
        # Look for matching RTTM file
        rttm_path = audio_path.with_suffix('.rttm')
        if not rttm_path.exists():
            rttm_path = audio_path.parent / f"{audio_path.stem}_annotated.rttm"

        if rttm_path.exists():
            files_with_rttm.append((audio_path, rttm_path))

    return files_with_rttm


def initialize_models(device: torch.device):
    """Initialize shared embedding model and pipeline template."""
    global EMBEDDING_MODEL, PIPELINE_TEMPLATE

    print("Initializing embedding model...")
    EMBEDDING_MODEL = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=get_default_checkpoint_path(),
        config_path=get_default_config_path(),
        embedding_dim=256,
        extract_all_dims=True,
        device=device
    )

    print("Initializing pipeline template...")
    hf_token = os.environ.get('HF_TOKEN')

    # Patch torch.load for pyannote models
    import functools
    original_load = torch.load

    @functools.wraps(original_load)
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load

    try:
        PIPELINE_TEMPLATE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token
        )
    except TypeError:
        PIPELINE_TEMPLATE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=hf_token
        )
    finally:
        torch.load = original_load

    # Override embedding
    PIPELINE_TEMPLATE._embedding = EMBEDDING_MODEL

    print(f"Initialized models successfully")


def apply_hyperparameters(
    pipeline,
    coarse_threshold: float,
    refined_threshold: float,
    boundary_threshold: float,
    min_duration_off: float
):
    """Apply hyperparameters to pipeline."""
    # Apply segmentation parameters
    if hasattr(pipeline, 'segmentation'):
        if hasattr(pipeline.segmentation, 'min_duration_off'):
            pipeline.segmentation.min_duration_off = min_duration_off

    # Apply clustering parameters
    # IMPORTANT: Use pipeline._embedding to maintain proper reference chain
    clustering = PyannoteStyleClustering(
        method='hierarchical_mrl',
        embedding_model=pipeline._embedding,
        coarse_threshold=coarse_threshold,
        refined_threshold=refined_threshold,
        boundary_threshold=boundary_threshold,
        min_cluster_size=2,  # Fixed
    )
    pipeline.clustering = clustering


def evaluate_single_file(
    audio_path: Path,
    rttm_path: Path,
    coarse_threshold: float,
    refined_threshold: float,
    boundary_threshold: float,
    min_duration_off: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Evaluate diarization on a single file.

    Returns:
        (DER, inference_time, speaker_count_error)
    """
    try:
        # Reset accumulated embeddings from previous run
        EMBEDDING_MODEL.reset_accumulated_embeddings()

        # Use the template pipeline directly (no deep copy to maintain references)
        pipeline = PIPELINE_TEMPLATE

        # Apply hyperparameters
        apply_hyperparameters(
            pipeline,
            coarse_threshold,
            refined_threshold,
            boundary_threshold,
            min_duration_off
        )

        # Run diarization
        start_time = time.time()
        diarization = pipeline(str(audio_path))
        inference_time = time.time() - start_time

        # Extract annotation
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        elif hasattr(diarization, 'labels'):
            annotation = diarization
        else:
            return None, None, None

        # Compute DER
        der = compute_der(rttm_path, annotation)

        # Compute speaker count error
        predicted_speakers = len(annotation.labels())
        ground_truth_speakers = get_ground_truth_speaker_count(rttm_path)
        speaker_count_error = abs(predicted_speakers - ground_truth_speakers)

        return der, inference_time, speaker_count_error

    except Exception as e:
        print(f"Error processing {audio_path.name}: {e}")
        traceback.print_exc()
        return None, None, None


def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
    """
    Objective function for Optuna optimization.

    Returns:
        (avg_DER, avg_inference_time, avg_speaker_count_MAE)
    """
    global TRIAL_COUNTER
    TRIAL_COUNTER += 1

    # Sample hyperparameters
    # NOTE: Lower thresholds allow more merging → fewer clusters → better results
    # Previous ranges were too high, causing severe over-clustering
    coarse_threshold = trial.suggest_float('coarse_threshold', 0.30, 0.60)
    refined_threshold = trial.suggest_float('refined_threshold', 0.25, 0.50)
    boundary_threshold = trial.suggest_float('boundary_threshold', 0.60, 0.80)
    min_duration_off = trial.suggest_float('min_duration_off', 0.0, 0.5)

    print(f"\n{'='*80}")
    print(f"Trial {trial.number} (#{TRIAL_COUNTER})")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  coarse_threshold:    {coarse_threshold:.4f}")
    print(f"  refined_threshold:   {refined_threshold:.4f}")
    print(f"  boundary_threshold:  {boundary_threshold:.4f}")
    print(f"  min_duration_off:    {min_duration_off:.4f}")
    print()

    try:
        # Evaluate on all files
        ders = []
        times = []
        speaker_errors = []

        for i, (audio_path, rttm_path) in enumerate(AUDIO_FILES_WITH_RTTM, 1):
            print(f"[{i}/{len(AUDIO_FILES_WITH_RTTM)}] Processing {audio_path.name}...")

            der, inference_time, speaker_error = evaluate_single_file(
                audio_path,
                rttm_path,
                coarse_threshold,
                refined_threshold,
                boundary_threshold,
                min_duration_off
            )

            if der is not None and inference_time is not None and speaker_error is not None:
                ders.append(der)
                times.append(inference_time)
                speaker_errors.append(speaker_error)
                print(f"  DER: {der:.2f}%, Time: {inference_time:.2f}s, Speaker error: {speaker_error}")
            else:
                print(f"  Skipped (error)")

        if not ders:
            print(f"\nTrial {trial.number} FAILED - No valid results")
            return float('inf'), float('inf'), float('inf')

        # Compute averages
        avg_der = np.mean(ders)
        avg_time = np.mean(times)
        avg_speaker_mae = np.mean(speaker_errors)

        print(f"\nTrial {trial.number} Results:")
        print(f"  Average DER:         {avg_der:.2f}%")
        print(f"  Average Time:        {avg_time:.2f}s")
        print(f"  Average Speaker MAE: {avg_speaker_mae:.2f}")
        print(f"{'='*80}\n")

        return avg_der, avg_time, avg_speaker_mae

    except Exception as e:
        error_log = LOGS_DIR / f"optuna_errors_{datetime.now().strftime('%Y-%m-%d')}.log"
        with open(error_log, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Trial {trial.number} failed at {datetime.now()}\n")
            f.write(f"Parameters: {trial.params}\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
            f.write(f"{'='*80}\n")

        print(f"\nTrial {trial.number} FAILED - Error logged to {error_log}")
        return float('inf'), float('inf'), float('inf')


def save_checkpoint(study: optuna.Study):
    """Save study checkpoint."""
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump({
                'study_name': study.study_name,
                'trials': len(study.trials),
                'best_trials': study.best_trials,
                'timestamp': datetime.now(),
            }, f)
        print(f"Checkpoint saved: {CHECKPOINT_FILE}")
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def print_progress_summary(study: optuna.Study):
    """Print progress summary every N trials."""
    n_trials = len(study.trials)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    elapsed = time.time() - STUDY_START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\n{'='*80}")
    print(f"Progress Summary (Trial {n_trials})")
    print(f"{'='*80}")
    print(f"  Complete: {n_complete}")
    print(f"  Failed:   {n_failed}")
    print(f"  Elapsed:  {hours}h {minutes}m")
    print(f"  Pareto-optimal trials: {len(study.best_trials)}")

    if study.best_trials:
        print(f"\nCurrent Pareto Front (Best Solutions):")
        for i, trial in enumerate(study.best_trials[:5], 1):  # Show top 5
            print(f"  {i}. DER={trial.values[0]:.2f}%, "
                  f"Time={trial.values[1]:.2f}s, "
                  f"Speaker MAE={trial.values[2]:.2f}")
    print(f"{'='*80}\n")


def generate_visualizations(study: optuna.Study):
    """Generate all optimization visualizations."""
    print("\nGenerating visualizations...")

    try:
        # 1. Pareto front (3D)
        if len(study.best_trials) > 0:
            fig = plot_pareto_front(study, target_names=["DER (%)", "Time (s)", "Speaker MAE"])
            fig.write_html(str(FIGURES_DIR / "pareto_front.html"))
            print(f"  - Pareto front: {FIGURES_DIR / 'pareto_front.html'}")
    except Exception as e:
        print(f"  Warning: Could not create Pareto front plot: {e}")

    try:
        # 2. Optimization history
        fig = plot_optimization_history(study, target_name="DER (%)")
        fig.write_html(str(FIGURES_DIR / "optimization_history_der.html"))
        print(f"  - Optimization history (DER): {FIGURES_DIR / 'optimization_history_der.html'}")
    except Exception as e:
        print(f"  Warning: Could not create optimization history: {e}")

    try:
        # 3. Parameter importances
        for i, objective_name in enumerate(["DER", "Time", "Speaker_MAE"]):
            fig = plot_param_importances(study, target=lambda t: t.values[i], target_name=objective_name)
            fig.write_html(str(FIGURES_DIR / f"param_importance_{objective_name.lower()}.html"))
        print(f"  - Parameter importances: {FIGURES_DIR / 'param_importance_*.html'}")
    except Exception as e:
        print(f"  Warning: Could not create parameter importance plots: {e}")

    try:
        # 4. Parallel coordinate plot
        fig = plot_parallel_coordinate(study, target_names=["DER (%)", "Time (s)", "Speaker MAE"])
        fig.write_html(str(FIGURES_DIR / "parallel_coordinate.html"))
        print(f"  - Parallel coordinate: {FIGURES_DIR / 'parallel_coordinate.html'}")
    except Exception as e:
        print(f"  Warning: Could not create parallel coordinate plot: {e}")

    print("Visualizations complete!")


def create_pareto_configs(study: optuna.Study, baseline_config_path: Path):
    """Create configuration files for Pareto-optimal solutions."""
    print("\nCreating Pareto-optimal configuration files...")

    # Load baseline config
    with open(baseline_config_path, 'r') as f:
        baseline_config = yaml.safe_load(f)

    if not study.best_trials:
        print("No Pareto-optimal trials found!")
        return

    # Find specific solutions
    best_trials = study.best_trials

    # 1. Accuracy-optimized (lowest DER)
    accuracy_trial = min(best_trials, key=lambda t: t.values[0])

    # 2. Speed-optimized (lowest time)
    speed_trial = min(best_trials, key=lambda t: t.values[1])

    # 3. Speaker count optimized (lowest MAE)
    speaker_trial = min(best_trials, key=lambda t: t.values[2])

    # 4. Balanced (minimize normalized sum)
    der_values = [t.values[0] for t in best_trials]
    time_values = [t.values[1] for t in best_trials]
    mae_values = [t.values[2] for t in best_trials]

    der_norm = [(v - min(der_values)) / (max(der_values) - min(der_values) + 1e-10) for v in der_values]
    time_norm = [(v - min(time_values)) / (max(time_values) - min(time_values) + 1e-10) for v in time_values]
    mae_norm = [(v - min(mae_values)) / (max(mae_values) - min(mae_values) + 1e-10) for v in mae_values]

    balanced_scores = [d + t + m for d, t, m in zip(der_norm, time_norm, mae_norm)]
    balanced_trial = best_trials[balanced_scores.index(min(balanced_scores))]

    configs = [
        ('accuracy', accuracy_trial, 'Accuracy-Optimized (Lowest DER)'),
        ('speed', speed_trial, 'Speed-Optimized (Fastest)'),
        ('speaker_count', speaker_trial, 'Speaker Count Optimized (Best speaker estimation)'),
        ('balanced', balanced_trial, 'Balanced (Best overall trade-off)'),
    ]

    for name, trial, description in configs:
        config = baseline_config.copy()

        # Update clustering parameters
        config['clustering']['coarse_threshold'] = trial.params['coarse_threshold']
        config['clustering']['refined_threshold'] = trial.params['refined_threshold']
        config['clustering']['boundary_threshold'] = trial.params['boundary_threshold']

        # Update segmentation parameters
        config['pipeline']['segmentation']['min_duration_off'] = trial.params['min_duration_off']

        # Add header comment
        header = f"""# ReDimNet-MRL Speaker Diarization Configuration - {description}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Hyperparameter Optimization Results:
#   Trial: {trial.number}
#   DER: {trial.values[0]:.2f}%
#   Inference Time: {trial.values[1]:.2f}s
#   Speaker Count MAE: {trial.values[2]:.2f}
#   Pareto Solution: {description}
#
# Tuned Parameters:
#   coarse_threshold: {trial.params['coarse_threshold']:.4f}
#   refined_threshold: {trial.params['refined_threshold']:.4f}
#   boundary_threshold: {trial.params['boundary_threshold']:.4f}
#   min_duration_off: {trial.params['min_duration_off']:.4f}

"""

        # Save config
        output_path = CONFIG_DIR / f"diar_config_tuned_{name}.yaml"
        with open(output_path, 'w') as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  - {output_path.name}: DER={trial.values[0]:.2f}%, Time={trial.values[1]:.2f}s, MAE={trial.values[2]:.2f}")


def generate_report(study: optuna.Study, baseline_config_path: Path):
    """Generate detailed optimization report."""
    print("\nGenerating optimization report...")

    report_file = REPORT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_hierarchical_mrl_diarization_tuning.md"

    n_trials = len(study.trials)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    elapsed = time.time() - STUDY_START_TIME
    hours = elapsed / 3600

    with open(report_file, 'w') as f:
        f.write(f"# Hierarchical MRL Diarization Hyperparameter Tuning Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Executive Summary\n\n")
        f.write(f"- **Number of trials:** {n_trials}\n")
        f.write(f"- **Successful trials:** {n_complete}\n")
        f.write(f"- **Failed trials:** {n_failed}\n")
        f.write(f"- **Files evaluated:** {len(AUDIO_FILES_WITH_RTTM)}\n")
        f.write(f"- **Optimization duration:** {hours:.2f} hours\n")
        f.write(f"- **Pareto-optimal solutions:** {len(study.best_trials)}\n\n")

        f.write(f"## Optimization Configuration\n\n")
        f.write(f"### Search Space\n\n")
        f.write(f"- `coarse_threshold`: [0.50, 0.90]\n")
        f.write(f"- `refined_threshold`: [0.30, 0.70]\n")
        f.write(f"- `boundary_threshold`: [0.60, 0.80]\n")
        f.write(f"- `min_duration_off`: [0.0, 0.5] seconds\n")
        f.write(f"- `min_cluster_size`: 2 (fixed)\n\n")

        f.write(f"### Objectives (Multi-objective Optimization)\n\n")
        f.write(f"1. **DER (Diarization Error Rate)**: Minimize\n")
        f.write(f"2. **Inference Time**: Minimize\n")
        f.write(f"3. **Speaker Count MAE**: Minimize\n\n")

        if study.best_trials:
            f.write(f"## Best Solutions (Pareto Front)\n\n")

            # Find specific solutions
            best_trials = study.best_trials

            accuracy_trial = min(best_trials, key=lambda t: t.values[0])
            speed_trial = min(best_trials, key=lambda t: t.values[1])
            speaker_trial = min(best_trials, key=lambda t: t.values[2])

            # Balanced
            der_values = [t.values[0] for t in best_trials]
            time_values = [t.values[1] for t in best_trials]
            mae_values = [t.values[2] for t in best_trials]

            der_norm = [(v - min(der_values)) / (max(der_values) - min(der_values) + 1e-10) for v in der_values]
            time_norm = [(v - min(time_values)) / (max(time_values) - min(time_values) + 1e-10) for v in time_values]
            mae_norm = [(v - min(mae_values)) / (max(mae_values) - min(mae_values) + 1e-10) for v in mae_values]

            balanced_scores = [d + t + m for d, t, m in zip(der_norm, time_norm, mae_norm)]
            balanced_trial = best_trials[balanced_scores.index(min(balanced_scores))]

            solutions = [
                ("Accuracy-Optimized", accuracy_trial),
                ("Speed-Optimized", speed_trial),
                ("Speaker Count Optimized", speaker_trial),
                ("Balanced", balanced_trial),
            ]

            for name, trial in solutions:
                f.write(f"### {name}\n\n")
                f.write(f"- **DER:** {trial.values[0]:.2f}%\n")
                f.write(f"- **Inference Time:** {trial.values[1]:.2f}s\n")
                f.write(f"- **Speaker Count MAE:** {trial.values[2]:.2f}\n")
                f.write(f"- **Hyperparameters:**\n")
                f.write(f"  - `coarse_threshold`: {trial.params['coarse_threshold']:.4f}\n")
                f.write(f"  - `refined_threshold`: {trial.params['refined_threshold']:.4f}\n")
                f.write(f"  - `boundary_threshold`: {trial.params['boundary_threshold']:.4f}\n")
                f.write(f"  - `min_duration_off`: {trial.params['min_duration_off']:.4f}\n")
                f.write(f"- **Configuration File:** `diar_config_tuned_{name.lower().replace(' ', '_').replace('-', '_')}.yaml`\n\n")

        f.write(f"## Evaluation Files\n\n")
        for i, (audio_path, rttm_path) in enumerate(AUDIO_FILES_WITH_RTTM, 1):
            f.write(f"{i}. {audio_path.name}\n")

        f.write(f"\n## Visualizations\n\n")
        f.write(f"Interactive visualizations have been generated in `{FIGURES_DIR.relative_to(CONFIG_DIR)}/`:\n\n")
        f.write(f"- `pareto_front.html`: 3D Pareto front visualization\n")
        f.write(f"- `optimization_history_der.html`: Optimization progress over trials\n")
        f.write(f"- `param_importance_*.html`: Parameter importance for each objective\n")
        f.write(f"- `parallel_coordinate.html`: Parallel coordinate plot of Pareto solutions\n\n")

        f.write(f"## Recommendations\n\n")
        f.write(f"- **Default recommendation:** Use the balanced solution for general use\n")
        f.write(f"- **For maximum accuracy:** Use the accuracy-optimized configuration\n")
        f.write(f"- **For speed-critical applications:** Use the speed-optimized configuration\n")
        f.write(f"- **For better speaker counting:** Use the speaker count optimized configuration\n\n")

        if n_failed > 0:
            f.write(f"## Failed Trials\n\n")
            f.write(f"{n_failed} trials failed during optimization. ")
            f.write(f"Check `{LOGS_DIR / 'optuna_errors_*.log'}` for details.\n\n")

    print(f"Report saved: {report_file}")


def main():
    """Main optimization loop."""
    global STUDY_START_TIME, AUDIO_FILES_WITH_RTTM

    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for hierarchical MRL diarization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR,
                       help=f'Directory with audio files (default: {INPUT_DIR})')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials (default: 100)')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Custom study name (default: hierarchical_mrl_diarization_YYYY-MM-DD)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing study')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference (default: cuda if available)')

    args = parser.parse_args()

    print("="*80)
    print("Hierarchical MRL Diarization - Hyperparameter Tuning")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Input directory: {args.input_dir}")
    print(f"Number of trials: {args.n_trials}")
    print()

    # Find audio files with RTTM
    AUDIO_FILES_WITH_RTTM = find_audio_files_with_rttm()

    if not AUDIO_FILES_WITH_RTTM:
        print(f"ERROR: No audio files with RTTM found in {args.input_dir}")
        return 1

    print(f"Found {len(AUDIO_FILES_WITH_RTTM)} audio file(s) with reference RTTM:")
    for audio_path, rttm_path in AUDIO_FILES_WITH_RTTM:
        print(f"  - {audio_path.name} → {rttm_path.name}")
    print()

    # Initialize models
    device = torch.device(args.device)
    initialize_models(device)

    # Create study
    study_name = args.study_name or f"hierarchical_mrl_diarization_{datetime.now().strftime('%Y-%m-%d')}"
    storage_path = f"sqlite:///{CONFIG_DIR / 'optuna_study.db'}"

    print(f"Creating Optuna study: {study_name}")
    print(f"Storage: {storage_path}")
    print()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        sampler=sampler,
        directions=["minimize", "minimize", "minimize"],  # DER, Time, Speaker MAE
        load_if_exists=args.resume
    )

    if args.resume and len(study.trials) > 0:
        print(f"Resuming study with {len(study.trials)} existing trials")

    # Run optimization
    STUDY_START_TIME = time.time()

    print("="*80)
    print("Starting optimization...")
    print("="*80)
    print()

    try:
        for i in range(args.n_trials):
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            # Save checkpoint every 10 trials
            if (i + 1) % 10 == 0:
                save_checkpoint(study)
                print_progress_summary(study)

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")

    print("\n" + "="*80)
    print("Optimization completed!")
    print("="*80)
    print()

    # Generate outputs
    baseline_config = CONFIG_DIR / "diar_config.yaml"

    generate_visualizations(study)
    create_pareto_configs(study, baseline_config)
    generate_report(study, baseline_config)

    print("\n" + "="*80)
    print("All outputs generated successfully!")
    print("="*80)
    print(f"\nConfiguration files: {CONFIG_DIR / 'diar_config_tuned_*.yaml'}")
    print(f"Report: {REPORT_DIR}")
    print(f"Visualizations: {FIGURES_DIR}")
    print(f"Study database: {CONFIG_DIR / 'optuna_study.db'}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
