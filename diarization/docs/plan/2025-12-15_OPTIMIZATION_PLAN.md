Always use uv to run Python code.

# Task: Create Optuna Hyperparameter Tuning Script

Create `optuna_tuning.py` which finds the best hyperparameters for hierarchical MRL diarization using multi-objective optimization.
When you run the script, run with nohup and & to run in the background.

## Configuration

- **Base configuration**: Use `diar_config.yaml` as the default configuration
- **Input data**: All audio files in `input/` directory with corresponding reference RTTM files

## Hyperparameters to Tune

Hyperparameter search spaces:
- `coarse_threshold`: [0.30, 0.60] (float) - CORRECTED: Lower values to prevent over-clustering
- `refined_threshold`: [0.25, 0.50] (float) - CORRECTED: Lower values to prevent over-clustering
- `boundary_threshold`: [0.60, 0.80] (float)
- `pipeline.segmentation.min_duration_off`: [0.0, 0.5] (float, in seconds)
- `min_cluster_size`: Fixed at 2 (not tuned)

**Note:** Lower similarity thresholds → higher distance thresholds → more merging → fewer clusters.
The original ranges ([0.50-0.90], [0.30-0.70]) were too high and caused severe over-clustering (100+ speakers for 3 ground truth speakers).

## Optimization Strategy

Use Optuna's **multi-objective optimization** to find the Pareto front:

**Three objectives** (all to minimize):
1. **DER (Diarization Error Rate)**: Primary metric for accuracy
2. **Inference time**: Total processing time per audio file (seconds)
3. **Speaker count MAE**: Mean Absolute Error of predicted vs ground truth speaker count

**Approach**:
- Use `optuna.create_study(directions=["minimize", "minimize", "minimize"])`
- Find Pareto-optimal solutions that balance all three objectives
- No single "best" solution, but a set of non-dominated solutions
- Allow user to choose based on their priority (accuracy vs speed vs speaker count)

## Evaluation Strategy

**Per trial evaluation**:
1. Apply hyperparameters to pipeline configuration
2. Process ALL audio files in `input/` directory
3. For each file with reference RTTM:
   - Compute DER
   - Measure inference time
   - Compute speaker count error: |predicted_speakers - ground_truth_speakers|
4. Aggregate metrics:
   - Average DER across all files with reference RTTM
   - Average inference time across all processed files
   - Average speaker count MAE across files with reference RTTM
5. Return tuple of (avg_DER, avg_inference_time, avg_speaker_count_MAE)

**File handling**:
- Auto-detect reference RTTM files using patterns:
  - `audio.flac` → `audio.rttm`
  - `audio.flac` → `audio_annotated.rttm`
- Skip DER and speaker count computation for files without reference RTTM
- Report which files were used for evaluation

## Optuna Configuration

```python
# Study configuration
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(
    directions=["minimize", "minimize", "minimize"],
    sampler=sampler,
    study_name=f"hierarchical_mrl_diarization_{date}",
    storage="sqlite:///optuna_study.db",  # Persistence
    load_if_exists=True
)

# Run optimization
n_trials = 100  # Or use early stopping
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
```

**Parameters**:
- Sampler: TPESampler (Tree-structured Parzen Estimator)
- Number of trials: 100
- Pruning: Optional MedianPruner for early stopping of poor trials
- Storage: SQLite database (`optuna_study.db`) for persistence and resumability
- Random seed: 42 (for reproducibility)

## Error Handling

**Robust trial execution**:
- Wrap each trial in try-except block
- If trial fails (audio processing error, OOM, etc.):
  - Log error with full traceback to `logs/optuna_errors_YYYY-MM-DD.log`
  - Return tuple of (inf, inf, inf) to mark as failed trial
  - Continue with next trial
- Track and report number of failed trials in final summary

## Progress Tracking and Visualization

**During optimization**:
- Log each trial's parameters and objectives
- Save intermediate results every 10 trials to `optuna_checkpoint.pkl`
- Print progress summary every 10 trials

**After optimization**:
Generate visualizations and save to `docs/report/figures/`:
- Pareto front plot (3D scatter for 3 objectives)
- Objective value vs trial number (3 subplots)
- Parameter importance plot (for each objective)
- Hyperparameter relationship plots
- Parallel coordinate plot showing Pareto-optimal trials

## Output Files

### 1. Configuration Files

**Pareto-optimal configurations**:
Since multi-objective optimization produces multiple solutions on the Pareto front, create multiple config files:

- `diar_config_tuned_balanced.yaml`: Best balance of all three objectives
- `diar_config_tuned_accuracy.yaml`: Prioritize DER (lowest DER on Pareto front)
- `diar_config_tuned_speed.yaml`: Prioritize inference time (fastest on Pareto front)
- `diar_config_tuned_speaker_count.yaml`: Prioritize speaker count accuracy

Each file should:
- Include all parameters from base `diar_config.yaml`
- Update only the tuned hyperparameters
- Add header comment documenting:
  - Generation date
  - Optimization metrics (DER, time, speaker count MAE)
  - Number of trials run
  - Which Pareto solution this represents

### 2. Report

Create detailed report: `docs/report/YYYY-MM-DD_hierarchical_mrl_diarization_tuning.md`

**Report structure**:

```markdown
# Hierarchical MRL Diarization Hyperparameter Tuning Report
Date: YYYY-MM-DD

## Executive Summary
- Number of trials: X
- Number of successful trials: X
- Number of files evaluated: X
- Optimization duration: X hours

## Baseline Performance
[Table comparing current diar_config.yaml performance]

## Pareto Front Analysis
- Number of Pareto-optimal solutions: X
- Trade-offs between objectives
- [3D Pareto front visualization]

## Best Solutions

### Solution 1: Accuracy-Optimized
- DER: X.XX%
- Inference time: X.Xs
- Speaker count MAE: X.XX
- Hyperparameters: {...}

### Solution 2: Speed-Optimized
- DER: X.XX%
- Inference time: X.Xs
- Speaker count MAE: X.XX
- Hyperparameters: {...}

### Solution 3: Balanced
- DER: X.XX%
- Inference time: X.Xs
- Speaker count MAE: X.XX
- Hyperparameters: {...}

### Solution 4: Speaker Count Optimized
- DER: X.XX%
- Inference time: X.Xs
- Speaker count MAE: X.XX
- Hyperparameters: {...}

## Per-File Performance Breakdown
[Table showing performance on each audio file]

## Parameter Importance Analysis
[For each objective, show which parameters matter most]

## Optimization History
[Plots showing convergence]

## Recommendations
- Default recommendation: balanced solution
- For few speakers (2-4): [specific solution]
- For many speakers (5+): [specific solution]
- For speed-critical applications: speed-optimized solution
- For maximum accuracy: accuracy-optimized solution

## Validation Results
[Performance comparison: baseline vs tuned on all files]

## Failed Trials
[Summary of any failed trials and reasons]
```

## Validation

**After hyperparameter tuning**:
1. Load each Pareto-optimal configuration
2. Run full evaluation on all input files
3. Compare with baseline (current `diar_config.yaml`)
4. Generate before/after comparison tables
5. Verify improvements are consistent across files
6. Include statistical significance tests if sufficient data

## Implementation Notes

- Use `compare_clustering_methods.py` or `test_diarization.py` as reference for diarization pipeline
- Leverage existing `apply_segmentation_params()` function for parameter application
- Ensure consistent random seeds for reproducibility
- Support resuming optimization from saved study
- Add command-line arguments:
  - `--input-dir`: Directory with audio files (default: `diarization/input/`)
  - `--n-trials`: Number of optimization trials (default: 100)
  - `--study-name`: Custom study name
  - `--resume`: Resume from existing study