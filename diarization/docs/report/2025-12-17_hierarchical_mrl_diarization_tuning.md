# Hierarchical MRL Diarization Hyperparameter Tuning Report

**Date:** 2025-12-17 16:50:11

## Executive Summary

- **Number of trials:** 100
- **Successful trials:** 100
- **Failed trials:** 0
- **Files evaluated:** 11
- **Optimization duration:** 48.59 hours
- **Pareto-optimal solutions:** 14

## Optimization Configuration

### Search Space

- `coarse_threshold`: [0.50, 0.90]
- `refined_threshold`: [0.30, 0.70]
- `boundary_threshold`: [0.60, 0.80]
- `min_duration_off`: [0.0, 0.5] seconds
- `min_cluster_size`: 2 (fixed)

### Objectives (Multi-objective Optimization)

1. **DER (Diarization Error Rate)**: Minimize
2. **Inference Time**: Minimize
3. **Speaker Count MAE**: Minimize

## Best Solutions (Pareto Front)

### Accuracy-Optimized

- **DER:** 58.34%
- **Inference Time:** 159.07s
- **Speaker Count MAE:** 0.82
- **Hyperparameters:**
  - `coarse_threshold`: 0.3781
  - `refined_threshold`: 0.3456
  - `boundary_threshold`: 0.7675
  - `min_duration_off`: 0.3049
- **Configuration File:** `diar_config_tuned_accuracy_optimized.yaml`

### Speed-Optimized

- **DER:** 59.37%
- **Inference Time:** 157.62s
- **Speaker Count MAE:** 1.91
- **Hyperparameters:**
  - `coarse_threshold`: 0.4096
  - `refined_threshold`: 0.3308
  - `boundary_threshold`: 0.7686
  - `min_duration_off`: 0.2885
- **Configuration File:** `diar_config_tuned_speed_optimized.yaml`

### Speaker Count Optimized

- **DER:** 60.08%
- **Inference Time:** 158.47s
- **Speaker Count MAE:** 0.82
- **Hyperparameters:**
  - `coarse_threshold`: 0.3577
  - `refined_threshold`: 0.3231
  - `boundary_threshold`: 0.7156
  - `min_duration_off`: 0.3522
- **Configuration File:** `diar_config_tuned_speaker_count_optimized.yaml`

### Balanced

- **DER:** 58.58%
- **Inference Time:** 157.92s
- **Speaker Count MAE:** 1.09
- **Hyperparameters:**
  - `coarse_threshold`: 0.3838
  - `refined_threshold`: 0.3427
  - `boundary_threshold`: 0.7577
  - `min_duration_off`: 0.3023
- **Configuration File:** `diar_config_tuned_balanced.yaml`

## Evaluation Files

<FILE LISTS REDACTED>

## Visualizations

Interactive visualizations have been generated in `docs/report/figures/`:

- `pareto_front.html`: 3D Pareto front visualization
- `optimization_history_der.html`: Optimization progress over trials
- `param_importance_*.html`: Parameter importance for each objective
- `parallel_coordinate.html`: Parallel coordinate plot of Pareto solutions

## Recommendations

- **Default recommendation:** Use the balanced solution for general use
- **For maximum accuracy:** Use the accuracy-optimized configuration
- **For speed-critical applications:** Use the speed-optimized configuration
- **For better speaker counting:** Use the speaker count optimized configuration

