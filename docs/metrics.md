# Metrics

This document explains the main evaluation metrics used in LearnEqConstraints.

The public-facing names below match the paper and analysis scripts:
- `distance_error`
- `coverage_error`

Some result files and internal code may also contain legacy field names kept for backward compatibility. Those mappings are listed explicitly below.

## Naming Map

Public metric name to legacy/internal key:

- `distance_error` -> `proj_manifold_dist`
- `coverage_error` -> `gt_to_learned_mean`
- `learned_distance_error` -> `learned_to_gt_mean`
- `bidirectional_chamfer` -> `bidirectional_chamfer`

Interpretation:
- `coverage_error` measures how well the learned manifold covers the ground-truth manifold.
- `distance_error` measures how far projected learned samples are from the ground-truth manifold.

## Core Metrics

### `distance_error`

Legacy key:
- `proj_manifold_dist`

Definition:
- Mean nearest-neighbor distance from projected samples on the learned manifold to the ground-truth manifold.

How it is computed:
1. Sample evaluation seeds.
2. Project them onto the learned manifold.
3. Measure each projected point's nearest-neighbor distance to the ground-truth manifold.
4. Average those distances.

Interpretation:
- Lower is better.
- This is the main "how far are learned predictions from the true manifold" metric.

### `coverage_error`

Legacy key:
- `gt_to_learned_mean`

Definition:
- Mean nearest-neighbor distance from ground-truth manifold samples to learned manifold samples.

How it is computed:
1. Sample or construct ground-truth manifold points.
2. Sample learned manifold points through projection.
3. For each ground-truth point, compute the nearest learned point distance.
4. Average those distances.

Interpretation:
- Lower is better.
- This is the main "how well does the learned manifold cover the full target manifold" metric.

### `learned_distance_error`

Legacy key:
- `learned_to_gt_mean`

Definition:
- Mean nearest-neighbor distance from learned manifold samples to ground-truth manifold samples.

Interpretation:
- Lower is better.
- Similar to `distance_error`, but computed from the learned-sample side rather than the projected-evaluation side.

### `bidirectional_chamfer`

Definition:
- `mean(gt -> learned) + mean(learned -> gt)`

Interpretation:
- Lower is better.
- Combines coverage quality and learned-sample accuracy into one symmetric metric.

## Projection Metrics

### `proj_true_dist`

Definition:
- Mean distance between projected points and the true projection targets of the same evaluation seeds.

Interpretation:
- Lower is better.
- Useful when an analytic or high-quality reference projection is available.

### `proj_v_residual`

Definition:
- Mean residual constraint value after projection.

Interpretation:
- Lower is better.
- Measures how well the projection routine actually lands on the learned zero set.

### `proj_steps`

Definition:
- Mean number of projection iterations.

Interpretation:
- Lower is faster.
- Should be read together with `distance_error` and `proj_v_residual`, not in isolation.

## Prediction Metrics

### `pred_recall`

Definition:
- Fraction of ground-truth manifold points classified as on-manifold by the learned model.

Interpretation:
- Higher is better.

### `pred_precision`

Definition:
- Fraction of projected learned points that are close enough to the ground-truth manifold.

Interpretation:
- Higher is better.

### `pred_FPrate`

Definition:
- Fraction of clearly off-manifold evaluation points that are incorrectly classified as on-manifold.

Interpretation:
- Lower is better.

### `pred_on_mean_v`

Definition:
- Mean constraint residual energy on training manifold points.

Interpretation:
- Lower is better.

## Percentile Metrics

### `gt_to_learned_p95`

Definition:
- 95th percentile of GT-to-learned nearest-neighbor distance.

Interpretation:
- Lower is better.
- Highlights worst-case coverage failures better than the mean.

### `learned_to_gt_p95`

Definition:
- 95th percentile of learned-to-GT nearest-neighbor distance.

Interpretation:
- Lower is better.

## Bookkeeping Metrics

### `eval_eps_used`

Definition:
- Threshold used to determine whether a point is treated as on-manifold during evaluation.

### `eps_stop`

Definition:
- Projection stopping threshold reported by the evaluation pipeline.

### `dist_space`

Definition:
- Space where distances are computed.

Typical values:
- `data_space`
- `workspace`

### `n_gt`

Definition:
- Number of ground-truth samples used in the metric computation.

### `n_learned`

Definition:
- Number of learned or projected samples used in the metric computation.

## Practical Reading Guide

For the paper-level comparison, the most important metrics are:
- `distance_error`
- `coverage_error`
- `pred_precision`
- `train_seconds`

Recommended interpretation:
- Use `distance_error` to judge local geometric accuracy.
- Use `coverage_error` to judge global manifold coverage.
- Use `pred_precision` to judge whether predicted on-manifold points are actually correct.
- Use `train_seconds` for efficiency tradeoffs.

## Source of Truth

Implementation details are in:
- `/home/baiyu/PycharmProjects/LearnEqConstraints/evaluation/evaluator.py`
- `/home/baiyu/PycharmProjects/LearnEqConstraints/evaluation/README_metrics.md`
