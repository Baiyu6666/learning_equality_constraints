# Evaluator Metrics

This document explains the unified evaluation pipeline and every metric produced by:
- `evaluator/evaluator.py`
- `core/eval_runner.py`

## Pipeline

1. Resolve evaluator config via `resolve_eval_cfg(...)`:
- base: `DEFAULT_EVAL_CFG`
- plus method override (`EVAL_METHOD_OVERRIDES`)
- plus dataset override (`EVAL_DATASET_OVERRIDES`)

2. Build GT manifold reference points:
- `resolve_gt_grid(dataset_name, cfg, x_train)`
- usually from `generate_dataset(...)` dense `grid`

3. Compute zero-threshold (`eval_eps_used` / `eps_stop`):
- `compute_eps_stop(model, x_train, cfg)`
- uses `|F(x_train)|` (or `||F(x_train)||` for multi-output)
- threshold is `percentile(abs(F_on), zero_eps_quantile)`

4. Projection metrics (`evaluate_projection_metrics`):
- sample eval seeds `x_eval` in bounds from training set
- evaluate prediction on `x_eval`
- run projection `project_fn(model, x_eval, eval_eps_used) -> (proj, steps)`
- compare projected points vs GT manifold

5. Bidirectional Chamfer (`evaluate_bidirectional_chamfer`):
- project random seeds to learned manifold samples
- compute nearest-neighbor distances both directions vs GT samples

6. Optional distance space:
- `dist_space="data_space"` by default
- if `embed_fn` provided (e.g. robot workspace), metrics are computed in embedded workspace

## Metrics

### Prediction / classification style

- `pred_on_mean_v`
  - Meaning: mean `0.5 * |F(x_train)|^2` on training manifold points
  - Better: lower

- `pred_recall`
  - Meaning: among GT manifold points (dataset `grid`), fraction predicted as on-manifold (`|F(x)| < eval_eps_used`)
  - Better: higher

- `pred_FPrate`
  - Meaning: among sampled eval points that are far from GT manifold (`d_true_eval >= tau`), fraction wrongly predicted as on-manifold
  - Better: lower

- `pred_precision`
  - Meaning: among projected points (`project_fn` outputs), fraction within GT-near threshold (`distance_to_GT < tau`)
  - Better: higher

### Projection quality

- `proj_manifold_dist`
  - Meaning: mean distance from projected points `proj` to GT manifold (nearest-neighbor)
  - Better: lower

- `proj_true_dist`
  - Meaning: mean distance between projected points `proj` and the true projection of same seeds `x_eval`
  - Better: lower

- `proj_v_residual`
  - Meaning: mean `|F(proj)|` after projection
  - Better: lower

- `proj_steps`
  - Meaning: mean projection iteration count returned by `project_fn`
  - Better: lower (for speed), but should be interpreted with accuracy metrics together

### Chamfer / coverage

- `bidirectional_chamfer`
  - Meaning: `mean(d(gt -> learned)) + mean(d(learned -> gt))`
  - Better: lower

- `gt_to_learned_mean`
  - Meaning: average nearest-neighbor distance GT samples -> learned samples
  - Better: lower

- `learned_to_gt_mean`
  - Meaning: average nearest-neighbor distance learned samples -> GT samples
  - Better: lower

- `gt_to_learned_p95`
  - Meaning: 95th percentile of GT->learned NN distance
  - Better: lower

- `learned_to_gt_p95`
  - Meaning: 95th percentile of learned->GT NN distance
  - Better: lower

### Context / bookkeeping

- `eval_eps_used`
  - Meaning: projection/prediction threshold computed from training residual quantile

- `eps_stop`
  - Meaning: same threshold value reported by chamfer routine

- `dist_space`
  - Meaning: `"data_space"` or `"workspace"` depending on whether `embed_fn` is used

- `n_gt`
  - Meaning: number of GT samples used in chamfer

- `n_learned`
  - Meaning: number of learned/projected samples used in chamfer

## Key evaluator hyperparameters (from `DEFAULT_EVAL_CFG`)

- `zero_eps_quantile`: quantile for `eval_eps_used`
- `eval_tau_ratio`: controls near/far threshold `tau` for recall/FP/precision
- `eval_gt_n_grid`: GT manifold grid size used by evaluator (for projection/chamfer reference)
- `eval_proj_n_points`: number of projected evaluation seed points
- `eval_chamfer_near_ratio`, `eval_chamfer_near_noise_std_ratio`: near-manifold seed sampling mix
- `eval_pad_ratio`, `eval_min_axis_span_ratio`: eval bounding box expansion from training data

## Notes

- Evaluator RNG is fixed (`EVALUATOR_FIXED_SEED = 2026`) for reproducibility.
- Metrics are method-agnostic; method differences come from model `F` and `project_fn`.
