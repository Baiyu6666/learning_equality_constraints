# Layered Configs

Priority (low -> high):

1. `base.json`
2. `methods/<method>.json`
3. `datasets/<dataset>.json`
4. `method_dataset/<method>_<dataset>.json`
5. CLI `--override key=value`

Supported methods:

- `eikonal`
- `margin`
- `delta`
- `vae`

Examples:

- `methods/eikonal.json`
- `datasets/3d_planar_arm_line_n3.json`
- `method_dataset/eikonal_3d_planar_arm_line_n3.json`

Current project policy for `n_train` is controlled at dataset layer (`configs/datasets/*.json`):

- 2D datasets: `n_train = 64`
- Lifted 2D datasets (`3d_0z_*`, `3d_vz_*`): `n_train = 64`
- 3D datasets: `n_train = 512`
- 4D datasets: `n_train = 512`
- 6D datasets: `n_train = 2048`

Example override:

```bash
--override epochs=1200 --override eval_tau_ratio=0.02
```

Nested override example:

```bash
--override train.lr=0.0001
```

Note: nested keys only apply if your config schema uses nested objects.
Current dataclass configs are mostly flat keys.

## W&B Sweep (Multi-dataset Average)

Use `runners/run_sweep.py` so one sweep trial runs all target datasets for one method at one seed, then logs averaged metrics over datasets.

Example:

```bash
python -m runners.run_sweep_launch --yaml configs/sweeps/eikonal_multids.yaml
```

In the sweep YAML:

- `command` should pass fixed run context to `runners/run_sweep.py`, including `--datasets`. `--method` is optional when `method` is provided by sweep parameters.
- Use sweep parameter `seed` and pass `--seed ${seed}` in `command`. This creates one run per seed.
- Sweep parameters (e.g. `lr`, `epochs`) are automatically converted to config overrides.
- Dataset-specific sweep key format: `ds__<dataset>__<param>` (example: `ds__3d_spiral__proj_steps`).
- Method-specific sweep key format: `m__<method>__<param>` (example: `m__eikonal__lam_eikonal`).
- Method+dataset-specific sweep key format: `m__<method>__ds__<dataset>__<param>`.
- Each trial adds a method tag to W&B: `method:<method>` (customizable via `--method-tag-prefix`).
- Recommended sweep algorithm here is `method: grid`.
