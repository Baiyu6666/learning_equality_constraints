# Datasets Overview

This folder contains dataset generators and dataset-related assets used by the project.

## 1) Constraint-Manifold Datasets
Main entry:
- `datasets/constraint_datasets.py`
- function: `generate_dataset(name, cfg)`

Dataset summary:

| dataset name | data_dim | codim | description |
| --- | ---: | ---: | --- |
| `2d_figure_eight` | 2 | 1 | 2D figure-eight curve |
| `2d_ellipse` | 2 | 1 | 2D ellipse curve |
| `2d_noisy_sine` | 2 | 1 | sine curve with stronger noise |
| `2d_sine` | 2 | 1 | clean sine curve |
| `2d_sparse_sine` | 2 | 1 | sparse sine samples |
| `2d_discontinuous` | 2 | 1 | piecewise/discontinuous sine-like curve |
| `2d_looped_spiro` | 2 | 1 | multi-loop spirograph-like curve |
| `2d_sharp_star` | 2 | 1 | star-like closed curve with sharp corners |
| `2d_hetero_noise` | 2 | 1 | manifold with non-uniform noise level |
| `2d_hairpin` | 2 | 1 | hairpin-like curve |
| `2d_planar_arm_line_n2` | 2 | 1 | 2-DoF planar arm, EE constrained on a workspace line |
| `3d_saddle_surface` | 3 | 1 | saddle-type surface in 3D |
| `3d_sphere_surface` | 3 | 1 | sphere surface |
| `3d_torus_surface` | 3 | 1 | torus surface |
| `3d_planar_arm_line_n3` | 3 | 1 | 3-DoF planar arm, EE constrained on a workspace line |
| `3d_spatial_arm_plane_n3` | 3 | 1 | 3-DoF spatial arm, EE constrained on a workspace plane |
| `3d_spatial_arm_circle_n3` | 3 | 2 | 3-DoF spatial arm, EE constrained on workspace circle (`x^2+y^2=r^2`, `z=z0`) |
| `4d_spatial_arm_plane_n4` | 4 | 1 | 4-DoF spatial arm, EE constrained on a workspace plane |
| `6d_spatial_arm_up_n6` | 6 | 2 | 6-DoF UR5-style arm, tool orientation constrained upward |

Lifted variants (used by `methods/vector_eikonal/vector_eikonal.py`):
- naming rule: `3d_vz_<base_2d_dataset>`
- `data_dim=3`, `codim=2`
- generated on-the-fly by lifting 2D base data with a smooth varying `z(x,y)`:
  - includes `3d_vz_2d_figure_eight`, `3d_vz_2d_ellipse`, `3d_vz_2d_noisy_sine`, `3d_vz_2d_sine`, `3d_vz_2d_sparse_sine`, `3d_vz_2d_discontinuous`, `3d_vz_2d_looped_spiro`, `3d_vz_2d_sharp_star`, `3d_vz_2d_hetero_noise`, `3d_vz_2d_hairpin`, `3d_vz_2d_planar_arm_line_n2`.

Return format:
- `x_train`: training points on manifold.
- `grid`: dense manifold samples used as reference/evaluation set.

## 2) VAE Baseline Datasets
Main entry:
- `datasets/vae_datasets.py`
- function: `build_datasets()`

Current keys:
- `3d_spiral`: 3D spiral manifold.
- `3d_paraboloid`: 3D paraboloid manifold.
- `3d_twosphere`: two-sphere style dataset in 3D.
- `2d_square`: 2D square boundary manifold.

## 3) UR5 Dataset Helpers
Files:
- `datasets/ur5_n6_dataset.py`
- `datasets/ur5_pybullet_utils.py`

Key functions:
- `sample_ur5_upward_dataset(...)`: pybullet-based UR5 upward-orientation dataset sampling.
- `sample_ur5_upward_dataset_analytic(...)`: analytic/no-pybullet approximation sampler.

## 4) Vendored UR5 Assets
Path:
- `datasets/assets/UR5+gripper/`

Contains URDF + mesh/texture resources copied into this repo for reproducible open-source usage:
- `ur5_gripper.urdf`
- `mesh/`
- `textures/`
