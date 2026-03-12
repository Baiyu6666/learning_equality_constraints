# Datasets Overview

This folder contains dataset generators and dataset-related assets used by the project.

## 1) Constraint-Manifold Datasets
Main entry:
- `datasets/constraint_datasets.py`
- function: `generate_dataset(name, cfg)`

Dataset summary (explicit names):

| Dataset Name               | Data Dim | Codim | Description                                                                 |
|---------------------------|---------:|------:|-----------------------------------------------------------------------------|
| `2d_figure_eight`         |        2 |     1 | 2D figure-eight curve                                                      |
| `2d_ellipse`              |        2 |     1 | 2D ellipse curve                                                           |
| `2d_noisy_sine`           |        2 |     1 | Sine curve with stronger noise                                             |
| `2d_sine`                 |        2 |     1 | Clean sine curve                                                           |
| `2d_sparse_sine`          |        2 |     1 | Sparse sine samples                                                        |
| `2d_discontinuous`        |        2 |     1 | Piecewise/discontinuous sine-like curve                                    |
| `2d_looped_spiro`         |        2 |     1 | Multi-loop spirograph-like curve                                           |
| `2d_sharp_star`           |        2 |     1 | Star-like closed curve with sharp corners                                  |
| `2d_hetero_noise`         |        2 |     1 | Manifold with non-uniform noise level                                      |
| `2d_square`               |        2 |     1 | 2D square boundary manifold                                                |
| `2d_planar_arm_line_n2`   |        2 |     1 | 2-DoF planar arm, EE constrained on a workspace line                       |
| `3d_spiral`               |        3 |     2 | 3D helix/spiral curve manifold                                             |
| `3d_paraboloid`           |        3 |     1 | 3D paraboloid surface manifold                                             |
| `3d_twosphere`            |        3 |     1 | Outer boundary of two-sphere union                                         |
| `3d_saddle_surface`       |        3 |     1 | Saddle-type surface in 3D                                                  |
| `3d_sphere_surface`       |        3 |     1 | Sphere surface                                                             |
| `3d_torus_surface`        |        3 |     1 | Torus surface                                                              |
| `3d_planar_arm_line_n3`   |        3 |     1 | 3-DoF planar arm, EE constrained on a workspace line                       |
| `3d_spatial_arm_plane_n3` |        3 |     1 | 3-DoF spatial arm, EE constrained on a workspace plane                     |
| `3d_spatial_arm_ellip_n3`|        3 |     2 | 3-DoF spatial arm, EE constrained on workspace ellipse (`x=a cos t, y=b sin t, z=z0`) |
| `6d_spatial_arm_up_n6`    |        6 |     2 | 6-DoF UR5-style arm upward-orientation set (pybullet backend)              |
| `6d_spatial_arm_up_n6_py` |        6 |     2 | 6-DoF UR5-style arm upward-orientation set (analytic backend)              |
| `6d_workspace_sine_surface_pose` | 6 | 3 | Workspace pose `[x,y,z,roll,pitch,yaw]`: position on sine-wave surface, local z-axis aligned with surface normal, free spin around normal |
| `12d_dual_arm` | 12 | 10 | Dual-arm guided-insertion pose `[pose1(6), pose2(6)]`: both end-effectors rigidly grasp the same object, whose center follows a fixed guide curve and may roll around the guide tangent |

Derived naming rules:
- `3d_vz_<base_2d_dataset>`: lift 2D base dataset to 3D with varying `z(x,y)` (`data_dim=3`, `codim=2`).
- `3d_0z_<base_2d_dataset>`: lift 2D base dataset to 3D with `z=0` (`data_dim=3`, `codim=2`).
- `<base_2d_dataset>` can be any of:
  `2d_figure_eight`, `2d_ellipse`, `2d_noisy_sine`, `2d_sine`, `2d_sparse_sine`,
  `2d_discontinuous`, `2d_looped_spiro`, `2d_sharp_star`, `2d_hetero_noise`,
  `2d_square`, `2d_planar_arm_line_n2`.

Return format:
- `x_train`: training points on manifold.
- `grid`: dense manifold samples used as reference/evaluation set.

## 2) UR5 Dataset Helpers
Files:
- `datasets/ur5_n6_dataset.py`
- `datasets/ur5_pybullet_utils.py`

Key functions:
- `sample_ur5_upward_dataset(...)`: pybullet-based UR5 upward-orientation dataset sampling.
- `sample_ur5_upward_dataset_analytic(...)`: analytic/no-pybullet approximation sampler.

## 3) Vendored UR5 Assets
Path:
- `datasets/assets/UR5+gripper/`

Contains URDF + mesh/texture resources copied into this repo for reproducible open-source usage:
- `ur5_gripper.urdf`
- `mesh/`
- `textures/`
