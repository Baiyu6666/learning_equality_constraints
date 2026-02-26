from __future__ import annotations

from typing import Any

import numpy as np

from datasets.constraint_datasets import (
    generate_dataset,
    lift_xy_to_3d_var,
    lift_xy_to_3d_zero,
)
from datasets.ur5_n6_dataset import (
    sample_ur5_upward_dataset,
    sample_ur5_upward_dataset_analytic,
)

BASE_2D_DATASETS = [
    "2d_figure_eight",
    "2d_ellipse",
    "2d_noisy_sine",
    "2d_sine",
    "2d_sparse_sine",
    "2d_discontinuous",
    "2d_looped_spiro",
    "2d_sharp_star",
    "2d_hetero_noise",
    "2d_hairpin",
    "2d_planar_arm_line_n2",
]

BASE_3D_DATASETS = [
    "3d_saddle_surface",
    "3d_sphere_surface",
    "3d_torus_surface",
    "3d_planar_arm_line_n3",
    "3d_spatial_arm_plane_n3",
    "3d_spatial_arm_circle_n3",
]

BASE_4D_DATASETS = [
    "4d_spatial_arm_plane_n4",
]

BASE_6D_DATASETS = [
    "6d_spatial_arm_up_n6",
    "6d_spatial_arm_up_n6_py",
]


def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


def resolve_dataset(
    name: str,
    cfg: Any,
    *,
    optimize_ur5_train_only: bool = False,
    ur5_backend: str | None = None,
) -> dict[str, Any]:
    if name in BASE_2D_DATASETS:
        x, grid = generate_dataset(name, cfg)
        labels = ("q1", "q2") if "2d_planar_arm_line_n2" in name else ("x", "y")
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 2,
            "axis_labels": labels,
            "true_codim": 1,
            "periodic_joint": bool("arm_" in str(name)),
        }
    if name in BASE_3D_DATASETS:
        x, grid = generate_dataset(name, cfg)
        labels = (
            ("q1", "q2", "q3")
            if (
                "3d_planar_arm_line_n3" in name
                or "3d_spatial_arm_plane_n3" in name
                or "3d_spatial_arm_circle_n3" in name
            )
            else ("x", "y", "z")
        )
        true_codim = 2 if name == "3d_spatial_arm_circle_n3" else 1
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": true_codim,
            "periodic_joint": bool("arm_" in str(name)),
        }
    if name in BASE_4D_DATASETS:
        x, grid = generate_dataset(name, cfg)
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 4,
            "axis_labels": ("q1", "q2", "q3", "q4"),
            "true_codim": 1,
            "periodic_joint": True,
        }
    if name in BASE_6D_DATASETS:
        backend = str(ur5_backend or "").lower().strip()
        if backend in ("", "auto"):
            backend = "analytic" if name.endswith("_py") else "pybullet"
        if backend in ("python",):
            backend = "analytic"
        if backend not in ("pybullet", "analytic"):
            raise ValueError(f"unknown ur5_backend '{ur5_backend}', expected 'pybullet' or 'analytic'")

        n_grid = 1 if optimize_ur5_train_only else int(cfg.n_grid)
        if backend == "pybullet":
            x, grid = sample_ur5_upward_dataset(
                int(cfg.n_train),
                int(max(1, n_grid)),
                seed=int(cfg.seed),
            )
        else:
            x, grid = sample_ur5_upward_dataset_analytic(
                int(cfg.n_train),
                int(max(1, n_grid)),
                seed=int(cfg.seed),
            )
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 6,
            "axis_labels": ("q1", "q2", "q3", "q4", "q5", "q6"),
            "true_codim": 2,
            "periodic_joint": True,
            "ur5_backend": backend,
        }
    if name.startswith("3d_0z_"):
        base = name[len("3d_0z_") :]
        if base not in BASE_2D_DATASETS:
            raise ValueError(f"unknown lifted dataset base for 3d_0z_: {base}")
        x2, grid2 = generate_dataset(base, cfg)
        labels = ("q1", "q2", "q3") if "2d_planar_arm_line_n2" in base else ("x", "y", "z")
        return {
            "name": name,
            "x_train": lift_xy_to_3d_zero(x2).astype(np.float32),
            "grid": lift_xy_to_3d_zero(grid2).astype(np.float32),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": 2,
            "periodic_joint": bool("arm_" in str(base)),
        }
    if name.startswith("3d_vz_"):
        base = name[len("3d_vz_") :]
        if base not in BASE_2D_DATASETS:
            raise ValueError(f"unknown lifted dataset base for 3d_vz_: {base}")
        x2, grid2 = generate_dataset(base, cfg)
        labels = ("q1", "q2", "q3") if "2d_planar_arm_line_n2" in base else ("x", "y", "z")
        return {
            "name": name,
            "x_train": lift_xy_to_3d_var(x2).astype(np.float32),
            "grid": lift_xy_to_3d_var(grid2).astype(np.float32),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": 2,
            "periodic_joint": bool("arm_" in str(base)),
        }
    raise ValueError(f"unknown dataset '{name}'")
