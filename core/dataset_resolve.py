from __future__ import annotations

from typing import Any
import os
import glob

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
    "2d_square",
    "2d_figure_eight",
    "2d_ellipse",
    "2d_noisy_sine",
    "2d_sine",
    "2d_sparse_sine",
    "2d_discontinuous",
    "2d_looped_spiro",
    "2d_sharp_star",
    "2d_hetero_noise",
    "2d_planar_arm_line_n2",
]

BASE_3D_DATASETS = [
    "3d_spiral",
    "3d_paraboloid",
    "3d_paraboloid_traj",
    "3d_twosphere",
    "3d_twosphere_traj",
    "3d_saddle_surface",
    "3d_saddle_surface_traj",
    "3d_sphere_surface",
    "3d_sphere_surface_traj",
    "3d_torus_surface",
    "3d_torus_surface_traj",
    "3d_planar_arm_line_n3",
    "3d_planar_arm_line_n3_traj",
    "3d_spatial_arm_plane_n3",
    "3d_spatial_arm_plane_n3_traj",
    "3d_spatial_arm_circle_n3",
]

BASE_4D_DATASETS = []

BASE_6D_DATASETS = [
    "6d_spatial_arm_up_n6",
    "6d_spatial_arm_up_n6_py",
    "6d_workspace_sine_surface_pose",
    "6d_workspace_sine_surface_pose_traj",
]


def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


_DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
_UR5_CACHE_DIR = os.path.join(_DATASETS_DIR, "generated")


def _print_red(msg: str) -> None:
    print(f"\033[31m{msg}\033[0m")


def _ur5_cache_path(
    *,
    dataset_name: str,
    backend: str,
    seed: int,
    n_train: int,
    n_grid: int,
) -> str:
    os.makedirs(_UR5_CACHE_DIR, exist_ok=True)
    safe_name = str(dataset_name).replace("/", "_")
    safe_backend = str(backend).replace("/", "_")
    fname = (
        f"{safe_name}__backend-{safe_backend}"
        f"__seed-{int(seed)}__ntrain-{int(n_train)}__ngrid-{int(n_grid)}.npz"
    )
    return os.path.join(_UR5_CACHE_DIR, fname)


def _try_load_ur5_cache(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            x = np.asarray(data["x_train"], dtype=np.float32)
            grid = np.asarray(data["grid"], dtype=np.float32)
        _print_red(f"[data][ur5][cache] load: {path}")
        return x, grid
    except Exception:
        return None


def _try_load_ur5_cache_relaxed_n_grid(
    *,
    dataset_name: str,
    backend: str,
    seed: int,
    n_train: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    safe_name = str(dataset_name).replace("/", "_")
    safe_backend = str(backend).replace("/", "_")
    pat = (
        f"{safe_name}__backend-{safe_backend}"
        f"__seed-{int(seed)}__ntrain-{int(n_train)}__ngrid-*.npz"
    )
    cand = glob.glob(os.path.join(_UR5_CACHE_DIR, pat))
    if not cand:
        return None
    # Prefer most recently updated cache file.
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return _try_load_ur5_cache(cand[0])


def _save_ur5_cache(path: str, x: np.ndarray, grid: np.ndarray) -> None:
    np.savez_compressed(
        path,
        x_train=np.asarray(x, dtype=np.float32),
        grid=np.asarray(grid, dtype=np.float32),
    )


def resolve_dataset(
    name: str,
    cfg: Any,
    *,
    optimize_ur5_train_only: bool = True,
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
        true_codim = 2 if name in ("3d_spatial_arm_circle_n3", "3d_spiral") else 1
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
        if name in ("6d_workspace_sine_surface_pose", "6d_workspace_sine_surface_pose_traj"):
            x, grid = generate_dataset(name, cfg)
            return {
                "name": name,
                "x_train": _as_float32(x),
                "grid": _as_float32(grid),
                "data_dim": 6,
                "axis_labels": ("x", "y", "z", "roll", "pitch", "yaw"),
                "true_codim": 3,
                "periodic_joint": False,
            }
        backend = str(ur5_backend or "").lower().strip()
        if backend in ("", "auto"):
            backend = "analytic" if name.endswith("_py") else "pybullet"
        if backend not in ("pybullet", "analytic"):
            raise ValueError(f"unknown ur5_backend '{ur5_backend}', expected 'pybullet' or 'analytic'")

        n_grid = int(max(1, (1 if optimize_ur5_train_only else int(cfg.n_grid))))
        n_train = int(cfg.n_train)
        seed = int(cfg.seed)
        cache_path = _ur5_cache_path(
            dataset_name=name,
            backend=backend,
            seed=seed,
            n_train=n_train,
            n_grid=n_grid,
        )
        cached = _try_load_ur5_cache(cache_path)
        if cached is None and bool(optimize_ur5_train_only):
            cached = _try_load_ur5_cache_relaxed_n_grid(
                dataset_name=name,
                backend=backend,
                seed=seed,
                n_train=n_train,
            )
        if cached is not None:
            x, grid = cached
        else:
            if backend == "pybullet":
                x, grid = sample_ur5_upward_dataset(
                    n_train,
                    n_grid,
                    seed=seed,
                )
            else:
                x, grid = sample_ur5_upward_dataset_analytic(
                    n_train,
                    n_grid,
                    seed=seed,
                )
            _save_ur5_cache(cache_path, x, grid)
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
            "x_train": lift_xy_to_3d_var(x2, cfg).astype(np.float32),
            "grid": lift_xy_to_3d_var(grid2, cfg).astype(np.float32),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": 2,
            "periodic_joint": bool("arm_" in str(base)),
        }
    raise ValueError(f"unknown dataset '{name}'")
