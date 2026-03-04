#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from datasets.constraint_datasets import generate_dataset


TRAJ_DATASETS = [
    "3d_paraboloid_traj",
    "3d_twosphere_traj",
    "3d_saddle_surface_traj",
    "3d_sphere_surface_traj",
    "3d_torus_surface_traj",
    "3d_planar_arm_line_n3_traj",
    "3d_spatial_arm_plane_n3_traj",
]


def _project_root() -> str:
    # tools/analysis/preview_3d_traj_datasets.py -> repo root is ../../
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_config_root(config_root: str) -> str:
    p = str(config_root).strip() or "configs"
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    # Fallback: resolve relative to project root (works when cwd != repo root).
    return os.path.join(_project_root(), p)


def _resolve_outdir(outdir: str) -> str:
    p = str(outdir).strip() or "outputs_unified/traj_preview_3d"
    if os.path.isabs(p):
        return p
    return os.path.join(_project_root(), p)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview 3D traj datasets with GT surface/grid.")
    p.add_argument("--config-root", default="configs", help="Config root containing datasets/*.json")
    p.add_argument("--outdir", default="outputs_unified/traj_preview_3d", help="Output directory")
    p.add_argument("--seed", type=int, default=173, help="Override dataset seed for preview")
    p.add_argument("--max-grid", type=int, default=2500, help="Max GT grid points to draw")
    return p.parse_args()


def _load_cfg(config_root: str, dataset_name: str, seed: int) -> SimpleNamespace:
    path = os.path.join(config_root, "datasets", f"{dataset_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["seed"] = int(seed)
    return SimpleNamespace(**cfg)


def _plot_one(
    *,
    name: str,
    x_train: np.ndarray,
    grid: np.ndarray,
    traj_count: int,
    out_path: str,
    max_grid: int,
) -> None:
    rng = np.random.default_rng(0)
    g = grid.astype(np.float32)
    if len(g) > int(max_grid):
        idx = rng.choice(len(g), size=int(max_grid), replace=False)
        g = g[idx]

    n = int(x_train.shape[0])
    k = max(1, int(traj_count))
    seg_len = max(2, int(math.ceil(n / float(k))))

    fig = plt.figure(figsize=(10.5, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Left: GT manifold surface (preferred) + dataset scatter overlay.
    # Fallback to GT scatter when triangulated surface is ill-conditioned.
    g_surf = grid.astype(np.float32)
    if len(g_surf) > 1800:
        idx = rng.choice(len(g_surf), size=1800, replace=False)
        g_surf = g_surf[idx]
    surface_drawn = False
    try:
        ax1.plot_trisurf(
            g_surf[:, 0],
            g_surf[:, 1],
            g_surf[:, 2],
            color="#A7C7E7",
            alpha=0.38,
            linewidth=0.12,
            edgecolor=(0.45, 0.55, 0.70, 0.35),
            antialiased=True,
        )
        surface_drawn = True
    except Exception:
        surface_drawn = False
    if not surface_drawn:
        ax1.scatter(g[:, 0], g[:, 1], g[:, 2], s=1.6, c="#9CA3AF", alpha=0.28, linewidths=0)
    ax1.scatter(
        x_train[:, 0], x_train[:, 1], x_train[:, 2],
        s=4.0, c="#1D4ED8", alpha=0.55, linewidths=0
    )
    ax1.set_title(f"{name} - GT manifold + dataset scatter")
    ax1.set_xlabel("x/q1")
    ax1.set_ylabel("y/q2")
    ax1.set_zlabel("z/q3")

    # Right: trajectory line plot
    ax2.scatter(g[:, 0], g[:, 1], g[:, 2], s=1.0, c="#D1D5DB", alpha=0.16, linewidths=0)
    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2"]
    for i in range(k):
        a = i * seg_len
        b = min(n, (i + 1) * seg_len)
        if b - a < 2:
            continue
        tr = x_train[a:b]
        c = colors[i % len(colors)]
        ax2.plot(tr[:, 0], tr[:, 1], tr[:, 2], "-", color=c, linewidth=1.3, alpha=0.9)
        ax2.scatter([tr[0, 0]], [tr[0, 1]], [tr[0, 2]], s=14, c=c, alpha=0.95)
    ax2.set_title(f"{name} - trajectory lines")
    ax2.set_xlabel("x/q1")
    ax2.set_ylabel("y/q2")
    ax2.set_zlabel("z/q3")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config_root = _resolve_config_root(args.config_root)
    outdir = _resolve_outdir(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    for ds_name in TRAJ_DATASETS:
        cfg = _load_cfg(config_root, ds_name, seed=int(args.seed))
        x_train, grid = generate_dataset(ds_name, cfg)
        traj_count = int(getattr(cfg, "traj_count", max(8, int(cfg.n_train) // 24)))
        out_path = os.path.join(outdir, f"{ds_name}_preview.png")
        _plot_one(
            name=ds_name,
            x_train=x_train,
            grid=grid,
            traj_count=traj_count,
            out_path=out_path,
            max_grid=int(args.max_grid),
        )
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
