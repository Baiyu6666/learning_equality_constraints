from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from methods.dataaug import knn_normal_bases


def _plot_knn_normals_2d(x_train: np.ndarray, out_path: str) -> None:
    k = max(int(round(0.08 * len(x_train))), 4)
    n_basis = knn_normal_bases(x_train.astype(np.float32), k=k, codim=1, cfg=None)
    nvec = n_basis[:, :, 0]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_train[:, 0], x_train[:, 1], s=8, c="gray", alpha=0.8, label="on-manifold")
    n = len(x_train)
    idx = np.linspace(0, max(0, n - 1), num=min(64, n), dtype=int)
    scale = 0.15 * float(np.mean(np.max(x_train, axis=0) - np.min(x_train, axis=0)))
    plt.quiver(
        x_train[idx, 0],
        x_train[idx, 1],
        nvec[idx, 0],
        nvec[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / max(scale, 1e-6),
        width=0.002,
        color="tab:green",
        alpha=0.9,
        label="KNN normal",
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("EcoMaNN KNN Normals (2D)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _reconstruct_filtered_aug_points_2d(
    x_train: np.ndarray,
    loader_data: dict[str, np.ndarray],
    cfg: Any,
) -> np.ndarray:
    if x_train.shape[1] != 2:
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)
    if not bool(getattr(cfg, "clean_aug_data", True)):
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)
    if not bool(getattr(cfg, "is_performing_data_augmentation", True)):
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)

    cov_null = np.asarray(loader_data["cov_nullspace"], dtype=np.float32)
    cov_svd_s = np.asarray(loader_data["cov_svd_s"], dtype=np.float32)
    n_on = int(x_train.shape[0])
    cov_null = cov_null[:n_on]
    cov_svd_s = cov_svd_s[:n_on]

    dim_ambient = int(x_train.shape[1])
    dim_normal = int(cov_null.shape[2])
    dim_tangent = int(dim_ambient - dim_normal)
    mean_tangent_eig = float(np.mean(cov_svd_s[:, :dim_tangent]))
    eps = float(getattr(cfg, "n_local_neighborhood_mult", 1.0)) * float(np.sqrt(max(mean_tangent_eig, 0.0)))

    # Matches upstream loader default.
    aug_clean_thresh = 1e-1
    kd_tree = cKDTree(data=x_train)
    filtered_pts: list[np.ndarray] = []

    for i in range(1, int(getattr(cfg, "n_normal_space_traversal", 1)) + 1):
        unsigned_level_mult = eps * i
        level_mult_eigvec_list: list[np.ndarray] = []
        for d in range(dim_normal):
            for sign in (-1.0, 1.0):
                level_mult_eigvec = sign * unsigned_level_mult * cov_null[:, :, d]
                level_mult_eigvec_list.append(level_mult_eigvec)
        for level_mult_eigvec in level_mult_eigvec_list:
            new_data = x_train + level_mult_eigvec
            for idx, x in enumerate(new_data):
                _, idx_near = kd_tree.query(x)
                if np.linalg.norm(new_data[idx_near] - x) > (aug_clean_thresh * unsigned_level_mult):
                    filtered_pts.append(new_data[idx])

    if not filtered_pts:
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)
    return np.asarray(filtered_pts, dtype=np.float32)


def _plot_aug_points_2d_with_filtered(
    x_all: np.ndarray,
    level: np.ndarray,
    filtered_pts: np.ndarray,
    out_path: str,
) -> None:
    level = level.reshape(-1)
    on_mask = np.isclose(level, 0.0)
    off_mask = ~on_mask

    plt.figure(figsize=(8, 6))
    if np.any(off_mask):
        sc = plt.scatter(
            x_all[off_mask, 0],
            x_all[off_mask, 1],
            c=level[off_mask],
            s=7,
            cmap="viridis",
            alpha=0.7,
            label="augmented (accepted)",
        )
        cb = plt.colorbar(sc)
        cb.set_label("norm_level_data")
    plt.scatter(x_all[on_mask, 0], x_all[on_mask, 1], s=10, c="black", alpha=0.85, label="on-manifold")
    if filtered_pts.size > 0:
        plt.scatter(
            filtered_pts[:, 0],
            filtered_pts[:, 1],
            s=30,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
            alpha=0.9,
            label="filtered by KNN clean",
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("EcoMaNN Augmented Data (2D)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_training_diagnostics_2d(
    *,
    dataset: str,
    outdir: str,
    x_train: np.ndarray,
    loader_data: dict[str, np.ndarray],
    cfg: Any,
) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    x_train = np.asarray(x_train, dtype=np.float32)
    if x_train.ndim != 2 or x_train.shape[1] != 2:
        return []

    x_all = np.asarray(loader_data.get("data", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    level = np.asarray(loader_data.get("norm_level_data", np.zeros((0, 1), dtype=np.float32)), dtype=np.float32).reshape(-1)
    filtered_pts = _reconstruct_filtered_aug_points_2d(x_train, loader_data, cfg)

    out_knn = os.path.join(outdir, f"{dataset}_ecomann_knn_normals.png")
    out_aug = os.path.join(outdir, f"{dataset}_ecomann_augmented_points.png")
    _plot_knn_normals_2d(x_train, out_knn)
    _plot_aug_points_2d_with_filtered(x_all, level, filtered_pts, out_aug)
    return [out_knn, out_aug]


def save_loss_curves(
    *,
    dataset: str,
    outdir: str,
    train_hist: dict[str, list[float]],
) -> str | None:
    if not isinstance(train_hist, dict) or len(train_hist) == 0:
        return None

    keys = [
        "loss",
        "norm_level",
        "J_nspace",
        "cov_nspace",
        "J_rspace",
        "cov_rspace",
        "siam_reflection",
        "siam_same_levelvec",
        "siam_frac_aug",
    ]
    used = [(k, np.asarray(train_hist.get(k, []), dtype=np.float32)) for k in keys]
    used = [(k, v) for (k, v) in used if v.size > 0 and np.any(np.isfinite(v))]
    if not used:
        return None

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{dataset}_ecomann_loss_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Left: total loss only (easy to read trend).
    for k, v in used:
        if k != "loss":
            continue
        x = np.arange(1, len(v) + 1, dtype=np.int32)
        axes[0].plot(x, v, lw=1.8, label=k)
    if len(axes[0].lines) == 0:
        k, v = used[0]
        x = np.arange(1, len(v) + 1, dtype=np.int32)
        axes[0].plot(x, v, lw=1.8, label=k)
    axes[0].set_title("EcoMaNN Training Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.6)
    axes[0].legend(loc="best", fontsize=8)

    # Right: component losses together (log y for scale separation).
    for k, v in used:
        x = np.arange(1, len(v) + 1, dtype=np.int32)
        axes[1].plot(x, np.clip(v, 1e-12, None), lw=1.3, label=k)
    axes[1].set_yscale("log")
    axes[1].set_title("Loss Components (log scale)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.6)
    axes[1].legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path
