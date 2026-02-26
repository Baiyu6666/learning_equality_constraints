#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.baseline_udf.baseline_udf import (
    Config,
    _as_sigmas,
    _local_pca_frame,
    _radius_knn_indices,
    compute_adaptive_sigma,
    effective_knn_norm_estimation_points,
    effective_knn_off_data_filter_points,
    generate_dataset,
    pairwise_sqdist,
    set_seed,
)


DEFAULT_DATASETS = [
    "2d_figure_eight",
    "2d_ellipse",
    "2d_sine",
    "2d_noisy_sine",
    "2d_sparse_sine",
    "2d_discontinuous",
    "2d_hetero_noise",
    "2d_looped_spiro",
    "2d_sharp_star",
    "2d_hairpin",
]


def _parse_sigmas(raw: str) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("sigmas cannot be empty")
    return tuple(vals)


def _sample_demo_off_points(
    x: np.ndarray,
    idx: int,
    nvec: np.ndarray,
    cfg: Config,
    sigma_per_point: np.ndarray | None,
    knn_off_data_filter_points: int,
    knn_norm_estimation_points: int,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    m = 30
    r_pos = None
    r_neg = None
    if sigma_per_point is not None:
        r = float(sigma_per_point[idx])
        if bool(getattr(cfg, "demo_adp_sigma_asymmetric_enable", False)):
            d2_row = pairwise_sqdist(x[idx : idx + 1], x)[0]
            order = np.argsort(d2_row)
            rank = min(
                len(order) - 1,
                max(
                    1,
                    int(knn_norm_estimation_points)
                    + int(getattr(cfg, "demo_adp_sigma_nonlocal_offset", 1)),
                ),
            )
            j = int(order[rank])
            side = float(np.dot(x[j] - x[idx], nvec))
            danger_scale = float(getattr(cfg, "demo_adp_sigma_danger_scale", 0.6))
            safe_scale = float(getattr(cfg, "demo_adp_sigma_safe_scale", 1.0))
            if side >= 0.0:
                r_pos = r * danger_scale
                r_neg = r * safe_scale
            else:
                r_pos = r * safe_scale
                r_neg = r * danger_scale
            r_pos = float(np.clip(r_pos, cfg.adp_sigma_r_min, cfg.adp_sigma_r_max))
            r_neg = float(np.clip(r_neg, cfg.adp_sigma_r_min, cfg.adp_sigma_r_max))
            z = np.random.randn(m, 1).astype(np.float32)
            s = np.where(z >= 0.0, z * (r_pos / 1.645), z * (r_neg / 1.645)).astype(
                np.float32
            )
            s = np.clip(s, -r_neg, r_pos).astype(np.float32)
        else:
            r_pos = r
            r_neg = r
            s = np.random.randn(m, 1).astype(np.float32) * (r / 1.645)
    else:
        sigmas = _as_sigmas(cfg.sigmas)
        sigma = np.random.choice(sigmas, size=(m, 1))
        s = np.random.randn(m, 1).astype(np.float32) * sigma
    delta_pts = x[idx : idx + 1] + s * nvec.reshape(1, -1)
    # A point is "pass" only if it satisfies all enabled checks.
    mask = np.ones(len(delta_pts), dtype=bool)
    if cfg.use_adaptive_sigma:
        if r_pos is not None and r_neg is not None:
            sv = s[:, 0]
            mask &= (sv >= -r_neg) & (sv <= r_pos)
        else:
            r_kappa = float(sigma_per_point[idx]) if sigma_per_point is not None else 0.0
            mask &= np.abs(s)[:, 0] <= r_kappa
    if cfg.use_knn_filter:
        d2_off = pairwise_sqdist(delta_pts, x)
        nn_idx = np.argsort(d2_off, axis=1)[:, :knn_off_data_filter_points]
        mask &= (nn_idx == idx).any(axis=1)
    return delta_pts[mask], delta_pts[~mask], r_pos, r_neg


def _smooth_sigma_on_knn_graph(
    x: np.ndarray,
    sigma: np.ndarray,
    k: int,
    iters: int,
    mode: str,
) -> np.ndarray:
    if len(sigma) == 0:
        return sigma
    d2 = pairwise_sqdist(x, x)
    order = np.argsort(d2, axis=1)
    k = int(np.clip(k, 1, max(1, len(x) - 1)))
    nbr_idx = order[:, 1 : k + 1]
    out = sigma.astype(np.float32).copy()
    for _ in range(max(0, int(iters))):
        new = out.copy()
        for i in range(len(out)):
            vals = np.concatenate(([out[i]], out[nbr_idx[i]]))
            if mode == "mean":
                new[i] = float(np.mean(vals))
            else:
                new[i] = float(np.median(vals))
        out = new
    return out


def _plot_one_dataset(
    ax: plt.Axes,
    name: str,
    cfg: Config,
    idx_count: int,
    rng: np.random.Generator,
) -> None:
    x, _ = generate_dataset(name, cfg)
    if x.shape[1] != 2:
        ax.set_title(f"{name} (skip: not 2D)")
        ax.axis("off")
        return

    knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x))
    knn_off_data_filter_points = effective_knn_off_data_filter_points(cfg, len(x))
    sigma_per_point = (
        compute_adaptive_sigma(
            x, cfg, knn_norm_estimation_points=knn_norm_estimation_points
        )
        if cfg.use_adaptive_sigma
        else None
    )
    if sigma_per_point is not None and getattr(cfg, "demo_sigma_smooth_enable", False):
        sigma_per_point = _smooth_sigma_on_knn_graph(
            x=x,
            sigma=sigma_per_point,
            k=int(getattr(cfg, "demo_sigma_smooth_k", 8)),
            iters=int(getattr(cfg, "demo_sigma_smooth_iters", 1)),
            mode=str(getattr(cfg, "demo_sigma_smooth_mode", "median")).lower(),
        )

    d2 = pairwise_sqdist(x, x)
    count = min(idx_count, len(x))
    idx_list = rng.choice(len(x), size=count, replace=False)

    ax.scatter(x[:, 0], x[:, 1], s=6, alpha=0.35, color="gray")
    for idx in idx_list:
        if cfg.use_radius_knn:
            nn_idx = _radius_knn_indices(d2[idx], cfg.radius_knn_k, cfg.radius_knn_scale)
        else:
            nn_idx = np.argsort(d2, axis=1)[idx, 1 : knn_norm_estimation_points + 1]
        nbrs = x[nn_idx]
        _, evecs, _, _ = _local_pca_frame(nbrs, x[idx], cfg=cfg)
        nvec = evecs[:, 0]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)

        ax.scatter(nbrs[:, 0], nbrs[:, 1], s=6, color="blue", alpha=0.7)
        ax.scatter(x[idx, 0], x[idx, 1], s=9, color="red")

        pass_pts, fail_pts, r_pos, r_neg = _sample_demo_off_points(
            x,
            idx,
            nvec,
            cfg,
            sigma_per_point,
            knn_off_data_filter_points,
            knn_norm_estimation_points,
        )
        if len(pass_pts) > 0:
            ax.scatter(pass_pts[:, 0], pass_pts[:, 1], s=3, color="green", alpha=0.45)
        if len(fail_pts) > 0:
            ax.scatter(fail_pts[:, 0], fail_pts[:, 1], s=3, color="orange", alpha=0.45)

        # Use one red segment to show the clipping interval along the normal.
        # In asymmetric mode, the two sides naturally have different lengths.
        if cfg.use_adaptive_sigma and sigma_per_point is not None:
            r_pos_vis = float(r_pos) if r_pos is not None else float(sigma_per_point[idx])
            r_neg_vis = float(r_neg) if r_neg is not None else float(sigma_per_point[idx])
        else:
            scale = 0.35
            r_pos_vis = scale
            r_neg_vis = scale
        p_pos = x[idx] + nvec * r_pos_vis
        p_neg = x[idx] - nvec * r_neg_vis
        ax.plot(
            [p_neg[0], p_pos[0]],
            [p_neg[1], p_pos[1]],
            color="gray",
            linewidth=1,
            alpha=0.95,
        )
        ax.scatter(
            [p_neg[0], p_pos[0]],
            [p_neg[1], p_pos[1]],
            s=4,
            color="black",
            alpha=0.95,
        )

    ax.set_title(name, fontsize=10)
    ax.set_aspect("equal", adjustable="box")


def _estimate_k_stats_for_dataset(
    x: np.ndarray,
    cfg: Config,
    idx_list: np.ndarray,
    knn_norm_estimation_points: int,
) -> tuple[float, float]:
    d2 = pairwise_sqdist(x, x)
    k_before_list = []
    k_after_list = []
    keep_ratio = float(np.clip(cfg.trimmed_pca_keep_ratio, 0.1, 1.0))
    for idx in idx_list:
        if cfg.use_radius_knn:
            nn_idx = _radius_knn_indices(d2[idx], cfg.radius_knn_k, cfg.radius_knn_scale)
        else:
            nn_idx = np.argsort(d2, axis=1)[idx, 1 : knn_norm_estimation_points + 1]
        k_before = len(nn_idx)
        k_after = k_before
        if cfg.use_trimmed_pca and k_before >= max(6, x.shape[1] + 2):
            k_after = int(np.clip(math.ceil(keep_ratio * k_before), x.shape[1] + 1, k_before))
        k_before_list.append(float(k_before))
        k_after_list.append(float(k_after))
    if not k_before_list:
        return 0.0, 0.0
    return float(np.mean(k_before_list)), float(np.mean(k_after_list))


def _plot_mode_comparison_per_dataset(
    datasets: List[str],
    cfg_base: Config,
    idx_count: int,
    seed: int,
    outdir: Path,
) -> None:
    modes = [
        ("none", False, False, False),
        ("trimmed", False, True, False),
        ("weighted", True, False, False),
        ("smoothed", False, False, True),
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        x_ds, _ = generate_dataset(ds, cfg_base)
        if x_ds.shape[1] != 2:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_arr = np.array(axes).reshape(-1)
        for i, (label, use_weighted, use_trimmed, use_smoothed) in enumerate(modes):
            cfg_mode = copy.deepcopy(cfg_base)
            cfg_mode.use_weighted_pca = use_weighted
            cfg_mode.use_trimmed_pca = use_trimmed
            cfg_mode.demo_sigma_smooth_enable = use_smoothed
            knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg_mode, len(x_ds))
            # Keep comparison fair: same idx draws and same off-point RNG seed per mode.
            idx_rng = np.random.default_rng(seed + 2026)
            idx_count_eff = min(idx_count, len(x_ds))
            idx_list = idx_rng.choice(len(x_ds), size=idx_count_eff, replace=False)
            k_before, k_after = _estimate_k_stats_for_dataset(
                x_ds, cfg_mode, idx_list, knn_norm_estimation_points
            )
            # Keep comparison fair: same idx draws and same off-point RNG seed per mode.
            rng_mode = np.random.default_rng(seed + 2026)
            np.random.seed(seed + 1234)
            _plot_one_dataset(axes_arr[i], ds, cfg_mode, idx_count, rng_mode)
            if cfg_mode.use_trimmed_pca:
                title = f"{ds}: {label} | k={k_before:.1f}->{k_after:.1f}"
            else:
                title = f"{ds}: {label} | k={k_before:.1f}"
            axes_arr[i].set_title(title, fontsize=10)
        fig.tight_layout()
        out_path = outdir / f"pca_mode_compare_{ds}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"saved: {out_path}")


def _plot_sampling_mode_comparison_per_dataset(
    datasets: List[str],
    cfg_base: Config,
    idx_count: int,
    seed: int,
    outdir: Path,
) -> None:
    # Four requested modes:
    # 1) fixed sigma
    # 2) fixed sigma + knn filter
    # 3) adaptive sigma
    # 4) adaptive sigma + knn filter
    modes = [
        ("fixed_sigma", False, False),
        ("fixed_sigma+filter", False, True),
        ("adaptive_sigma", True, False),
        ("adaptive_sigma+filter", True, True),
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        x_ds, _ = generate_dataset(ds, cfg_base)
        if x_ds.shape[1] != 2:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_arr = np.array(axes).reshape(-1)
        for i, (label, use_adaptive, use_filter) in enumerate(modes):
            cfg_mode = copy.deepcopy(cfg_base)
            cfg_mode.use_adaptive_sigma = use_adaptive
            cfg_mode.use_knn_filter = use_filter
            # Keep index and off-point randomness fixed across modes for fair comparison.
            rng_mode = np.random.default_rng(seed + 2026)
            np.random.seed(seed + 1234)
            _plot_one_dataset(axes_arr[i], ds, cfg_mode, idx_count, rng_mode)
            axes_arr[i].set_title(f"{ds}: {label}", fontsize=10)
        fig.tight_layout()
        out_path = outdir / f"sampling_mode_compare_{ds}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs_levelset_datasets/adaptive_sigma_knn_demo.png")
    parser.add_argument(
        "--compare_mode_outdir",
        default="outputs_levelset_datasets/pca_mode_comparison",
    )
    parser.add_argument("--compare_modes", action="store_true", default=False)
    parser.add_argument(
        "--compare_sampling_modes",
        action="store_true",
        default=False,
        help="Compare fixed/adaptive sigma with/without knn filter per dataset.",
    )
    parser.add_argument(
        "--no_compare",
        action="store_true",
        default=False,
        help="Force disable all compare modes even if compare flags are present.",
    )
    parser.add_argument(
        "--compare_sampling_outdir",
        default="outputs_levelset_datasets/sampling_mode_comparison",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--idx_count", type=int, default=120)
    parser.add_argument("--sigmas", type=str, default="0.2")
    parser.add_argument("--use_adaptive_sigma", action="store_true", default=False)
    parser.add_argument("--no_adaptive_sigma", action="store_true")
    parser.add_argument("--use_knn_filter", action="store_true", default=True)
    parser.add_argument("--use_radius_knn", action="store_true", default=False)
    parser.add_argument("--use_weighted_pca", action="store_true", default=True)
    parser.add_argument("--no_weighted_pca", action="store_true")
    parser.add_argument("--use_trimmed_pca", action="store_true", default=False)
    parser.add_argument("--trimmed_pca_keep_ratio", type=float, default=0.6)
    parser.add_argument("--smooth_sigma_graph", action="store_true", default=not False)
    parser.add_argument("--sigma_smooth_k", type=int, default=20)
    parser.add_argument("--sigma_smooth_iters", type=int, default=1)
    parser.add_argument("--sigma_smooth_mode", choices=["median", "mean"], default="median")
    parser.add_argument("--knn_norm_estimation_ratio", type=float, default=0.08)
    parser.add_argument("--knn_norm_estimation_min_points", type=int, default=5)
    parser.add_argument("--knn_off_data_filter_ratio", type=float, default=0.03)
    parser.add_argument("--knn_off_data_filter_min_points", type=int, default=1)
    parser.add_argument("--radius_knn_k", type=int, default=4)
    parser.add_argument("--radius_knn_scale", type=float, default=1.5)
    parser.add_argument("--adp_sigma_scale", type=float, default=1)
    parser.add_argument("--adp_sigma_kappa_exp", type=float, default=1)
    parser.add_argument("--adp_sigma_r_min", type=float, default=0.01)
    parser.add_argument("--adp_sigma_r_max", type=float, default=1)
    parser.add_argument("--adp_sigma_eps", type=float, default=1e-6)
    parser.add_argument("--adp_sigma_asymmetric", action="store_true", default=not False)
    parser.add_argument("--adp_sigma_danger_scale", type=float, default=0.6)
    parser.add_argument("--adp_sigma_safe_scale", type=float, default=1.0)
    parser.add_argument("--adp_sigma_nonlocal_offset", type=int, default=1)
    args = parser.parse_args()
    if args.no_compare:
        args.compare_modes = False
        args.compare_sampling_modes = False

    set_seed(args.seed)
    cfg = Config()
    cfg.device = "cpu"
    cfg.n_train = int(args.n_train)
    cfg.sigmas = _parse_sigmas(args.sigmas)
    cfg.knn_norm_estimation_ratio = float(args.knn_norm_estimation_ratio)
    cfg.knn_norm_estimation_min_points = int(args.knn_norm_estimation_min_points)
    cfg.knn_off_data_filter_ratio = float(args.knn_off_data_filter_ratio)
    cfg.knn_off_data_filter_min_points = int(args.knn_off_data_filter_min_points)
    cfg.use_knn_filter = bool(args.use_knn_filter)
    cfg.use_radius_knn = bool(args.use_radius_knn)
    cfg.use_weighted_pca = bool(args.use_weighted_pca) and (not args.no_weighted_pca)
    cfg.use_trimmed_pca = bool(args.use_trimmed_pca)
    cfg.trimmed_pca_keep_ratio = float(args.trimmed_pca_keep_ratio)
    cfg.demo_sigma_smooth_enable = bool(args.smooth_sigma_graph)
    cfg.demo_sigma_smooth_k = int(args.sigma_smooth_k)
    cfg.demo_sigma_smooth_iters = int(args.sigma_smooth_iters)
    cfg.demo_sigma_smooth_mode = str(args.sigma_smooth_mode)
    cfg.radius_knn_k = int(args.radius_knn_k)
    cfg.radius_knn_scale = float(args.radius_knn_scale)
    cfg.use_adaptive_sigma = bool(args.use_adaptive_sigma) and (not args.no_adaptive_sigma)
    cfg.adp_sigma_scale = float(args.adp_sigma_scale)
    cfg.adp_sigma_kappa_exp = float(args.adp_sigma_kappa_exp)
    cfg.adp_sigma_r_min = float(args.adp_sigma_r_min)
    cfg.adp_sigma_r_max = float(args.adp_sigma_r_max)
    cfg.adp_sigma_eps = float(args.adp_sigma_eps)
    cfg.demo_adp_sigma_asymmetric_enable = bool(args.adp_sigma_asymmetric)
    cfg.demo_adp_sigma_danger_scale = float(args.adp_sigma_danger_scale)
    cfg.demo_adp_sigma_safe_scale = float(args.adp_sigma_safe_scale)
    cfg.demo_adp_sigma_nonlocal_offset = int(args.adp_sigma_nonlocal_offset)

    datasets: List[str] = list(args.datasets)
    if args.compare_sampling_modes:
        _plot_sampling_mode_comparison_per_dataset(
            datasets=datasets,
            cfg_base=cfg,
            idx_count=args.idx_count,
            seed=args.seed,
            outdir=Path(args.compare_sampling_outdir),
        )
        return

    if args.compare_modes:
        _plot_mode_comparison_per_dataset(
            datasets=datasets,
            cfg_base=cfg,
            idx_count=args.idx_count,
            seed=args.seed,
            outdir=Path(args.compare_mode_outdir),
        )
        return

    n = len(datasets)
    cols = 4
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    axes_arr = np.array(axes).reshape(-1)
    rng = np.random.default_rng(args.seed + 2026)

    for i, name in enumerate(datasets):
        _plot_one_dataset(axes_arr[i], name, cfg, args.idx_count, rng)
    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(
        (
            "KNN normal demo (same logic as main): "
            f"adaptive={cfg.use_adaptive_sigma}, knn_filter={cfg.use_knn_filter}, "
            f"radius_knn={cfg.use_radius_knn}, weighted_pca={cfg.use_weighted_pca}, "
            f"trimmed_pca={cfg.use_trimmed_pca}, kappa_exp={cfg.adp_sigma_kappa_exp:g}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
