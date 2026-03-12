from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Iterable

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_METHODS = ["dataaug", "oncl", "vae", "ecomann"]
DEFAULT_METRICS = [
    "proj_manifold_dist",
    "pred_precision",
    "train_seconds",
    "gt_to_learned_mean",
    "learned_to_gt_mean",
    "bidirectional_chamfer",
]


def _split_csv(text: str | None, default: Iterable[str] | None = None) -> list[str]:
    if text is None or str(text).strip() == "":
        return list(default) if default is not None else []
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _resolve_bench_dir(bench: str) -> str:
    b = str(bench).strip()
    if not b:
        raise ValueError("--bench is required")
    if os.path.isabs(b):
        return b
    p1 = os.path.abspath(os.path.join(_PROJECT_ROOT, "outputs", "bench", b))
    if os.path.isdir(p1):
        return p1
    p2 = os.path.abspath(os.path.join(_PROJECT_ROOT, b))
    if os.path.isdir(p2):
        return p2
    p3 = os.path.abspath(b)
    return p3


def _to_float(v: object, default: float = np.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


def _load_from_per_case_csv(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "dataset": str(r.get("dataset", "")).strip(),
                    "method": str(r.get("method", "")).strip(),
                    "seed": str(r.get("seed", "")).strip() or "0",
                    "bidirectional_chamfer": _to_float(r.get("bidirectional_chamfer", "")),
                    "proj_manifold_dist": _to_float(r.get("proj_manifold_dist", "")),
                    "pred_precision": _to_float(r.get("pred_precision", "")),
                    "train_seconds": _to_float(r.get("train_seconds", "")),
                    "gt_to_learned_mean": _to_float(r.get("gt_to_learned_mean", "")),
                    "learned_to_gt_mean": _to_float(r.get("learned_to_gt_mean", "")),
                }
            )
    return rows


def _load_from_per_run_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            metrics = r.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            rows.append(
                {
                    "dataset": str(r.get("dataset", "")).strip(),
                    "method": str(r.get("method", "")).strip(),
                    "seed": str(r.get("seed", "")).strip() or "0",
                    # per_run_metrics.jsonl stores metrics under nested key "metrics";
                    # keep backward-compatible fallback to top-level fields.
                    "bidirectional_chamfer": _to_float(metrics.get("bidirectional_chamfer", r.get("bidirectional_chamfer", ""))),
                    "proj_manifold_dist": _to_float(metrics.get("proj_manifold_dist", r.get("proj_manifold_dist", ""))),
                    "pred_precision": _to_float(metrics.get("pred_precision", r.get("pred_precision", ""))),
                    "train_seconds": _to_float(metrics.get("train_seconds", r.get("train_seconds", ""))),
                    "gt_to_learned_mean": _to_float(metrics.get("gt_to_learned_mean", r.get("gt_to_learned_mean", ""))),
                    "learned_to_gt_mean": _to_float(metrics.get("learned_to_gt_mean", r.get("learned_to_gt_mean", ""))),
                }
            )
    return rows


def _load_from_eval_jsons(bench_dir: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_dir in sorted(glob.glob(os.path.join(bench_dir, "*"))):
        if not os.path.isdir(method_dir):
            continue
        method = os.path.basename(method_dir)
        for p in sorted(glob.glob(os.path.join(method_dir, "*_eval.json"))):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            stem = os.path.basename(p)
            dataset = stem
            if dataset.endswith("_eval.json"):
                dataset = dataset[: -len("_eval.json")]
            if dataset.endswith(f"_{method}"):
                dataset = dataset[: -(len(method) + 1)]
            dataset = dataset.replace("_oncl", "")
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": "last",
                    "bidirectional_chamfer": _to_float(d.get("bidirectional_chamfer", "")),
                    "proj_manifold_dist": _to_float(d.get("proj_manifold_dist", "")),
                    "pred_precision": _to_float(d.get("pred_precision", "")),
                    "train_seconds": _to_float(d.get("train_seconds", "")),
                    "gt_to_learned_mean": _to_float(d.get("gt_to_learned_mean", "")),
                    "learned_to_gt_mean": _to_float(d.get("learned_to_gt_mean", "")),
                }
            )
    return rows


def _load_rows(bench_dir: str) -> tuple[list[dict[str, object]], str]:
    per_run = os.path.join(bench_dir, "per_run_metrics.jsonl")
    if os.path.isfile(per_run):
        return _load_from_per_run_jsonl(per_run), "per_run_metrics.jsonl"
    per_case = os.path.join(bench_dir, "per_case_metrics.csv")
    if os.path.isfile(per_case):
        return _load_from_per_case_csv(per_case), "per_case_metrics.csv"
    return _load_from_eval_jsons(bench_dir), "*_eval.json"


def _infer_dataset_order(rows: list[dict[str, object]], methods: list[str]) -> list[str]:
    known = [
        "2d_ellipse",
        "2d_looped_spiro",
        "2d_planar_arm_line_n2",
        "2d_sparse_sine",
        "3d_planar_arm_line_n3",
        "3d_planar_arm_line_n3_traj",
        "3d_spatial_arm_ellip_n3",
        "3d_spatial_arm_ellip_n3_traj",
        "3d_twosphere",
        "3d_twosphere_traj",
        "3d_torus_surface",
        "3d_torus_surface_traj",
        "3d_vz_2d_ellipse",
        "3d_vz_2d_ellipse_traj",
        "6d_spatial_arm_up_n6_py",
        "6d_spatial_arm_up_n6_py_traj",
        "6d_workspace_sine_surface_pose",
        "6d_workspace_sine_surface_pose_traj",
    ]
    seen = sorted({str(r["dataset"]) for r in rows if str(r.get("dataset", ""))})
    out: list[str] = [d for d in known if d in seen]
    out.extend([d for d in seen if d not in out])
    return out


def _build_samples(
    rows: list[dict[str, object]],
    datasets: list[str],
    methods: list[str],
    metric: str,
    fill_missing_zero: bool,
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {d: {m: [] for m in methods} for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        if d not in out or m not in out[d]:
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            out[d][m].append(float(v))

    if fill_missing_zero:
        for d in datasets:
            for m in methods:
                if len(out[d][m]) == 0:
                    out[d][m] = [0.0]
    return out


def _compute_mean_std(samples: dict[str, dict[str, list[float]]], datasets: list[str], methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    stds = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    for mi, m in enumerate(methods):
        for di, d in enumerate(datasets):
            vals = np.asarray(samples[d][m], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means[mi, di] = 0.0
                stds[mi, di] = 0.0
            else:
                means[mi, di] = float(np.mean(vals))
                stds[mi, di] = float(np.std(vals, ddof=0))
    return means, stds


def _plot_grouped_bar_with_error(
    out_path: str,
    title: str,
    ylabel: str,
    datasets: list[str],
    methods: list[str],
    samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    means, stds = _compute_mean_std(samples, datasets=datasets, methods=methods)

    n_ds = len(datasets)
    n_m = max(1, len(methods))
    fig_w = max(12, 1.1 * n_ds)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    x = np.arange(n_ds, dtype=np.float32)
    width = 0.8 / n_m

    for mi, m in enumerate(methods):
        pos = x + (mi - (n_m - 1) / 2.0) * width
        ax.bar(
            pos,
            means[mi],
            width=width * 0.90,
            color=color_map[m],
            alpha=0.72,
            edgecolor=color_map[m],
            linewidth=1.0,
            label=m if mi == 0 else None,
            zorder=2,
        )
        ax.errorbar(
            pos,
            means[mi],
            yerr=stds[mi],
            fmt="none",
            ecolor="#1F2937",
            elinewidth=1.2,
            capsize=2.8,
            capthick=1.0,
            alpha=0.95,
            zorder=3,
        )

    # Visual separators between adjacent datasets.
    for i in range(n_ds - 1):
        ax.axvline(
            x=float(i) + 0.5,
            color="#6B7280",
            linestyle="-",
            linewidth=1.2,
            alpha=0.5,
            zorder=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_xlim(-0.6, float(n_ds) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    handles = [
        plt.Line2D([0], [0], color=color_map[m], lw=6, alpha=0.7, label=m)
        for m in methods
    ]
    ax.legend(handles=handles, frameon=False, ncol=min(4, len(methods)), loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_per_run_scatter(
    out_path: str,
    title: str,
    ylabel: str,
    datasets: list[str],
    methods: list[str],
    samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    n_ds = len(datasets)
    n_m = max(1, len(methods))
    fig_w = max(12, 1.1 * n_ds)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    x = np.arange(n_ds, dtype=np.float32)
    width = 0.8 / n_m

    for mi, m in enumerate(methods):
        base_pos = x + (mi - (n_m - 1) / 2.0) * width
        for di, d in enumerate(datasets):
            vals = np.asarray(samples[d][m], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            # Small deterministic jitter so each run is visible.
            span = width * 0.28
            if vals.size == 1:
                jitter = np.array([0.0], dtype=np.float64)
            else:
                jitter = np.linspace(-span, span, vals.size, dtype=np.float64)
            px = np.full((vals.size,), float(base_pos[di]), dtype=np.float64) + jitter
            ax.scatter(
                px,
                vals,
                s=18,
                c=color_map[m],
                alpha=0.78,
                edgecolors="none",
                zorder=3,
            )

    for i in range(n_ds - 1):
        ax.axvline(
            x=float(i) + 0.5,
            color="#6B7280",
            linestyle="-",
            linewidth=1.2,
            alpha=0.5,
            zorder=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_xlim(-0.6, float(n_ds) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=color_map[m], markeredgecolor="none", alpha=0.8, label=m
        )
        for m in methods
    ]
    ax.legend(handles=handles, frameon=False, ncol=min(4, len(methods)), loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_matrix_csv(path: str, datasets: list[str], methods: list[str], samples: dict[str, dict[str, list[float]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "n", "mean", "median", "std", "min", "max"])
        for d in datasets:
            for m in methods:
                vals = np.asarray(samples[d][m], dtype=np.float64)
                if vals.size == 0:
                    continue
                w.writerow(
                    [
                        d,
                        m,
                        int(vals.size),
                        float(np.mean(vals)),
                        float(np.median(vals)),
                        float(np.std(vals, ddof=0)),
                        float(np.min(vals)),
                        float(np.max(vals)),
                    ]
                )


def _normalize_per_dataset(
    rows: list[dict[str, object]],
    *,
    datasets: list[str],
    methods: list[str],
    metric: str,
    fill_missing_zero: bool,
) -> dict[str, dict[str, list[float]]]:
    per_ds_vals: dict[str, list[float]] = {d: [] for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        if d not in per_ds_vals:
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            per_ds_vals[d].append(float(v))

    out: dict[str, dict[str, list[float]]] = {d: {m: [] for m in methods} for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        if d not in out or m not in out[d]:
            continue
        v = _to_float(r.get(metric, np.nan))
        if not np.isfinite(v):
            continue
        ds_pool = np.asarray(per_ds_vals[d], dtype=np.float64)
        if ds_pool.size == 0:
            nv = 0.0
        else:
            lo = float(np.min(ds_pool))
            hi = float(np.max(ds_pool))
            if hi <= lo + 1e-12:
                nv = 0.5
            else:
                # Keep distance-style orientation: lower is better.
                nv = (float(v) - lo) / (hi - lo)
        out[d][m].append(float(np.clip(nv, 0.0, 1.0)))

    if fill_missing_zero:
        for d in datasets:
            for m in methods:
                if len(out[d][m]) == 0:
                    out[d][m] = [0.0]
    return out


def _plot_cross_dataset_average(
    out_path: str,
    title: str,
    methods: list[str],
    datasets: list[str],
    norm_samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    # First average over runs(seeds) within each dataset, then compute
    # cross-dataset mean/std (error bar = dataset variability).
    ds_method_mean = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    for mi, m in enumerate(methods):
        for di, d in enumerate(datasets):
            vals = np.asarray(norm_samples[d][m], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            ds_method_mean[mi, di] = float(np.mean(vals)) if vals.size > 0 else 0.0

    method_mean = np.mean(ds_method_mean, axis=1)
    method_std = np.std(ds_method_mean, axis=1, ddof=0)

    x = np.arange(len(methods), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(max(7.8, 1.35 * len(methods)), 5.4))
    bar_colors = [color_map[m] for m in methods]
    ax.bar(x, method_mean, width=0.68, color=bar_colors, alpha=0.78, edgecolor=bar_colors, linewidth=1.0, zorder=2)
    ax.errorbar(
        x,
        method_mean,
        yerr=method_std,
        fmt="none",
        ecolor="#111827",
        elinewidth=1.2,
        capsize=3.0,
        capthick=1.1,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("normalized distance (lower is better)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_cross_dataset_csv(path: str, methods: list[str], datasets: list[str], norm_samples: dict[str, dict[str, list[float]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_over_datasets", "std_over_datasets", "n_datasets"])
        for m in methods:
            ds_means: list[float] = []
            for d in datasets:
                vals = np.asarray(norm_samples[d][m], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                ds_means.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
            arr = np.asarray(ds_means, dtype=np.float64)
            w.writerow([m, float(np.mean(arr)), float(np.std(arr, ddof=0)), int(arr.size)])


def main() -> None:
    p = argparse.ArgumentParser(description="Plot benchmark grouped bar charts with error bars.")
    p.add_argument("--bench", type=str, default="paper_mix_2d_3d6d_traj_vs_nontraj_7seed", help="Benchmark name (under outputs/bench) or absolute path.")
    p.add_argument("--methods", type=str, default=None, help="Comma-separated methods. Default: dataaug,oncl,vae,ecomann")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated datasets; default auto from data.")
    p.add_argument("--metrics", type=str, default="proj_manifold_dist,pred_precision, train_seconds,gt_to_learned_mean", help="Comma-separated metrics to plot.")
    p.add_argument("--analysis-dir", type=str, default="analysis_bars", help="Output dir under bench dir.")
    p.add_argument("--fill-missing-zero", action="store_true", default=True)
    p.add_argument("--no-fill-missing-zero", dest="fill_missing_zero", action="store_false")
    args = p.parse_args()

    bench_dir = _resolve_bench_dir(args.bench)
    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"benchmark directory not found: {bench_dir}")

    rows, source_name = _load_rows(bench_dir)
    if not rows:
        raise RuntimeError(f"no benchmark rows found under: {bench_dir}")

    methods = _split_csv(args.methods, DEFAULT_METHODS)
    metrics = _split_csv(args.metrics, DEFAULT_METRICS)
    all_methods_in_rows = sorted({str(r.get("method", "")) for r in rows if str(r.get("method", ""))})
    if args.methods is None:
        methods = [m for m in DEFAULT_METHODS if m in all_methods_in_rows] + [m for m in all_methods_in_rows if m not in DEFAULT_METHODS]
    else:
        methods = [m for m in methods if m in all_methods_in_rows]

    datasets = _split_csv(args.datasets, None)
    if not datasets:
        datasets = _infer_dataset_order(rows, methods)

    outdir = os.path.join(bench_dir, args.analysis_dir)
    os.makedirs(outdir, exist_ok=True)

    for metric in metrics:
        samples = _build_samples(rows, datasets, methods, metric=metric, fill_missing_zero=bool(args.fill_missing_zero))
        png_name = f"{metric}_all_datasets_bar_with_error.png"
        png_scatter_name = f"{metric}_all_datasets_per_run_scatter.png"
        csv_name = f"{metric}_all_datasets_bar_stats.csv"
        out_png = os.path.join(outdir, png_name)
        out_png_scatter = os.path.join(outdir, png_scatter_name)
        out_csv = os.path.join(outdir, csv_name)
        _plot_grouped_bar_with_error(
            out_png,
            title=f"Benchmark ({metric})",
            ylabel=metric,
            datasets=datasets,
            methods=methods,
            samples=samples,
        )
        _plot_per_run_scatter(
            out_png_scatter,
            title=f"Benchmark Per-Run Scatter ({metric})",
            ylabel=metric,
            datasets=datasets,
            methods=methods,
            samples=samples,
        )
        _save_matrix_csv(out_csv, datasets=datasets, methods=methods, samples=samples)
        print(f"[saved] {out_png}")
        print(f"[saved] {out_png_scatter}")
        print(f"[saved] {out_csv}")

        norm_samples = _normalize_per_dataset(
            rows,
            datasets=datasets,
            methods=methods,
            metric=metric,
            fill_missing_zero=bool(args.fill_missing_zero),
        )
        out_png_cross = os.path.join(outdir, f"{metric}_cross_dataset_normalized_mean_bar.png")
        out_csv_cross = os.path.join(outdir, f"{metric}_cross_dataset_normalized_mean_stats.csv")
        _plot_cross_dataset_average(
            out_png_cross,
            title=f"Cross-dataset mean (normalized, {metric})",
            methods=methods,
            datasets=datasets,
            norm_samples=norm_samples,
        )
        _save_cross_dataset_csv(
            out_csv_cross,
            methods=methods,
            datasets=datasets,
            norm_samples=norm_samples,
        )
        print(f"[saved] {out_png_cross}")
        print(f"[saved] {out_csv_cross}")

    print(f"[info] bench={bench_dir}")
    print(f"[info] source={source_name}")
    print(f"[info] methods={','.join(methods)}")
    print(f"[info] datasets={len(datasets)}")


if __name__ == "__main__":
    main()
