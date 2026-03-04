#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build dataset correlation/covariance matrices using proj_manifold_dist across runs."
    )
    p.add_argument("--input-csv", default="per_case_metrics.csv", help="Path to per_case_metrics.csv")
    p.add_argument(
        "--outdir",
        default="corr_cov_proj_manifold_dist",
        help="Output directory for matrices and heatmaps",
    )
    return p.parse_args()


def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def _pair_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    sx = float(xx.std(ddof=1))
    sy = float(yy.std(ddof=1))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    c = float(np.corrcoef(xx, yy)[0, 1])
    return c


def _pair_cov(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    return float(np.cov(xx, yy, ddof=1)[0, 1])


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * t)),
        int(round(c1[1] + (c2[1] - c1[1]) * t)),
        int(round(c1[2] + (c2[2] - c1[2]) * t)),
    )


def _corr_color(v: float) -> tuple[int, int, int]:
    neg = _hex_to_rgb("#3B4CC0")
    mid = _hex_to_rgb("#F7F7F7")
    pos = _hex_to_rgb("#B40426")
    if not math.isfinite(v):
        return _hex_to_rgb("#D9D9D9")
    vv = max(-1.0, min(1.0, v))
    if vv < 0:
        return _lerp_color(mid, neg, -vv)
    return _lerp_color(mid, pos, vv)


def _cov_color(v: float, vmax_abs: float) -> tuple[int, int, int]:
    low = _hex_to_rgb("#F7FBFF")
    high = _hex_to_rgb("#084594")
    if not math.isfinite(v):
        return _hex_to_rgb("#D9D9D9")
    if vmax_abs <= 0:
        return low
    t = max(0.0, min(1.0, abs(v) / vmax_abs))
    return _lerp_color(low, high, t)


def _draw_heatmap(
    mat: np.ndarray,
    labels: list[str],
    title: str,
    path: str,
    *,
    mode: str,
) -> None:
    n = len(labels)
    cell = 58
    left = 260
    top = 120
    right = 50
    bottom = 60
    width = left + n * cell + right
    height = top + n * cell + bottom
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    try:
        f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        f_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        f_val = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except Exception:
        f_title = ImageFont.load_default()
        f_label = ImageFont.load_default()
        f_val = ImageFont.load_default()

    draw.text((22, 24), title, fill="#111111", font=f_title)

    vmax_abs = float(np.nanmax(np.abs(mat))) if np.any(np.isfinite(mat)) else 1.0
    for i in range(n):
        for j in range(n):
            v = float(mat[i, j])
            if mode == "corr":
                c = _corr_color(v)
            else:
                c = _cov_color(v, vmax_abs=vmax_abs)
            x0 = left + j * cell
            y0 = top + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=c, outline="#E0E0E0", width=1)
            txt = "nan" if not math.isfinite(v) else f"{v:.2f}"
            tw = draw.textlength(txt, font=f_val)
            draw.text((x0 + (cell - tw) / 2.0, y0 + 20), txt, fill="#111111", font=f_val)

    for i, lab in enumerate(labels):
        y = top + i * cell + 20
        draw.text((20, y), lab, fill="#2A2A2A", font=f_label)
    for j, lab in enumerate(labels):
        x = left + j * cell + 3
        draw.text((x, top - 22), lab[:8], fill="#2A2A2A", font=f_label)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


def _save_matrix_csv(path: str, labels: list[str], mat: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", *labels])
        for i, lab in enumerate(labels):
            row = [lab]
            for j in range(len(labels)):
                v = float(mat[i, j])
                row.append("" if not math.isfinite(v) else f"{v:.10g}")
            w.writerow(row)


def main() -> None:
    args = _parse_args()
    rows = list(csv.DictReader(open(args.input_csv, "r", encoding="utf-8")))

    # A "run" is (method, seed). Each run should have one value per dataset.
    by_run_ds: dict[str, dict[str, float]] = defaultdict(dict)
    datasets_set: set[str] = set()
    for r in rows:
        method = str(r.get("method", "")).strip()
        seed = str(r.get("seed", "")).strip()
        dataset = str(r.get("dataset", "")).strip()
        if not method or not seed or not dataset:
            continue
        run_id = f"{method}__seed{seed}"
        v = _to_float(str(r.get("proj_manifold_dist", "")))
        by_run_ds[run_id][dataset] = v
        datasets_set.add(dataset)

    datasets = sorted(datasets_set)
    runs = sorted(by_run_ds.keys())
    X = np.full((len(runs), len(datasets)), np.nan, dtype=np.float64)
    for i, run in enumerate(runs):
        for j, ds in enumerate(datasets):
            X[i, j] = by_run_ds[run].get(ds, float("nan"))

    n = len(datasets)
    corr = np.full((n, n), np.nan, dtype=np.float64)
    cov = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        xi = X[:, i]
        for j in range(n):
            xj = X[:, j]
            corr[i, j] = _pair_corr(xi, xj)
            cov[i, j] = _pair_cov(xi, xj)

    os.makedirs(args.outdir, exist_ok=True)
    run_matrix_csv = os.path.join(args.outdir, "run_by_dataset_proj_manifold_dist.csv")
    with open(run_matrix_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", *datasets])
        for i, run in enumerate(runs):
            row = [run]
            for j in range(len(datasets)):
                v = float(X[i, j])
                row.append("" if not math.isfinite(v) else f"{v:.10g}")
            w.writerow(row)

    corr_csv = os.path.join(args.outdir, "dataset_corr_proj_manifold_dist.csv")
    cov_csv = os.path.join(args.outdir, "dataset_cov_proj_manifold_dist.csv")
    _save_matrix_csv(corr_csv, datasets, corr)
    _save_matrix_csv(cov_csv, datasets, cov)

    corr_png = os.path.join(args.outdir, "dataset_corr_proj_manifold_dist.png")
    cov_png = os.path.join(args.outdir, "dataset_cov_proj_manifold_dist.png")
    _draw_heatmap(corr, datasets, "Dataset Correlation (proj_manifold_dist)", corr_png, mode="corr")
    _draw_heatmap(cov, datasets, "Dataset Covariance (proj_manifold_dist)", cov_png, mode="cov")

    print(f"[saved] {run_matrix_csv}")
    print(f"[saved] {corr_csv}")
    print(f"[saved] {cov_csv}")
    print(f"[saved] {corr_png}")
    print(f"[saved] {cov_png}")
    print(f"[info] runs={len(runs)}, datasets={len(datasets)}")


if __name__ == "__main__":
    main()
