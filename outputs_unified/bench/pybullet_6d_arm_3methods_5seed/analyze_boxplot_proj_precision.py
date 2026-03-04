#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot method boxplots across datasets for proj_manifold_dist and pred_precision."
    )
    p.add_argument(
        "--input-csv",
        default="per_case_metrics.csv",
        help="Path to per_case_metrics.csv (default: per_case_metrics.csv in cwd).",
    )
    p.add_argument(
        "--output-png",
        default="boxplot_proj_manifold_dist_pred_precision.png",
        help="(Deprecated) single output plot path. Kept for compatibility.",
    )
    p.add_argument(
        "--output-dir",
        default="boxplots_by_dataset",
        help="Directory to save one boxplot PNG per dataset.",
    )
    p.add_argument(
        "--output-csv",
        default="dataset_method_seedmean_metrics.csv",
        help="Output aggregated CSV path.",
    )
    return p.parse_args()


def _to_float(value: str) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x


def _seed_mean_drop_nan(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return 0.0
    if all(abs(v) == 0.0 for v in finite):
        return 0.0
    return float(sum(finite) / len(finite))


def _load_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aggregate_seed_means(rows: list[dict[str, str]], metrics: list[str]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        dataset = str(r.get("dataset", "")).strip()
        method = str(r.get("method", "")).strip()
        if not dataset or not method:
            continue
        for metric in metrics:
            grouped[(dataset, method)][metric].append(_to_float(str(r.get(metric, ""))))

    out: list[dict[str, str | float]] = []
    for (dataset, method), metric_map in sorted(grouped.items()):
        row: dict[str, str | float] = {"dataset": dataset, "method": method}
        for metric in metrics:
            row[metric] = _seed_mean_drop_nan(metric_map.get(metric, []))
        out.append(row)
    return out


def _group_by_dataset_method(
    rows: list[dict[str, str]], metrics: list[str]
) -> dict[str, dict[str, dict[str, list[float]]]]:
    out: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        dataset = str(r.get("dataset", "")).strip()
        method = str(r.get("method", "")).strip()
        if not dataset or not method:
            continue
        for metric in metrics:
            out[dataset][method][metric].append(_to_float(str(r.get(metric, ""))))
    return out


def _write_agg_csv(path: str, rows: list[dict[str, str | float]], metrics: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", *metrics])
        for r in rows:
            w.writerow([r["dataset"], r["method"], *[r[m] for m in metrics]])


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _box_stats(values: list[float]) -> dict[str, float]:
    vals = sorted(values) if values else [0.0]
    q1 = _quantile(vals, 0.25)
    q2 = _quantile(vals, 0.50)
    q3 = _quantile(vals, 0.75)
    iqr = q3 - q1
    lo_bound = q1 - 1.5 * iqr
    hi_bound = q3 + 1.5 * iqr
    inlier = [v for v in vals if lo_bound <= v <= hi_bound]
    whisker_lo = min(inlier) if inlier else q1
    whisker_hi = max(inlier) if inlier else q3
    return {"q1": q1, "median": q2, "q3": q3, "wlo": whisker_lo, "whi": whisker_hi}


def _plot_dataset_boxplot(
    path: str,
    dataset: str,
    methods: list[str],
    metric_to_values: dict[str, dict[str, list[float]]],
    metrics: list[str],
) -> None:
    panel_w, panel_h = 760, 560
    margin_l, margin_r, margin_t, margin_b = 86, 36, 74, 78
    gap = 26
    width = panel_w * len(metrics) + gap * (len(metrics) - 1)
    height = panel_h
    img = Image.new("RGBA", (width, height), "#EAEAF2")
    draw = ImageDraw.Draw(img)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    method_color = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        font_metric = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_tick = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font_title = ImageFont.load_default()
        font_metric = ImageFont.load_default()
        font_tick = ImageFont.load_default()
        font_label = ImageFont.load_default()

    for midx, metric in enumerate(metrics):
        x0 = midx * (panel_w + gap)
        y_min = margin_t
        y_max = panel_h - margin_b
        x_min = x0 + margin_l
        x_max = x0 + panel_w - margin_r
        draw.rectangle([x_min, y_min, x_max, y_max], outline="#C9CEDA", width=1, fill="#FFFFFF")
        draw.text((x_min + 8, 32), metric, fill="#2F3640", font=font_metric)

        data_by_method: list[list[float]] = []
        for m in methods:
            raw_vals = metric_to_values.get(metric, {}).get(m, [])
            vals = [v for v in raw_vals if not math.isnan(v)]
            if not vals:
                vals = [0.0]
            elif all(abs(v) == 0.0 for v in vals):
                vals = [0.0]
            data_by_method.append(vals)
        all_vals = [v for vals in data_by_method for v in vals]
        vmin = min(all_vals) if all_vals else 0.0
        vmax = max(all_vals) if all_vals else 1.0
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        pad = 0.08 * (vmax - vmin)
        vmin -= pad
        vmax += pad

        def y_map(v: float) -> int:
            t = (v - vmin) / (vmax - vmin)
            return int(y_max - t * (y_max - y_min))

        # y ticks
        for i in range(6):
            tv = vmin + (vmax - vmin) * i / 5.0
            yy = y_map(tv)
            draw.line([(x_min + 1, yy), (x_max - 1, yy)], fill="#E5E7EF", width=1)
            draw.text((x0 + 8, yy - 8), f"{tv:.3f}", fill="#6B7280", font=font_tick)

        n = max(len(methods), 1)
        box_span = (x_max - x_min) / n
        for i, (method, vals) in enumerate(zip(methods, data_by_method)):
            stats = _box_stats(vals)
            cx = int(x_min + (i + 0.5) * box_span)
            bw = int(min(46, box_span * 0.46))
            q1y, q3y = y_map(stats["q1"]), y_map(stats["q3"])
            medy = y_map(stats["median"])
            wloy, whiy = y_map(stats["wlo"]), y_map(stats["whi"])
            c = method_color[method]

            fill = c + "66"
            draw.line([(cx, whiy), (cx, q3y)], fill=c, width=2)
            draw.line([(cx, q1y), (cx, wloy)], fill=c, width=2)
            draw.line([(cx - bw // 3, whiy), (cx + bw // 3, whiy)], fill=c, width=2)
            draw.line([(cx - bw // 3, wloy), (cx + bw // 3, wloy)], fill=c, width=2)
            draw.rectangle([cx - bw, q3y, cx + bw, q1y], outline=c, width=2, fill=fill)
            draw.line([(cx - bw, medy), (cx + bw, medy)], fill="#2E3440", width=2)

            # Overlay individual seed points with deterministic horizontal jitter.
            for j, v in enumerate(vals):
                xx = cx + int((j - (len(vals) - 1) / 2.0) * 8)
                yy = y_map(v)
                r = 2
                draw.ellipse([xx - r, yy - r, xx + r, yy + r], fill=c, outline=c, width=1)

            draw.text((cx - int(len(method) * 3.8), y_max + 14), method, fill="#374151", font=font_label)

    draw.text((12, 10), f"{dataset}", fill="#1F2937", font=font_title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.convert("RGB").save(path)


def main() -> None:
    args = _parse_args()
    metrics = ["proj_manifold_dist", "pred_precision"]
    rows = _load_rows(args.input_csv)
    by_ds_method = _group_by_dataset_method(rows, metrics)
    methods = sorted({str(r.get("method", "")).strip() for r in rows if str(r.get("method", "")).strip()})

    # Save per-dataset-per-method seed means (drop NaN; if empty then 0).
    agg_rows = _aggregate_seed_means(rows, metrics)
    _write_agg_csv(args.output_csv, agg_rows, metrics)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for dataset in sorted(by_ds_method.keys()):
        metric_to_values: dict[str, dict[str, list[float]]] = {m: {} for m in metrics}
        for method in methods:
            for metric in metrics:
                metric_to_values[metric][method] = by_ds_method[dataset][method][metric]
        out_png = os.path.join(out_dir, f"{dataset}__boxplot_proj_precision.png")
        _plot_dataset_boxplot(out_png, dataset, methods, metric_to_values, metrics)
        saved += 1

    print(f"[saved] {args.output_csv}")
    print(f"[saved] {saved} dataset boxplots under: {out_dir}")


if __name__ == "__main__":
    main()
