#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import wandb

from heatmap_utils import plot_matrix_heatmap


def _to_float(val):
    try:
        return float(val)
    except Exception:
        return None


def _collect_runs(entity, project, limit):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=limit, order="-created_at")
    return runs[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="pby")
    parser.add_argument("--project", default="equality constraint learning")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "avg/proj_manifold_dist",
            "avg/proj_true_dist",
            "avg/pred_recall",
            "avg/pred_precision",
            "avg/pred_manifold_dist",
        ],
    )
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(args.entity, args.project, args.limit)
    rows = []
    for r in runs:
        summary = r.summary or {}
        row = {"run_id": r.id}
        for m in args.metrics:
            row[m] = _to_float(summary.get(m))
        rows.append(row)

    if not rows:
        raise RuntimeError("No runs found for this project.")

    df = pd.DataFrame(rows).set_index("run_id")
    df = df.dropna(axis=1, how="all")
    if df.empty:
        raise RuntimeError("No metrics found for correlation.")
    corr = df.corr(min_periods=5)
    corr_path = outdir / "metric_correlation.csv"
    corr.to_csv(corr_path)

    heat_path = outdir / "metric_correlation_heatmap.png"
    plot_matrix_heatmap(
        table=corr,
        title="Metric Correlation",
        out_path=heat_path,
        y_label="metric",
        x_label="metric",
        cmap_name="YlGnBu",
        decimals=3,
        dpi=160,
    )

    print(f"saved: {corr_path}")
    print(f"saved: {heat_path}")


if __name__ == "__main__":
    main()
