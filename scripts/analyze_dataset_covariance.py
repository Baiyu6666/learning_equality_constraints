#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
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
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "figure_eight",
            "ellipse",
            "noise_only",
            "sparse_only",
            # "hetero_noise",
            "looped_spiro",
        ],
    )
    parser.add_argument("--metric", default="proj_manifold_dist")
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(args.entity, args.project, args.limit)
    rows = []
    for r in runs:
        summary = r.summary or {}
        row = {"run_id": r.id}
        for ds in args.datasets:
            key = f"{ds}/pred_manifold_dist"
            val = _to_float(summary.get(key))
            row[ds] = val
        rows.append(row)

    if not rows:
        raise RuntimeError("No runs found for this project.")

    df = pd.DataFrame(rows).set_index("run_id")
    df = df.dropna(axis=1, how="all")
    if df.empty:
        raise RuntimeError("No runs with any dataset metrics found.")
    cov = df.cov(min_periods=5)
    corr = df.corr(min_periods=5)
    order = corr.mean(axis=0).sort_values(ascending=False).index.tolist()
    cov = cov.loc[order, order]
    corr = corr.loc[order, order]
    stats = df.agg(["mean", "std"]).T
    stats = stats.loc[order]

    cov_path = outdir / "dataset_covariance.csv"
    corr_path = outdir / "dataset_correlation.csv"
    stats_path = outdir / "dataset_metric_stats.csv"
    cov.to_csv(cov_path)
    corr.to_csv(corr_path)
    stats.to_csv(stats_path)

    def plot_heatmap(mat, title, path):
        plot_matrix_heatmap(
            table=mat,
            title=title,
            out_path=path,
            y_label="dataset",
            x_label="dataset",
            cmap_name="YlGnBu",
            decimals=3,
            dpi=160,
        )

    plot_heatmap(cov, "Dataset Covariance", outdir / "dataset_covariance_heatmap.png")
    plot_heatmap(corr, "Dataset Correlation", outdir / "dataset_correlation_heatmap.png")

    print(f"saved: {cov_path}")
    print(f"saved: {corr_path}")
    print(f"saved: {stats_path}")
    print("dataset metric stats (mean/std):")
    print(stats)


if __name__ == "__main__":
    main()
