#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import wandb

from heatmap_utils import plot_matrix_heatmap


NORM_PROJ_BASELINE = {
    "figure_eight": {"mean": 0.120565, "std": 0.107640},
    "ellipse": {"mean": 0.081101, "std": 0.125372},
    "noise_only": {"mean": 0.459975, "std": 0.225468},
    "sparse_only": {"mean": 0.264254, "std": 0.239843},
    "looped_spiro": {"mean": 0.190081, "std": 0.200199},
}


def _to_float(val):
    try:
        return float(val)
    except Exception:
        return None


def _collect_runs(entity, project, sweep_id):
    api = wandb.Api()
    try:
        return api.runs(f"{entity}/{project}", filters={"sweep": sweep_id})
    except Exception:
        return [
            r for r in api.runs(f"{entity}/{project}") if r.sweep and r.sweep.id == sweep_id
        ]


def _param_signature(cfg, drop_keys):
    items = []
    for k, v in cfg.items():
        if k in drop_keys:
            continue
        if isinstance(v, (int, float, bool, str)):
            items.append((k, v))
    return tuple(sorted(items))


def _heatmap_table(rows, row_key, col_key, val_key):
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.dropna(subset=[row_key, col_key, val_key])
    if df.empty:
        return None
    table = (
        df.groupby([row_key, col_key])[val_key]
        .mean()
        .unstack(col_key)
        .sort_index()
    )
    return table


def _plot_heatmap(table, title, out_path, y_label, x_label, col_order=None):
    plot_matrix_heatmap(
        table=table,
        title=title,
        out_path=out_path,
        y_label=y_label,
        x_label=x_label,
        col_order=col_order,
        cmap_name="YlGnBu",
        decimals=3,
        dpi=160,
    )


def _metric_key(dataset, metric):
    if "/" in metric:
        return metric
    return f"{dataset}/{metric}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="pby")
    parser.add_argument("--project", default="equality constraint learning")
    parser.add_argument("--sweep", default="7pw0fek3")
    parser.add_argument("--param", default="off_curriculum_epochs")
    parser.add_argument("--metric", default="proj_true_dist")
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    parser.add_argument("--normalize_baseline", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "figure_eight",
            "ellipse",
            "noise_only",
            "sparse_only",
            "looped_spiro",
        ],
    )
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(args.entity, args.project, args.sweep)
    rows = []
    for r in runs:
        cfg = r.config or {}
        param_val = _to_float(cfg.get(args.param))
        if param_val is None:
            continue
        sig = _param_signature(cfg, {"seed", "method", "method_tag", "_wandb"})
        summary = r.summary or {}
        for ds in args.datasets:
            key = _metric_key(ds, args.metric)
            rows.append(
                {
                    "param": param_val,
                    "dataset": ds,
                    "value": _to_float(summary.get(key)),
                    "sig": sig,
                }
            )

    if not rows:
        raise RuntimeError("No matching runs/metrics found for this sweep.")

    import pandas as pd

    df = pd.DataFrame(rows).dropna(subset=["param", "dataset", "value"])
    if df.empty:
        raise RuntimeError("No valid data to plot.")
    df = df.groupby(["param", "dataset", "sig"])["value"].mean().reset_index()
    df = df.groupby(["param", "dataset"])["value"].mean().reset_index()

    if args.no_normalize:
        args.normalize_baseline = False

    if args.normalize_baseline:
        normed = []
        for _, row in df.iterrows():
            stats = NORM_PROJ_BASELINE.get(row["dataset"])
            if stats is None or stats["std"] <= 0:
                continue
            normed.append(
                {
                    "param": row["param"],
                    "dataset": row["dataset"],
                    "value": (row["value"] - stats["mean"]) / stats["std"],
                }
            )
        df = pd.DataFrame(normed)
        if df.empty:
            raise RuntimeError("No valid data after baseline normalization.")

    # Append an avg column: mean across datasets for each parameter value.
    avg_df = df.groupby("param")["value"].mean().reset_index()
    avg_df["dataset"] = "avg"
    df = pd.concat([df, avg_df[["param", "dataset", "value"]]], ignore_index=True)

    table = _heatmap_table(
        df.to_dict(orient="records"), "param", "dataset", "value"
    )
    if table is None:
        raise RuntimeError("No valid data to plot.")
    title = f"{args.metric} by dataset vs {args.param}"
    if args.normalize_baseline:
        title = f"normalized {title}"
    out_path = outdir / f"heatmap_{args.param}_by_dataset.png"
    _plot_heatmap(
        table,
        title,
        out_path,
        args.param,
        "dataset",
        col_order=list(args.datasets) + ["avg"],
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
