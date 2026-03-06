#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import wandb

from methods.baseline_udf.scripts.heatmap_utils import plot_matrix_heatmap

DATASET_ALIASES = {
    "noise_on": "2d_noisy_sine",
    "noise-only": "2d_noisy_sine",
    "sparse-on": "2d_sparse_sine",
}


def _to_float(val):
    try:
        return float(val)
    except Exception:
        return None


def _collect_runs(entity, project, group):
    api = wandb.Api()
    try:
        return api.runs(
            f"{entity}/{project}",
            filters={"group": group},
        )
    except Exception:
        return [
            r for r in api.runs(f"{entity}/{project}") if r.group == group
        ]


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


def _select_tag(tags, prefer_prefix=None):
    if not tags:
        return None
    if prefer_prefix:
        for t in tags:
            if str(t).startswith(prefer_prefix):
                return str(t)
    return str(tags[0])


def _canon_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="pby")
    parser.add_argument("--project", default="equality constraint learning")
    parser.add_argument("--group", default="compare1")
    parser.add_argument(
        "--metric",
        default="proj_true_dist",
        help="Metric name under each dataset key, e.g. proj_manifold_dist",
    )
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    parser.add_argument("--tag_prefix", default=None)
    parser.add_argument(
        "--avg_mode",
        choices=["none", "all"],
        default="all",
        help="Whether to append an avg column.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "2d_figure_eight",
            "2d_ellipse",
            "2d_noisy_sine",
            "2d_sparse_sine",
            "2d_looped_spiro",
        ],
    )
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(args.entity, args.project, args.group)
    wanted = {_canon_dataset(ds) for ds in args.datasets}
    rows = []
    for r in runs:
        tag = _select_tag(r.tags or [], args.tag_prefix)
        if tag is None:
            continue
        summary = r.summary or {}
        for key, val in summary.items():
            if not isinstance(key, str) or "/" not in key:
                continue
            ds, metric = key.split("/", 1)
            if metric != args.metric:
                continue
            ds = _canon_dataset(ds)
            if ds not in wanted:
                continue
            rows.append(
                {
                    "method": tag,
                    "dataset": ds,
                    "value": _to_float(val),
                }
            )

    if not rows:
        raise RuntimeError("No matching runs/metrics found for this group.")

    import pandas as pd

    df = pd.DataFrame(rows).dropna(subset=["method", "dataset", "value"])
    if df.empty:
        raise RuntimeError("No matching runs/metrics found for this group.")
    data_rows = []
    for _, row in df.iterrows():
        data_rows.append(
            {
                "method": row["method"],
                "dataset": row["dataset"],
                "value": row["value"],
            }
        )
    avg_col = None
    if args.avg_mode == "all":
        avg_rows = list(data_rows)
        if avg_rows:
            avg_df = pd.DataFrame(avg_rows)
            avg_vals = avg_df.groupby("method")["value"].mean().reset_index()
            avg_col = "avg"
            for _, row in avg_vals.iterrows():
                data_rows.append(
                    {
                        "method": row["method"],
                        "dataset": avg_col,
                        "value": row["value"],
                    }
                )
    table = _heatmap_table(data_rows, "method", "dataset", "value")
    if table is None:
        raise RuntimeError("No valid data to plot.")
    out_path = outdir / f"heatmap_compare_group_{args.metric}.png"
    _plot_heatmap(
        table,
        f"{args.metric} by dataset vs method tag",
        out_path,
        "method tag",
        "dataset",
        col_order=list(
            dict.fromkeys([_canon_dataset(ds) for ds in args.datasets])
        ) if avg_col is None else list(
            dict.fromkeys([_canon_dataset(ds) for ds in args.datasets] + [avg_col])
        ),
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
