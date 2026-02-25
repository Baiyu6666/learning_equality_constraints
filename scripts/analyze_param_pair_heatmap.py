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


def _plot_heatmap(table, title, out_path, y_label, x_label):
    plot_matrix_heatmap(
        table=table,
        title=title,
        out_path=out_path,
        y_label=y_label,
        x_label=x_label,
        cmap_name="YlGnBu",
        decimals=3,
        dpi=160,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="pby")
    parser.add_argument("--project", default="equality constraint learning")
    parser.add_argument("--sweep", default="8mx33kux")
    parser.add_argument("--param_x", default="adp_sigma_danger_scale")
    parser.add_argument("--param_y", default="adp_sigma_r_max")
    parser.add_argument("--metric", default="proj_manifold_dist")
    parser.add_argument("--include_avg", action="store_true", default=True)
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(args.entity, args.project, args.sweep)
    rows = []
    datasets = set()
    for r in runs:
        cfg = r.config or {}
        x_val = _to_float(cfg.get(args.param_x))
        y_val = _to_float(cfg.get(args.param_y))
        if x_val is None or y_val is None:
            continue
        sig = _param_signature(cfg, {"seed", "method", "method_tag", "_wandb"})
        summary = r.summary or {}
        for key, val in summary.items():
            if not isinstance(key, str) or "/" not in key:
                continue
            ds, metric_name = key.split("/", 1)
            if metric_name != args.metric:
                continue
            datasets.add(ds)
            rows.append(
                {
                    "param_x": x_val,
                    "param_y": y_val,
                    "value": _to_float(val),
                    "sig": sig,
                    "dataset": ds,
                }
            )
        if args.include_avg:
            if args.metric == "proj_manifold_dist":
                print('Warning!Using local norm parameter')
                norm_vals = []
                for ds, stats in NORM_PROJ_BASELINE.items():
                    raw = _to_float(summary.get(f"{ds}/proj_manifold_dist"))
                    if raw is None or stats["std"] <= 0:
                        continue
                    norm_vals.append((raw - stats["mean"]) / stats["std"])
                if norm_vals:
                    rows.append(
                        {
                            "param_x": x_val,
                            "param_y": y_val,
                            "value": float(np.mean(norm_vals)),
                            "sig": sig,
                            "dataset": "avg",
                        }
                    )
                    datasets.add("avg")
            elif args.metric == "avg/local_norm_proj_manifold_dist":
                norm_vals = []
                for ds, stats in NORM_PROJ_BASELINE.items():
                    raw = _to_float(summary.get(f"{ds}/proj_manifold_dist"))
                    if raw is None or stats["std"] <= 0:
                        continue
                    norm_vals.append((raw - stats["mean"]) / stats["std"])
                if norm_vals:
                    rows.append(
                        {
                            "param_x": x_val,
                            "param_y": y_val,
                            "value": float(np.mean(norm_vals)),
                            "sig": sig,
                            "dataset": "avg",
                        }
                    )
                    datasets.add("avg")
            else:
                avg_key = f"avg/{args.metric}"
                metric = summary.get(avg_key)
                if metric is not None:
                    rows.append(
                        {
                            "param_x": x_val,
                            "param_y": y_val,
                            "value": _to_float(metric),
                            "sig": sig,
                            "dataset": "avg",
                        }
                    )
                    datasets.add("avg")

    if not rows:
        raise RuntimeError("No matching runs/metrics found for this sweep.")

    import pandas as pd

    df = pd.DataFrame(rows).dropna(subset=["param_x", "param_y", "value", "dataset"])
    if df.empty:
        raise RuntimeError("No valid data to plot.")
    df = df.groupby(["dataset", "param_x", "param_y", "sig"])["value"].mean().reset_index()
    df = df.groupby(["dataset", "param_x", "param_y"])["value"].mean().reset_index()

    saved_paths = []
    for ds in sorted(datasets):
        sub = df[df["dataset"] == ds]
        table = _heatmap_table(
            sub.to_dict(orient="records"), "param_y", "param_x", "value"
        )
        if table is None:
            continue
        name = "avg" if ds == "avg" else ds
        out_path = outdir / f"heatmap_{name}_{args.param_x}_vs_{args.param_y}.png"
        _plot_heatmap(
            table,
            f"{name}/{args.metric} by {args.param_y} vs {args.param_x}",
            out_path,
            args.param_y,
            args.param_x,
        )
        saved_paths.append(out_path)
    for p in saved_paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
