#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import wandb

from heatmap_utils import plot_matrix_heatmap


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


def _build_heatmap(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["baseline_w_on", "baseline_w_off", "value"])
    if df.empty:
        return None
    table = (
        df.groupby(["baseline_w_on", "baseline_w_off"])["value"]
        .mean()
        .unstack("baseline_w_off")
        .sort_index()
    )
    return table


def _plot_heatmap(table, out_path, title):
    plot_matrix_heatmap(
        table=table,
        title=title,
        out_path=out_path,
        y_label="baseline_w_on",
        x_label="baseline_w_off",
        cmap_name="YlGnBu",
        decimals=3,
        dpi=160,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="pby")
    parser.add_argument("--project", default="equality constraint learning")
    parser.add_argument("--sweep", default="3u130zxv")
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(args.entity, args.project, args.sweep)
    rows = []
    for r in runs:
        cfg = r.config or {}
        w_on = _to_float(cfg.get("baseline_w_on"))
        w_off = _to_float(cfg.get("baseline_w_off"))
        if w_on is None or w_off is None:
            continue
        summary = r.summary or {}
        val = _to_float(summary.get("avg/norm_proj_manifold_dist"))
        rows.append({"baseline_w_on": w_on, "baseline_w_off": w_off, "value": val})

    table = _build_heatmap(rows)
    if table is None:
        raise RuntimeError("No matching runs/metrics found for this sweep.")

    out_path = outdir / "heatmap_baseline_norm_proj_manifold_dist.png"
    _plot_heatmap(table, out_path, "avg/norm_proj_manifold_dist")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
