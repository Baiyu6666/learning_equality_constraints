#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path

import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont

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


def _find_image_file(run, dataset):
    key = f"{dataset}/data_contour"
    summary = run.summary or {}
    img_info = summary.get(key)
    if isinstance(img_info, dict):
        path = img_info.get("path")
        if path:
            try:
                return run.file(path)
            except Exception:
                pass
    want = f"media/images/{dataset}/data_contour"
    for f in run.files():
        name = f.name
        if want in name and name.endswith(".png"):
            return f
    return None


def _norm_proj_from_summary(summary, dataset):
    key = f"{dataset}/norm_proj_manifold_dist"
    val = _to_float(summary.get(key))
    if val is not None:
        return val
    base = NORM_PROJ_BASELINE.get(dataset)
    if not base or base["std"] <= 0:
        return None
    raw = _to_float(summary.get(f"{dataset}/proj_manifold_dist"))
    if raw is None:
        return None
    return (raw - base["mean"]) / base["std"]


def _best_run(runs):
    best = None
    best_val = None
    for r in runs:
        val = _to_float(r.summary.get("avg/norm_proj_manifold_dist"))
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best = r
    return best, best_val


def _same_hparams(run, candidates, keys):
    cfg = run.config or {}
    target = {k: _to_float(cfg.get(k)) for k in keys}
    out = []
    for r in candidates:
        rcfg = r.config or {}
        match = True
        for k, v in target.items():
            rv = _to_float(rcfg.get(k))
            if rv is None or v is None or abs(rv - v) > 1e-9:
                match = False
                break
        if match:
            out.append(r)
    return out, target


def _make_grid(images, labels, out_path, cell_size=(320, 240)):
    cols = len(images[0])
    rows = len(images)
    w, h = cell_size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * w, r * h
            img = images[r][c]
            if img is not None:
                img = img.resize((w, h))
                grid.paste(img, (x0, y0))
            label = labels[r][c]
            if label:
                draw.rectangle([x0, y0, x0 + w, y0 + 18], fill=(255, 255, 255))
                draw.text((x0 + 4, y0 + 2), label, fill=(0, 0, 0), font=font)
    grid.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="pby")
    parser.add_argument("--project", default="equality constraint learning")
    parser.add_argument("--energy_sweep", default="jarzgdrw")
    parser.add_argument("--baseline_sweep", default="3u130zxv")
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    parser.add_argument("--seed_count", type=int, default=5)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "figure_eight",
            "ellipse",
            "noise_only",
            "sparse_only",
            "hetero_noise",
            "looped_spiro",
        ],
    )
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)

    energy_runs = _collect_runs(args.entity, args.project, args.energy_sweep)
    baseline_runs = _collect_runs(args.entity, args.project, args.baseline_sweep)

    best_energy, best_energy_val = _best_run(energy_runs)
    best_base, best_base_val = _best_run(baseline_runs)
    if best_energy is None or best_base is None:
        raise RuntimeError("could not find best runs for energy/baseline")

    energy_keys = ["lam_dist", "lam_dir", "lam_pl", "lam_denoise"]
    base_keys = ["baseline_w_on", "baseline_w_off"]
    energy_group, energy_params = _same_hparams(best_energy, energy_runs, energy_keys)
    base_group, base_params = _same_hparams(best_base, baseline_runs, base_keys)

    rng = random.Random(7)
    rng.shuffle(energy_group)
    rng.shuffle(base_group)
    energy_group = energy_group[: args.seed_count]
    base_group = base_group[: args.seed_count]

    # Summary table
    summary_path = outdir / "best_norm_comparison.csv"
    avg_keys = [
        "avg/on_mean_v",
        "avg/proj_manifold_dist",
        "avg/proj_v_residual",
        "avg/proj_true_dist",
        "avg/corr_v_d2",
        "avg/slope_v_d2",
        "avg/proj_steps",
        "avg/pred_recall",
        "avg/FPrate",
        "avg/norm_proj_manifold_dist",
    ]
    with summary_path.open("w", encoding="utf-8") as f:
        header = ["dataset", "metric", "energy_value", "baseline_value"]
        f.write(",".join(header) + "\n")
        for k in avg_keys:
            e_val = _to_float(best_energy.summary.get(k))
            b_val = _to_float(best_base.summary.get(k))
            f.write(f"avg,{k},{e_val},{b_val}\n")
        for ds in args.datasets:
            e_val = _norm_proj_from_summary(best_energy.summary, ds)
            b_val = _norm_proj_from_summary(best_base.summary, ds)
            f.write(f"{ds},norm_proj_manifold_dist,{e_val},{b_val}\n")

    # Contour grids
    saved_grids = []
    for ds in args.datasets:
        images = [[], []]
        labels = [[], []]
        for r in energy_group:
            f_img = _find_image_file(r, ds)
            img = None
            if f_img is not None:
                local = f_img.download(root=str(outdir), replace=True).name
                try:
                    img = Image.open(local).convert("RGB")
                except Exception:
                    img = None
            images[0].append(img)
            labels[0].append(f"energy s={r.config.get('seed')}")
        for r in base_group:
            f_img = _find_image_file(r, ds)
            img = None
            if f_img is not None:
                local = f_img.download(root=str(outdir), replace=True).name
                try:
                    img = Image.open(local).convert("RGB")
                except Exception:
                    img = None
            images[1].append(img)
            labels[1].append(f"delta s={r.config.get('seed')}")
        out_path = outdir / f"best_contours_{ds}.png"
        _make_grid(images, labels, out_path)
        saved_grids.append(out_path)

    print("best energy params:", energy_params, "avg/norm", best_energy_val)
    print("best baseline params:", base_params, "avg/norm", best_base_val)
    print(f"saved summary: {summary_path}")
    for p in saved_grids:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
