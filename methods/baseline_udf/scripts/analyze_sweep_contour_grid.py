#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import tempfile
from pathlib import Path

import wandb
from PIL import Image, ImageDraw, ImageFont


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


def _match_run_by_param_seed(runs, param_key, param_value, seed, tol=1e-6):
    target_num = _to_float(param_value)
    for r in runs:
        cfg = r.config or {}
        raw_val = cfg.get(param_key)
        # Handle values stored as scalar or singleton list/tuple.
        if isinstance(raw_val, (list, tuple)) and len(raw_val) == 1:
            raw_val = raw_val[0]
        val_num = _to_float(raw_val)
        rseed = cfg.get("seed")
        if rseed is None:
            continue
        if int(rseed) != int(seed):
            continue
        if target_num is not None and val_num is not None:
            if abs(val_num - target_num) <= tol:
                return r
            continue
        if str(raw_val) == str(param_value):
            return r
    return None


def _make_grid(images, labels, out_path, cell_size=(520, 400)):
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
    parser.add_argument("--sweep", default="1aatrlc4")
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
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
    parser.add_argument(
        "--param-key",
        default='sigmas',
        help="Config key used to group runs, e.g. sigmas or lam_smooth",
    )
    parser.add_argument(
        "--param-values",
        nargs="+",
        type=str,
        default=["0.1", "0.2", "0.4", "0.8", "1.5"],
        help="Values of --param-key to plot (as strings; numeric compare is supported)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47]
    )
    args = parser.parse_args()

    outdir = Path(args.outdir) / Path(__file__).stem
    outdir.mkdir(parents=True, exist_ok=True)
    runs = _collect_runs(args.entity, args.project, args.sweep)

    param_values = args.param_values

    with tempfile.TemporaryDirectory() as tmpdir:
        for ds in args.datasets:
            images = []
            labels = []
            for pv in param_values:
                row_imgs = []
                row_labels = []
                for seed in args.seeds:
                    run = _match_run_by_param_seed(
                        runs, args.param_key, pv, seed
                    )
                    if run is None:
                        row_imgs.append(None)
                        row_labels.append(
                            f"{args.param_key}={pv} seed={seed} (missing)"
                        )
                        continue
                    f = _find_image_file(run, ds)
                    if f is None:
                        row_imgs.append(None)
                        row_labels.append(
                            f"{args.param_key}={pv} seed={seed} (no img)"
                        )
                        continue
                    local_path = f.download(root=tmpdir, replace=True).name
                    try:
                        img = Image.open(local_path).convert("RGB")
                    except Exception:
                        img = None
                    row_imgs.append(img)
                    row_labels.append(f"{args.param_key}={pv} seed={seed}")
                images.append(row_imgs)
                labels.append(row_labels)
            out_path = outdir / f"grid_{ds}_contours.png"
            _make_grid(images, labels, out_path)
            print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
