#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

import numpy as np

from methods.baseline_udf.baseline_udf import Config
from datasets.constraint_datasets import generate_dataset, set_seed
from methods.vector_eikonal.codim_utils import estimate_codim_local_pca


ALL_DATASETS = [
    "2d_figure_eight",
    "2d_ellipse",
    "2d_noisy_sine",
    "2d_sine",
    "2d_sparse_sine",
    "2d_discontinuous",
    "2d_looped_spiro",
    "2d_sharp_star",
    "2d_hetero_noise",
    "2d_hairpin",
    "2d_planar_arm_line_n2",
    "3d_0z_2d_ellipse",
    "3d_planar_arm_line_n3",
    "3d_spatial_arm_plane_n3",
    "3d_spatial_arm_circle_n3",
    "4d_spatial_arm_plane_n4",
    "3d_saddle_surface",
    "3d_sphere_surface",
    "3d_torus_surface",
    "6d_spatial_arm_up_n6",
]


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _is_joint_arm_dataset(name: str) -> bool:
    return "arm_" in str(name)


def _load_dataset(name: str, n_train: int, n_grid: int, seed: int) -> np.ndarray:
    cfg = Config()
    cfg.n_train = int(n_train)
    cfg.n_grid = int(n_grid)
    cfg.seed = int(seed)
    set_seed(cfg.seed)
    x, _ = generate_dataset(name, cfg)
    x = x.astype(np.float32)
    if _is_joint_arm_dataset(name):
        x = _wrap_to_pi(x)
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset names. Use 'all' for built-in list.",
    )
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-grid", type=int, default=4096)
    parser.add_argument("--sample-ratio", type=float, default=0.02)
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=0,
        help="0 means auto(sqrt(N), clipped).",
    )
    parser.add_argument("--seed", type=int, default=2116)
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs_levelset_datasets/codim_estimation",
    )
    args = parser.parse_args()

    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        datasets = list(ALL_DATASETS)
    else:
        datasets = list(args.datasets)

    os.makedirs(args.outdir, exist_ok=True)
    out_json = os.path.join(args.outdir, "local_pca_codim_results.json")
    out_txt = os.path.join(args.outdir, "local_pca_codim_results.txt")

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(datasets):
        try:
            x = _load_dataset(name, args.n_train, args.n_grid, seed=args.seed + i)
            is_joint = _is_joint_arm_dataset(name)
            res = estimate_codim_local_pca(
                x=x,
                sample_ratio=float(args.sample_ratio),
                k_neighbors=int(args.k_neighbors),
                periodic_joint=is_joint,
                seed=args.seed + 1000 + i,
            )
            row = {"dataset": name, **res}
            rows.append(row)
            print(
                f"[ok] {name:24s} | d={res['d']} | codim={res['estimated_codim']} "
                f"| mode_frac={res['mode_fraction']:.3f} | k={res['k_neighbors']}"
            )
        except Exception as e:
            rows.append({"dataset": name, "error": str(e)})
            print(f"[fail] {name}: {e}")

    payload = {
        "config": {
            "n_train": int(args.n_train),
            "n_grid": int(args.n_grid),
            "sample_ratio": float(args.sample_ratio),
            "k_neighbors": int(args.k_neighbors),
            "seed": int(args.seed),
            "datasets": datasets,
            "base_levelset_config": asdict(Config()),
        },
        "results": rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("Local-PCA eigengap codim estimation")
    lines.append(f"datasets={len(datasets)}, sample_ratio={args.sample_ratio}, k_neighbors={args.k_neighbors}")
    lines.append("")
    for row in rows:
        if "error" in row:
            lines.append(f"{row['dataset']}: ERROR {row['error']}")
            continue
        lines.append(
            f"{row['dataset']}: d={row['d']}, codim={row['estimated_codim']}, "
            f"mode_fraction={row['mode_fraction']:.3f}, hist={row['codim_histogram']}"
        )
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"saved: {out_json}")
    print(f"saved: {out_txt}")


if __name__ == "__main__":
    main()
