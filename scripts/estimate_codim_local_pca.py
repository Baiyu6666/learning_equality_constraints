#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from levelset_energy_algorithm import Config, generate_dataset, set_seed


ALL_DATASETS = [
    "figure_eight",
    "high_curvature",
    "ellipse",
    "noise_only",
    "sine",
    "sparse_only",
    "discontinuous",
    "looped_spiro",
    "sharp_star",
    "hetero_noise",
    "double_valley",
    "hairpin",
    "planar_arm_line_n2",
    "planar_arm_line_n3",
    "spatial_arm_plane_n3",
    "spatial_arm_circle_n3",
    "spatial_arm_plane_n4",
    "saddle_surface",
    "sphere_surface",
    "torus_surface",
    "spatial_arm_up_n6",
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


def _local_neighbors(
    x: np.ndarray,
    idx: int,
    k: int,
    is_periodic_joint: bool,
) -> np.ndarray:
    k = int(max(2, min(k, len(x) - 1)))
    center = x[idx]
    if is_periodic_joint:
        diff = _wrap_to_pi(x - center.reshape(1, -1))
        d2 = np.sum(diff * diff, axis=1)
        order = np.argsort(d2)
        nbr = order[1 : k + 1]
        return nbr.astype(np.int64)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(x.astype(np.float64))
        _, nbr = tree.query(center.astype(np.float64), k=k + 1)
        nbr = np.asarray(nbr, dtype=np.int64).reshape(-1)
        nbr = nbr[nbr != idx]
        if len(nbr) >= k:
            return nbr[:k]
    except Exception:
        pass
    d2 = np.sum((x - center.reshape(1, -1)) ** 2, axis=1)
    order = np.argsort(d2)
    return order[1 : k + 1].astype(np.int64)


def _estimate_point_codim(
    x: np.ndarray,
    idx: int,
    k: int,
    is_periodic_joint: bool,
    eps: float = 1e-10,
) -> int:
    d = int(x.shape[1])
    nbr_idx = _local_neighbors(x, idx, k, is_periodic_joint=is_periodic_joint)
    center = x[idx]
    neigh = x[nbr_idx]
    if is_periodic_joint:
        neigh = center.reshape(1, -1) + _wrap_to_pi(neigh - center.reshape(1, -1))
    mu = np.mean(neigh, axis=0, keepdims=True)
    xc = neigh - mu
    cov = (xc.T @ xc) / max(1, len(neigh))
    evals = np.linalg.eigvalsh(cov.astype(np.float64))
    evals = np.maximum(evals, eps)
    # Gap between small-normal and large-tangent eigenvalues.
    # score_j corresponds to codim = j+1.
    log_ratio = np.log(evals[1:] + eps) - np.log(evals[:-1] + eps)
    codim = int(np.argmax(log_ratio) + 1)
    codim = int(np.clip(codim, 1, d - 1))
    return codim


def estimate_codim_local_pca(
    x: np.ndarray,
    sample_ratio: float = 0.2,
    k_neighbors: int = 0,
    periodic_joint: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    n, d = x.shape
    n_sample = int(max(16, min(n, round(float(sample_ratio) * n))))
    rng = np.random.default_rng(int(seed))
    sample_idx = rng.choice(n, size=n_sample, replace=False)

    if int(k_neighbors) > 0:
        k = int(k_neighbors)
    else:
        k = int(np.clip(round(np.sqrt(n)), d + 2, min(96, n - 1)))

    codims = np.zeros((n_sample,), dtype=np.int64)
    for i, idx in enumerate(sample_idx):
        codims[i] = _estimate_point_codim(
            x,
            int(idx),
            k=k,
            is_periodic_joint=bool(periodic_joint),
        )

    counts = np.bincount(codims, minlength=d + 1)
    mode_codim = int(np.argmax(counts[1:]) + 1)
    mode_count = int(counts[mode_codim])
    mode_frac = float(mode_count / max(1, n_sample))

    hist = {str(i): int(counts[i]) for i in range(1, d) if counts[i] > 0}
    return {
        "n": int(n),
        "d": int(d),
        "sample_ratio": float(sample_ratio),
        "n_sample": int(n_sample),
        "k_neighbors": int(k),
        "estimated_codim": mode_codim,
        "mode_count": mode_count,
        "mode_fraction": mode_frac,
        "codim_histogram": hist,
    }


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
    parser.add_argument("--sample-ratio", type=float, default=0.2)
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
