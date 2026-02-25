#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from levelset_energy_algorithm import Config, generate_dataset, sample_eval_points, true_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="figure_eight")
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = Config()
    rng = np.random.default_rng(args.seed)
    x_train, grid = generate_dataset(args.dataset, cfg)
    x_eval = sample_eval_points(x_train, grid, cfg, rng)

    d_true = true_distance(x_eval, grid)
    tau = 0.02 * float(np.mean(grid.max(axis=0) - grid.min(axis=0)))
    near = d_true < tau
    print('percentage of on data:', near.sum()/ d_true.shape[0])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"near_band_{args.dataset}.png"

    plt.figure(figsize=(7, 5))
    plt.scatter(x_train[:, 0], x_train[:, 1], s=14, alpha=0.7, label="train (on)")
    plt.scatter(
        x_eval[near, 0],
        x_eval[near, 1],
        s=10,
        alpha=0.6,
        label=f"eval near (tau={tau:.3f})",
    )
    plt.title(f"{args.dataset}: near-band samples")
    plt.legend(loc="best")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
