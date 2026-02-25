#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from levelset_energy_algorithm import (
    Config,
    MLP,
    energy_from_f,
    generate_dataset,
    sample_eval_points,
    true_distance,
    evaluate_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ellipse")
    parser.add_argument("--outdir", default="outputs_sweep_analysis")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=3000)
    args = parser.parse_args()

    cfg = Config()
    cfg.epochs = args.epochs
    cfg.device = "cpu"
    cfg.train_n = 512
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    x_train, grid = generate_dataset(args.dataset, cfg)

    # Supervised model: regress V(x) ≈ d_true^2 on on+off samples
    x_on = x_train
    n_off = x_on.shape[0] * 3
    mins = grid.min(axis=0)
    maxs = grid.max(axis=0)
    span = maxs - mins
    mins = mins - 0.5 * span
    maxs = maxs + 0.5 * span
    x_off = rng.uniform(mins, maxs, size=(n_off, grid.shape[1])).astype(np.float32)
    x_all = np.concatenate([x_on, x_off], axis=0)
    d_true = true_distance(x_all, grid)
    y_all = (d_true ** 2).astype(np.float32)

    x_all_t = torch.from_numpy(x_all)
    y_all_t = torch.from_numpy(y_all).unsqueeze(1)
    model = MLP(in_dim=x_all.shape[1], hidden=64, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        f = model(x_all_t)
        v = energy_from_f(f)
        loss = torch.mean((v - y_all_t) ** 2)
        loss.backward()
        opt.step()

    x_eval = sample_eval_points(x_train, grid, cfg, rng)
    metrics = evaluate_metrics(model, x_train, grid, x_eval, cfg)

    # Visualization: predicted on vs true on
    x_eval_t = torch.from_numpy(x_eval)
    with torch.no_grad():
        f_eval = model(x_eval_t).numpy().reshape(-1)
    d_eval = true_distance(x_eval, grid)
    tau = cfg.eval_tau_ratio * float(np.mean(grid.max(axis=0) - grid.min(axis=0)))
    true_on = d_eval < tau
    pred_on = np.abs(f_eval) < metrics["eval_eps_used"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_plot = outdir / f"eval_demo_{args.dataset}.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(
        x_eval[~true_on, 0],
        x_eval[~true_on, 1],
        s=10,
        alpha=0.2,
        label="true off",
    )
    ax.scatter(
        x_eval[true_on, 0],
        x_eval[true_on, 1],
        s=10,
        alpha=0.4,
        label="true on",
    )
    ax.scatter(
        x_eval[pred_on, 0],
        x_eval[pred_on, 1],
        s=18,
        alpha=0.6,
        marker="x",
        color="red",
        label="pred on",
    )
    ax.set_title(f"{args.dataset} eval demo | eps={metrics['eval_eps_used']:.4f}")
    ax.set_xlim(x_eval[:, 0].min(), x_eval[:, 0].max())
    ax.set_ylim(x_eval[:, 1].min(), x_eval[:, 1].max())
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.scatter(x_train[:, 0], x_train[:, 1], s=12, alpha=0.6, label="train (on)")
    x_min, x_max = x_eval[:, 0].min(), x_eval[:, 0].max()
    y_min, y_max = x_eval[:, 1].min(), x_eval[:, 1].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        f_grid = model(torch.from_numpy(grid_eval)).numpy().reshape(xx.shape)
    ax.contour(xx, yy, f_grid, levels=15, linewidths=0.6, alpha=0.7)
    ax.contour(
        xx,
        yy,
        f_grid,
        levels=[-metrics["eval_eps_used"], metrics["eval_eps_used"]],
        colors=["red"],
        linewidths=1.5,
    )
    ax.set_title("train + contour (adaptive eps)")
    ax.set_xlim(x_eval[:, 0].min(), x_eval[:, 0].max())
    ax.set_ylim(x_eval[:, 1].min(), x_eval[:, 1].max())
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"saved plot: {out_plot}")


if __name__ == "__main__":
    main()
