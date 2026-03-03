from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from evaluator.evaluator import eval_bounds_from_train


def plot_contour_traj_2d(
    *,
    model: nn.Module,
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str],
    cfg: Any,
    line_color: str = "green",
) -> None:
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    if axis_labels[0].startswith("q") and axis_labels[1].startswith("q"):
        mins = mins.copy()
        maxs = maxs.copy()
        mins[0] = min(float(mins[0]), -np.pi)
        maxs[0] = max(float(maxs[0]), np.pi)
        mins[1] = min(float(mins[1]), -np.pi)
        maxs[1] = max(float(maxs[1]), np.pi)

    xx, yy = np.meshgrid(
        np.linspace(float(mins[0]), float(maxs[0]), 300),
        np.linspace(float(mins[1]), float(maxs[1]), 300),
    )
    grid = np.stack([xx, yy], axis=2).reshape(-1, 2).astype(np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        fg = model(torch.from_numpy(grid).to(device))
        if fg.dim() == 1:
            fg = fg.unsqueeze(1)
        v = (0.5 * torch.sum(fg * fg, dim=1)).detach().cpu().numpy().reshape(xx.shape)
        fon = model(torch.from_numpy(x_train.astype(np.float32)).to(device))
        if fon.dim() == 1:
            fon = fon.unsqueeze(1)
        h_on = torch.linalg.norm(fon, dim=1).detach().cpu().numpy().reshape(-1)

    eps_q = float(getattr(cfg, "zero_eps_quantile", 90.0))
    eps_h = max(float(np.percentile(np.abs(h_on), eps_q)), 1e-6)
    v_max = float(np.percentile(v, 95))
    levels = (np.linspace(0.0, 1.0, 25) ** 3) * max(v_max, 1e-10)

    plt.figure(figsize=(8.0, 6.0))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8, colors="#475569")
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    # Zero-band contour region (legacy style).
    plt.contourf(xx, yy, np.sqrt(np.maximum(v * 2.0, 0.0)), levels=[0.0, eps_h], colors=["#f59e0b"], alpha=0.32)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=8, c="gray", alpha=0.62, zorder=3)

    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], "-", color=line_color, linewidth=0.9, alpha=0.9)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(float(mins[0]), float(maxs[0]))
    plt.ylim(float(mins[1]), float(maxs[1]))
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
