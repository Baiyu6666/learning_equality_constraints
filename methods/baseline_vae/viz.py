#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch

from methods.baseline_vae.utils import to_tensor


def visualize_all(
    ds,
    x_train,
    ae_pack=None,   # (cache, z_eval, x_dec)
    vae_pack=None,  # (cache, z_eval, x_dec)
    latent_dim=1,
    max_arrows_2d=30,
    max_arrows_3d=30,
    arrow_color="tab:gray",
    save_path: str | None = None,
    show: bool = True,
):
    """
    2x3 layout:
      row 0: AE  -> [latent, decode, projection+arrows]
      row 1: VAE -> [latent, decode, projection+arrows]

    ae_pack / vae_pack:
      cache: dict with keys ["x", "y_true", "x_proj", "err"] from eval
      z_eval: (N, latent_dim) encoded eval points (AE: z, VAE: mu)
      x_dec: (M, dim) decoded points from latent sampling
    """
    fig = plt.figure(figsize=(16, 10))

    def _plot_latent(ax, z_eval, y_true, z_sample, title):
        on = y_true == 1
        off = y_true == 0

        if z_eval.shape[1] == 1:
            jitter = 0.02 * np.random.randn(len(z_eval))
            # jitter are random values to visually separate points along y-axis; they don't represent any real value
            ax.scatter(z_eval[on, 0], jitter[on], s=12, label="GT ON")
            ax.scatter(z_eval[off, 0], jitter[off], s=12, label="GT OFF")

            # new sampled latent points
            jitter2 = 0.02 * np.random.randn(len(z_sample))
            ax.scatter(
                z_sample[:, 0],
                jitter2,
                s=20,
                marker="x",
                color="red",
                label="sampled z",
            )

            ax.set_yticks([])
            ax.set_xlabel("z[0]")
        else:
            ax.scatter(z_eval[on, 0], z_eval[on, 1], s=12, label="GT ON")
            ax.scatter(z_eval[off, 0], z_eval[off, 1], s=12, label="GT OFF")

            ax.scatter(z_sample[:, 0], z_sample[:, 1], s=25, marker="x", color="red", label="sampled z")

            ax.set_xlabel("z[0]")
            ax.set_ylabel("z[1]")

        ax.set_title(title)
        ax.legend(loc="best")

    def _plot_decode(ax, x_dec, title):
        # add legend required
        if ds.dim == 2:
            ax.scatter(x_train[:, 0], x_train[:, 1], s=8, alpha=0.20, label="train (GT ON)")
            ax.scatter(x_dec[:, 0], x_dec[:, 1], s=14, alpha=0.9, label="decoded")
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.legend(loc="best")
        else:
            ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=8, alpha=0.12, label="train (GT ON)")
            ax.scatter(x_dec[:, 0], x_dec[:, 1], x_dec[:, 2], s=18, alpha=0.9, label="decoded")
            ax.set_title(title)
            ax.legend(loc="best")

    def _plot_projection(ax, cache, title):
        # add legend required; arrows as dashed lines with same color
        x = cache["x"]
        y_true = cache["y_true"]
        x_proj = cache["x_proj"]

        on = y_true == 1
        off = y_true == 0

        if ds.dim == 2:
            ax.scatter(x[on, 0], x[on, 1], s=10, alpha=0.35, label="GT ON (orig)")
            ax.scatter(x[off, 0], x[off, 1], s=10, alpha=0.35, label="GT OFF (orig)")

            n = len(x)
            idx = np.arange(n)
            if n > max_arrows_2d:
                idx = np.random.choice(n, size=3 * max_arrows_2d, replace=False)

            # dashed line segments: same color for all
            for i in idx:
                ax.plot(
                    [x[i, 0], x_proj[i, 0]],
                    [x[i, 1], x_proj[i, 1]],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    color=arrow_color,
                )

            ax.scatter(
                x_proj[idx, 0],
                x_proj[idx, 1],
                s=26,
                alpha=0.9,
                marker="x",
                color="red",
                label="projected",
            )
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.legend(loc="best")
        else:
            ax.scatter(x[on, 0], x[on, 1], x[on, 2], s=12, alpha=0.22, label="GT ON (orig)")
            ax.scatter(x[off, 0], x[off, 1], x[off, 2], s=12, alpha=0.22, label="GT OFF (orig)")

            n = len(x)
            idx = np.arange(n)
            if n > max_arrows_3d:
                idx = np.random.choice(n, size=3 * max_arrows_3d, replace=False)

            for i in idx:
                ax.plot(
                    [x[i, 0], x_proj[i, 0]],
                    [x[i, 1], x_proj[i, 1]],
                    [x[i, 2], x_proj[i, 2]],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    color=arrow_color,
                )

            ax.scatter(
                x_proj[idx, 0],
                x_proj[idx, 1],
                x_proj[idx, 2],
                s=28,
                alpha=0.9,
                marker="x",
                color="red",
                label="projected",
            )
            ax.set_title(title)
            ax.legend(loc="best")

    # Helper to create correct axes (2D vs 3D) per column
    def _make_ax(row, col):
        # col: 0 latent(2D), 1 decode(2D/3D), 2 projection(2D/3D)
        pos = row * 3 + col + 1  # 1..6
        if col == 0:
            return fig.add_subplot(2, 3, pos)  # latent always 2D plot
        if ds.dim == 2:
            return fig.add_subplot(2, 3, pos)
        return fig.add_subplot(2, 3, pos, projection="3d")

    # --- Row 0: AE ---
    if ae_pack is not None:
        cache, z_eval, x_dec, z_samp = ae_pack
        ax = _make_ax(0, 0)
        _plot_latent(ax, z_eval, cache["y_true"], z_samp, title="AE: latent (E(x))")

        ax = _make_ax(0, 1)
        _plot_decode(ax, x_dec, title="AE: decode (sample z → D(z))")

        ax = _make_ax(0, 2)
        _plot_projection(ax, cache, title="AE: projection (x → D(E(x)))")
    else:
        # keep layout consistent
        for c in range(3):
            ax = _make_ax(0, c)
            ax.set_axis_off()
            ax.set_title("AE: (not run)")

    # --- Row 1: VAE ---
    if vae_pack is not None:
        cache, z_eval, x_dec, z_samp = vae_pack
        ax = _make_ax(1, 0)
        _plot_latent(ax, z_eval, cache["y_true"], z_samp, title="VAE: latent (mu(x))")

        ax = _make_ax(1, 1)
        _plot_decode(ax, x_dec, title="VAE: decode (z~N(0,1) → D(z))")

        ax = _make_ax(1, 2)
        _plot_projection(ax, cache, title="VAE: projection (x → D(mu(x)))")
    else:
        for c in range(3):
            ax = _make_ax(1, c)
            ax.set_axis_off()
            ax.set_title("VAE: (not run)")

    fig.suptitle(f"Dataset: {ds.name} | dim={ds.dim} | latent_dim={latent_dim}", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_conditional_decodes(
    ds,
    x_on: np.ndarray,
    model,
    device: torch.device,
    n_points: int = 3,
    n_samples: int = 30,
    seed: int = 123,
) -> None:
    """
    Pick a few ON-manifold points, sample z ~ q(z|x), decode, and plot the decoded clouds.
    """
    if x_on.shape[0] == 0:
        return

    rng = np.random.default_rng(seed)
    idx = rng.choice(x_on.shape[0], size=min(n_points, x_on.shape[0]), replace=False)
    x_sel = x_on[idx]

    with torch.no_grad():
        xt = to_tensor(x_sel, device)
        mu, logvar = model.encode(xt)
        std = torch.exp(0.5 * logvar)

        all_decoded = []
        for i in range(x_sel.shape[0]):
            eps = torch.randn(n_samples, mu.shape[1], device=device)
            z = mu[i].unsqueeze(0) + eps * std[i].unsqueeze(0)
            x_dec = model.decode(z).cpu().numpy()
            all_decoded.append(x_dec)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_decoded)))

    if ds.dim == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x_on[:, 0], x_on[:, 1], s=8, alpha=0.18, label="GT ON")
        for i, x_dec in enumerate(all_decoded):
            ax.scatter(
                x_dec[:, 0],
                x_dec[:, 1],
                s=18,
                alpha=0.9,
                color=colors[i],
                label=f"x#{i+1}",
            )
            ax.scatter(
                x_sel[i, 0],
                x_sel[i, 1],
                s=120,
                marker="X",
                edgecolor="black",
                linewidth=1.0,
                color=colors[i],
                label=f"x#{i+1} (orig)",
            )
        ax.set_aspect("equal")
        ax.set_title("VAE: q(z|x) samples decoded")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_on[:, 0], x_on[:, 1], x_on[:, 2], s=8, alpha=0.15, label="GT ON")
        for i, x_dec in enumerate(all_decoded):
            ax.scatter(
                x_dec[:, 0],
                x_dec[:, 1],
                x_dec[:, 2],
                s=22,
                alpha=0.9,
                color=colors[i],
                label=f"x#{i+1}",
            )
            ax.scatter(
                x_sel[i, 0],
                x_sel[i, 1],
                x_sel[i, 2],
                s=140,
                marker="X",
                edgecolor="black",
                linewidth=1.0,
                color=colors[i],
                label=f"x#{i+1} (orig)",
            )
        ax.set_title("VAE: q(z|x) samples decoded")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()


def plot_planner_grid(
    ds,
    x_train: np.ndarray,
    cases,
    title_prefix: str = "Planner",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    row_order = [k for k in ["projected", "latent", "linear_projected"] if k in cases]
    if not row_order:
        return
    n_rows = len(row_order)
    n_cols = max(len(cases.get(k, [])) for k in row_order)
    n_cols = max(1, n_cols)
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    def _make_ax(row, col):
        pos = row * n_cols + col + 1
        if ds.dim == 2:
            return fig.add_subplot(n_rows, n_cols, pos)
        return fig.add_subplot(n_rows, n_cols, pos, projection="3d")

    def _plot_case(ax, case, title):
        traj = case["traj"]
        x0 = case["x0"]
        x1 = case["x1"]
        metrics = case["metrics"]

        if ds.dim == 2:
            ax.scatter(x_train[:, 0], x_train[:, 1], s=6, alpha=0.10, label="train (GT ON)")
            ax.plot(traj[:, 0], traj[:, 1], color="tab:orange", linewidth=2.0, label="traj")
            ax.scatter([x0[0]], [x0[1]], s=80, color="green", marker="o", label="start")
            ax.scatter([x1[0]], [x1[1]], s=80, color="red", marker="X", label="end")
            ax.set_aspect("equal")
        else:
            ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=6, alpha=0.08, label="train (GT ON)")
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:orange", linewidth=2.0, label="traj")
            ax.scatter([x0[0]], [x0[1]], [x0[2]], s=80, color="green", marker="o", label="start")
            ax.scatter([x1[0]], [x1[1]], [x1[2]], s=80, color="red", marker="X", label="end")

        ax.set_title(title)
        ax.legend(loc="best")
        text = (
            f"mean dist GT={metrics['mean_gt_dist']:.4f}\n"
            f"on/off={metrics['on_count']}/{metrics['off_count']}  on%={metrics['on_ratio'] * 100:.1f}%"
        )
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    title_map = {
        "projected": "projected",
        "latent": "latent",
        "linear_projected": "linear->DE",
    }
    for r, key in enumerate(row_order):
        for i, case in enumerate(cases.get(key, [])):
            ax = _make_ax(r, i)
            _plot_case(ax, case, f"{title_prefix}: {title_map.get(key, key)} path #{i+1}")

    fig.suptitle(f"{title_prefix} | Dataset: {ds.name}", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
