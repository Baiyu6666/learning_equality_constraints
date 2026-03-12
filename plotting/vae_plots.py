from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_vae_loss_curves(
    history: dict[str, list[float]],
    out_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    for k, c in (
        ("loss_total", "#2563eb"),
        ("loss_recon", "#16a34a"),
        ("loss_kl", "#dc2626"),
    ):
        vals = history.get(k, [])
        if vals:
            ax.plot(vals, linewidth=1.4, label=k, color=c)
    beta = history.get("beta", [])
    if beta:
        ax2 = ax.twinx()
        ax2.plot(beta, linewidth=1.2, alpha=0.6, color="#7c3aed", label="beta")
        ax2.set_ylabel("beta")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_all(
    ds: SimpleNamespace,
    x_train: np.ndarray,
    ae_pack=None,
    vae_pack=None,
    latent_dim: int = 1,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    fig = plt.figure(figsize=(12, 5))

    def _make_ax(col: int):
        pos = col + 1
        if int(ds.dim) == 2:
            return fig.add_subplot(1, 2, pos)
        return fig.add_subplot(1, 2, pos, projection="3d")

    if vae_pack is None:
        for c in range(2):
            ax = _make_ax(c)
            ax.set_axis_off()
            ax.set_title("VAE: (not available)")
    else:
        cache, z_eval, x_dec, z_samp = vae_pack
        x = cache["x"]
        y_true = cache["y_true"]
        x_proj = cache["x_proj"]
        on = y_true == 1
        off = y_true == 0
        rng = np.random.default_rng(0)
        off_idx_all = np.where(off)[0]
        if len(off_idx_all) > 16:
            off_idx = rng.choice(off_idx_all, size=16, replace=False)
            off_mask = np.zeros_like(off, dtype=bool)
            off_mask[off_idx] = True
        else:
            off_mask = off

        ax = _make_ax(0)
        if int(ds.dim) == 2:
            ax.scatter(x_train[:, 0], x_train[:, 1], s=8, alpha=0.2, label="train")
            ax.scatter(x_dec[:, 0], x_dec[:, 1], s=14, alpha=0.9, label="generated from sampled z")
            ax.set_aspect("equal")
        else:
            ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=8, alpha=0.12, label="train")
            ax.scatter(x_dec[:, 0], x_dec[:, 1], x_dec[:, 2], s=16, alpha=0.9, label="generated from sampled z")
        ax.set_title("VAE: decode (z~N(0,1))")
        ax.legend(loc="best")

        ax = _make_ax(1)
        if int(ds.dim) == 2:
            ax.scatter(x[on, 0], x[on, 1], s=8, alpha=0.3, label="GT ON (orig)")
            ax.scatter(x[off_mask, 0], x[off_mask, 1], s=8, alpha=0.3, label="GT OFF (orig)")
            idx = np.where(off_mask)[0]
            if len(idx) == 0:
                n = len(x)
                idx = np.arange(n) if n <= 16 else np.random.choice(n, size=16, replace=False)
            for i in idx.tolist():
                ax.plot([x[i, 0], x_proj[i, 0]], [x[i, 1], x_proj[i, 1]], "--", linewidth=1.0, alpha=0.8, color="gray")
            ax.scatter(x_proj[idx, 0], x_proj[idx, 1], s=18, marker="x", color="red", label="projected")
            ax.set_aspect("equal")
        else:
            rendered_surface = False
            try:
                from skimage import measure  # type: ignore

                pts = x_dec.astype(np.float32)
                mins = np.min(pts, axis=0)
                maxs = np.max(pts, axis=0)
                span = np.maximum(maxs - mins, 1e-6)
                mins = mins - 0.08 * span
                maxs = maxs + 0.08 * span

                n = 34
                gx, gy, gz = np.meshgrid(
                    np.linspace(float(mins[0]), float(maxs[0]), n),
                    np.linspace(float(mins[1]), float(maxs[1]), n),
                    np.linspace(float(mins[2]), float(maxs[2]), n),
                    indexing="ij",
                )
                grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

                try:
                    from scipy.spatial import cKDTree  # type: ignore

                    tree = cKDTree(pts.astype(np.float64))
                    d_grid, _ = tree.query(grid.astype(np.float64), k=1)
                    if len(pts) >= 8:
                        d_nn, _ = tree.query(pts.astype(np.float64), k=2)
                        nn = d_nn[:, 1]
                        radius = float(np.percentile(nn, 70))
                    else:
                        radius = float(0.05 * np.mean(span))
                except Exception:
                    aa = grid.astype(np.float32)
                    bb = pts.astype(np.float32)
                    d2 = np.sum((aa[:, None, :] - bb[None, :, :]) ** 2, axis=2)
                    d_grid = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0)).astype(np.float64)
                    radius = float(0.05 * np.mean(span))

                radius = max(radius, 1e-4)
                vol = d_grid.reshape(n, n, n).astype(np.float32)
                dx = float((maxs[0] - mins[0]) / max(1, n - 1))
                dy = float((maxs[1] - mins[1]) / max(1, n - 1))
                dz = float((maxs[2] - mins[2]) / max(1, n - 1))
                verts, faces, _, _ = measure.marching_cubes(
                    vol,
                    level=radius,
                    spacing=(dx, dy, dz),
                )
                verts = verts + np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
                poly = Poly3DCollection(
                    verts[np.asarray(faces, dtype=np.int32)],
                    alpha=0.18,
                    facecolor="#22d3ee",
                    edgecolor=(0, 0, 0, 0.08),
                    linewidth=0.05,
                )
                poly.set_label("learned surface (from generated)")
                ax.add_collection3d(poly)
                rendered_surface = True
            except Exception:
                rendered_surface = False

            if not rendered_surface:
                ax.scatter(
                    x_dec[:, 0],
                    x_dec[:, 1],
                    x_dec[:, 2],
                    s=4,
                    alpha=0.18,
                    color="#22d3ee",
                    label="learned surface (generated pts)",
                )
            ax.scatter(x[on, 0], x[on, 1], x[on, 2], s=10, alpha=0.2, label="GT ON (orig)")
            ax.scatter(x[off_mask, 0], x[off_mask, 1], x[off_mask, 2], s=10, alpha=0.2, label="GT OFF (orig)")
            idx = np.where(off_mask)[0]
            if len(idx) == 0:
                n = len(x)
                idx = np.arange(n) if n <= 16 else np.random.choice(n, size=16, replace=False)
            for i in idx.tolist():
                ax.plot(
                    [x[i, 0], x_proj[i, 0]],
                    [x[i, 1], x_proj[i, 1]],
                    [x[i, 2], x_proj[i, 2]],
                    "--",
                    linewidth=1.0,
                    alpha=0.8,
                    color="gray",
                )
            ax.scatter(x_proj[idx, 0], x_proj[idx, 1], x_proj[idx, 2], s=18, marker="x", color="red", label="projected")
        ax.set_title("VAE: projection (x -> D(mu(x)))")
        ax.legend(loc="best")

    fig.suptitle(f"Dataset: {ds.name} | dim={ds.dim} | latent_dim={latent_dim}", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
