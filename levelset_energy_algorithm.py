"""
Standalone implementation of a level-set energy learning algorithm for
equality-constraint manifolds. This file does not import or interact with
any other project modules.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class Config:
    seed: int = 7
    device: str = "auto"
    n_train: int = 256
    n_grid: int = 4096
    k_ratio: float = 0.08
    k_min: int = 4
    k_accept_ratio: float = 0.2
    k_accept_min: int = 5
    sigmas: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.8)
    smooth_sigma: float = 0.01
    hidden: int = 128
    depth: int = 3
    lr: float = 3e-4
    epochs: int = 2000
    batch_size: int = 128
    # loss weights
    lam_on: float = 1.0
    lam_dist: float = .3
    lam_dir: float = 0.
    lam_pl: float = 0.05
    lam_smooth: float = 0.02
    warmup_epochs: int = 500
    margin: float = 0.5
    baseline_w_on: float = 5.0
    baseline_w_off: float = 1.0
    zero_level_eps: float = 0.01
    loss_ema_alpha: float = 0.85
    lam_thin: float = 0.0
    denoise_step: float = 0.1
    lam_denoise: float = 1.0
    lam_recon: float = 1.0
    # loss hyperparams
    beta_pl: float = 0.5
    eps: float = 1e-8
    # projection params
    proj_alpha: float = 0.2
    proj_steps: int = 100
    tau_g: float = 1e-5
    tau_v: float = 1e-7
    tau_x: float = 1e-7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sine_manifold(n: int) -> np.ndarray:
    t = np.random.uniform(-math.pi, math.pi, size=(n, 1))
    x = np.concatenate([t, np.sin(t)], axis=1)
    return x.astype(np.float32)


def make_sine_grid(n: int) -> np.ndarray:
    t = np.linspace(-math.pi, math.pi, n).reshape(-1, 1)
    x = np.concatenate([t, np.sin(t)], axis=1)
    return x.astype(np.float32)


def generate_dataset(name: str, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    if name == "figure_eight":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        x = np.concatenate([np.sin(t), np.sin(2 * t)], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([np.sin(tg), np.sin(2 * tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "high_curvature":
        t = np.random.uniform(-3.0, 3.0, size=(cfg.n_train, 1))
        y = np.sin(3 * t) + 0.3 * np.sin(9 * t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-3.0, 3.0, cfg.n_grid).reshape(-1, 1)
        yg = np.sin(3 * tg) + 0.3 * np.sin(9 * tg)
        grid = np.concatenate([tg, yg], axis=1).astype(np.float32)
        return x, grid
    if name == "ellipse":
        t = np.random.uniform(0.0, 2.0 * math.pi, size=(cfg.n_train, 1))
        a, b = 2.0, 1.0
        x = np.concatenate([a * np.cos(t), b * np.sin(t)], axis=1)
        # Remove a top arc segment (hole)
        mask = x[:, 1] < 0.75
        x = x[mask]
        # Add dense noise cluster (bottom-left)
        n_dense = max(80, cfg.n_train // 2)
        dense = np.random.randn(n_dense, 2).astype(np.float32) * 0.08
        dense += np.array([-1.5, -0.6], dtype=np.float32)
        # Add sparse large-variance noise (bottom-right)
        n_sparse = max(20, cfg.n_train // 8)
        sparse = np.random.randn(n_sparse, 2).astype(np.float32) * 0.2
        sparse += np.array([1.6, -0.8], dtype=np.float32)
        x = np.concatenate([x.astype(np.float32), dense, sparse], axis=0)
        tg = np.linspace(0.0, 2.0 * math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([a * np.cos(tg), b * np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "saddle_surface":
        n = cfg.n_train
        xy = np.random.uniform(-1.5, 1.5, size=(n, 2))
        z = 0.3 * (xy[:, 0:1] ** 2 - xy[:, 1:2] ** 2)
        x = np.concatenate([xy, z], axis=1).astype(np.float32)
        m = int(math.sqrt(cfg.n_grid))
        gx = np.linspace(-1.5, 1.5, m)
        gy = np.linspace(-1.5, 1.5, m)
        gx, gy = np.meshgrid(gx, gy)
        gz = 0.3 * (gx ** 2 - gy ** 2)
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        return x, grid
    if name == "sphere_surface":
        n = cfg.n_train
        u = np.random.uniform(0.0, 2.0 * math.pi, size=(n, 1))
        v = np.random.uniform(0.0, math.pi, size=(n, 1))
        r = 1.5
        x = np.concatenate(
            [
                r * np.cos(u) * np.sin(v),
                r * np.sin(u) * np.sin(v),
                r * np.cos(v),
            ],
            axis=1,
        ).astype(np.float32)
        m = int(math.sqrt(cfg.n_grid))
        ug = np.linspace(0.0, 2.0 * math.pi, m)
        vg = np.linspace(0.0, math.pi, m)
        ug, vg = np.meshgrid(ug, vg)
        grid = np.stack(
            [
                r * np.cos(ug) * np.sin(vg),
                r * np.sin(ug) * np.sin(vg),
                r * np.cos(vg),
            ],
            axis=2,
        ).reshape(-1, 3).astype(np.float32)
        return x, grid
    if name == "torus_surface":
        n = cfg.n_train
        u = np.random.uniform(0.0, 2.0 * math.pi, size=(n, 1))
        v = np.random.uniform(0.0, 2.0 * math.pi, size=(n, 1))
        R, r = 2.0, 0.6
        x = np.concatenate(
            [
                (R + r * np.cos(v)) * np.cos(u),
                (R + r * np.cos(v)) * np.sin(u),
                r * np.sin(v),
            ],
            axis=1,
        ).astype(np.float32)
        m = int(math.sqrt(cfg.n_grid))
        ug = np.linspace(0.0, 2.0 * math.pi, m)
        vg = np.linspace(0.0, 2.0 * math.pi, m)
        ug, vg = np.meshgrid(ug, vg)
        grid = np.stack(
            [
                (R + r * np.cos(vg)) * np.cos(ug),
                (R + r * np.cos(vg)) * np.sin(ug),
                r * np.sin(vg),
            ],
            axis=2,
        ).reshape(-1, 3).astype(np.float32)
        return x, grid
    if name == "noise_only":
        n_sparse = max(64, cfg.n_train // 4)
        t = np.random.uniform(-math.pi, math.pi, size=(n_sparse, 1))
        y = np.sin(t) + 0.15 * np.random.randn(n_sparse, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "sparse_only":
        n_sparse = max(32, cfg.n_train // 8)
        t = np.random.uniform(-math.pi, math.pi, size=(n_sparse, 1))
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "discontinuous":
        n_half = cfg.n_train // 2
        t1 = np.random.uniform(-math.pi, -0.7, size=(n_half, 1))
        t2 = np.random.uniform(0.7, math.pi, size=(cfg.n_train - n_half, 1))
        t = np.vstack([t1, t2])
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg1 = np.linspace(-math.pi, -0.7, cfg.n_grid // 2).reshape(-1, 1)
        tg2 = np.linspace(0.7, math.pi, cfg.n_grid - cfg.n_grid // 2).reshape(-1, 1)
        tg = np.vstack([tg1, tg2])
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "high_freq_knot":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        y = np.sin(4 * t) + 0.35 * np.sin(12 * t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        yg = np.sin(4 * tg) + 0.35 * np.sin(12 * tg)
        grid = np.concatenate([tg, yg], axis=1).astype(np.float32)
        return x, grid
    if name == "hetero_noise":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        sigma = 0.02 + 0.3 * np.exp(-0.5 * (t / 0.8) ** 2)
        y = np.sin(t) + sigma * np.random.randn(cfg.n_train, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "double_valley":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        choose = np.random.rand(cfg.n_train, 1) < 0.5
        y1 = np.sin(t)
        y2 = np.sin(t) + 0.6
        y = np.where(choose, y1, y2)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        grid1 = np.concatenate([tg, np.sin(tg)], axis=1)
        grid2 = np.concatenate([tg, np.sin(tg) + 0.35], axis=1)
        grid = np.vstack([grid1, grid2]).astype(np.float32)
        return x, grid
    if name == "hairpin":
        t = np.random.uniform(0.0, 1.0, size=(cfg.n_train, 1))
        x1 = 2.0 * t - 1.0
        y1 = 0.8 * (t ** 2)
        x2 = 1.0 - 2.0 * t
        y2 = 0.8 * (t ** 2) + 0.15
        choose = np.random.rand(cfg.n_train, 1) < 0.5
        x = np.where(choose, x1, x2)
        y = np.where(choose, y1, y2)
        x = np.concatenate([x, y], axis=1).astype(np.float32)
        tg = np.linspace(0.0, 1.0, cfg.n_grid).reshape(-1, 1)
        grid1 = np.concatenate([2.0 * tg - 1.0, 0.8 * (tg ** 2)], axis=1)
        grid2 = np.concatenate([1.0 - 2.0 * tg, 0.8 * (tg ** 2) + 0.15], axis=1)
        grid = np.vstack([grid1, grid2]).astype(np.float32)
        return x, grid
    raise ValueError(f"unknown dataset: {name}")


def pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N, d), b: (M, d)
    a = np.nan_to_num(a, copy=False).astype(np.float64)
    b = np.nan_to_num(b, copy=False).astype(np.float64)
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    d2 = np.maximum(d2, 0.0)
    return d2


def knn_normals(x: np.ndarray, k: int) -> np.ndarray:
    n, d = x.shape
    d2 = pairwise_sqdist(x, x)
    idx = np.argsort(d2, axis=1)[:, 1 : k + 1]
    normals = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        nbrs = x[idx[i]]
        center = nbrs.mean(axis=0, keepdims=True)
        y = nbrs - center
        cov = (y.T @ y) / max(k - 1, 1)
        w, v = np.linalg.eigh(cov)
        nvec = v[:, np.argmin(w)]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        normals[i] = nvec.astype(np.float32)
    return normals


def effective_k(cfg: Config, n: int) -> int:
    return max(cfg.k_min, int(round(cfg.k_ratio * n)))


def effective_k_accept(cfg: Config, n: int) -> int:
    return max(cfg.k_accept_min, int(round(cfg.k_ratio * n * cfg.k_accept_ratio)))


def knn_normals_with_quality(
    x: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = x.shape
    d2 = pairwise_sqdist(x, x)
    idx = np.argsort(d2, axis=1)[:, 1 : k + 1]
    normals = np.zeros((n, d), dtype=np.float32)
    quality = np.zeros((n,), dtype=np.float32)
    thickness = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        nbrs = x[idx[i]]
        center = nbrs.mean(axis=0, keepdims=True)
        y = nbrs - center
        cov = (y.T @ y) / max(k - 1, 1)
        w, v = np.linalg.eigh(cov)
        nvec = v[:, np.argmin(w)]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        normals[i] = nvec.astype(np.float32)
        quality[i] = float(w[0] / (w[-1] + 1e-12))
        proj = (nbrs - x[i : i + 1]) @ nvec.reshape(-1, 1)
        thickness[i] = float(np.sqrt(np.mean(proj ** 2)))
    return normals, quality, thickness


def true_distance(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    d2 = pairwise_sqdist(x, grid)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(np.min(d2, axis=1))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def energy_from_f(f: torch.Tensor) -> torch.Tensor:
    return 0.5 * (f ** 2)


def sample_off_manifold(
    x_on: torch.Tensor, n_hat: torch.Tensor, sigmas: Tuple[float, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sample scalar offsets along the normal direction only
    sigma_choices = torch.tensor(sigmas, device=x_on.device, dtype=x_on.dtype)
    idx = torch.randint(0, len(sigmas), (x_on.shape[0], 1), device=x_on.device)
    sigma = sigma_choices[idx]
    s = torch.randn_like(x_on[:, :1]) * sigma
    x_off = x_on + s * n_hat
    return x_off, s


def filter_off_by_knn(
    x_off: torch.Tensor, x_ref: torch.Tensor, idx_on: torch.Tensor, k_accept: int
) -> torch.Tensor:
    d2 = torch.cdist(x_off, x_ref) ** 2
    nn_idx = torch.topk(d2, k_accept, largest=False).indices
    idx_on = idx_on.view(-1, 1)
    mask = (nn_idx == idx_on).any(dim=1)
    return mask


def thickness_weight(
    t: torch.Tensor, cfg: Config
) -> torch.Tensor:
    return torch.ones_like(t)


def compute_losses(
    model: nn.Module,
    x_on: torch.Tensor,
    n_hat: torch.Tensor,
    t_on: torch.Tensor,
    x_ref: torch.Tensor,
    idx_on: torch.Tensor,
    cfg: Config,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    f_on = model(x_on)
    v_on = energy_from_f(f_on)
    loss_on = (f_on ** 2).mean()
    # Explicit noise model: one-step denoising toward the zero set
    x_on_detached = x_on.detach().requires_grad_(True)
    v_on_det = energy_from_f(model(x_on_detached))
    grad_on = torch.autograd.grad(v_on_det.sum(), x_on_detached, create_graph=True)[0]
    y = x_on_detached - cfg.denoise_step * grad_on
    loss_denoise = (model(y) ** 2).mean()
    loss_recon = ((x_on_detached - y) ** 2).sum(dim=1).mean()

    x_off, s = sample_off_manifold(x_on, n_hat, cfg.sigmas)
    k_accept = effective_k_accept(cfg, x_ref.shape[0])
    mask = filter_off_by_knn(x_off, x_ref, idx_on, k_accept)
    if mask.sum() > 0:
        x_off = x_off[mask]
        s = s[mask]
        n_hat_m = n_hat[mask]
        t_on_m = t_on[mask]
    else:
        n_hat_m = n_hat
        t_on_m = t_on
    x_off.requires_grad_(True)
    f_off = model(x_off)
    v_off = energy_from_f(f_off)

    grad = torch.autograd.grad(
        v_off.sum(), x_off, create_graph=True, retain_graph=True
    )[0]
    grad_norm = torch.norm(grad, dim=1, keepdim=True) + cfg.eps

    w = thickness_weight(t_on_m, cfg)
    loss_dist = (w * (torch.abs(f_off) - torch.abs(s)) ** 2).mean()

    cos = (grad * n_hat_m).sum(dim=1, keepdim=True) / grad_norm
    # Downweight direction loss in thick/unstable regions
    loss_dir = (w * torch.clamp(1.0 - torch.abs(cos), min=0.0)).mean()

    loss_pl = torch.clamp(cfg.beta_pl * v_off - (grad_norm ** 2), min=0.0).mean()

    # Local smoothness: compare gradients at nearby points
    noise = torch.randn_like(x_off) * cfg.smooth_sigma
    x_off_2 = x_off + noise
    x_off_2.requires_grad_(True)
    v_off_2 = energy_from_f(model(x_off_2))
    grad_2 = torch.autograd.grad(v_off_2.sum(), x_off_2, create_graph=True)[0]

    diff_g = grad - grad_2
    diff_x = x_off - x_off_2
    denom = (diff_x * diff_x).sum(dim=1, keepdim=True) + cfg.eps
    loss_smooth = (diff_g * diff_g).sum(dim=1, keepdim=True) / denom
    loss_smooth = (w * loss_smooth).mean()
    # Thinness: use the same thickness weight to favor thin regions
    loss_thin = ((f_on ** 2) / (t_on + cfg.eps)).mean()

    stats = {
        "loss_on": float(loss_on.detach().cpu()),
        "loss_denoise": float(loss_denoise.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_dist": float(loss_dist.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_pl": float(loss_pl.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_thin": float(loss_thin.detach().cpu()),
    }
    parts = {
        "loss_on": loss_on,
        "loss_denoise": loss_denoise,
        "loss_recon": loss_recon,
        "loss_dist": loss_dist,
        "loss_dir": loss_dir,
        "loss_pl": loss_pl,
        "loss_smooth": loss_smooth,
        "loss_thin": loss_thin,
    }
    return parts, stats


def train_with_data(
    cfg: Config, x: np.ndarray
) -> Tuple[
    nn.Module,
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, List[float]],
]:
    k_use = effective_k(cfg, len(x))
    n_hat, quality, thickness = knn_normals_with_quality(x, k_use)
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_hat)
    t_t = torch.from_numpy(thickness).unsqueeze(1)
    idx_t = torch.arange(len(x), dtype=torch.long)

    ds = TensorDataset(x_t, n_t, t_t, idx_t)
    use_cuda = cfg.device == "cuda"
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0,
    )

    model = MLP(in_dim=x.shape[1], hidden=cfg.hidden, depth=cfg.depth).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    last_stats: Dict[str, float] = {}
    history: Dict[str, List[float]] = {
        "loss_on": [],
        "loss_denoise": [],
        "loss_recon": [],
        "loss_dist": [],
        "loss_dir": [],
        "loss_pl": [],
        "loss_smooth": [],
        "loss_thin": [],
        "loss_total": [],
        "w_loss_on": [],
        "w_loss_denoise": [],
        "w_loss_recon": [],
        "w_loss_dist": [],
        "w_loss_dir": [],
        "w_loss_pl": [],
        "w_loss_smooth": [],
        "w_loss_thin": [],
    }
    for epoch in range(cfg.epochs):
        x_ref = x_t.to(cfg.device)
        for xb, nb, tb, idxb in dl:
            xb = xb.to(cfg.device)
            nb = nb.to(cfg.device)
            tb = tb.to(cfg.device)
            idxb = idxb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            parts, stats = compute_losses(model, xb, nb, tb, x_ref, idxb, cfg)
            warm = min(1.0, (epoch + 1) / max(cfg.warmup_epochs, 1))
            lam_pl = cfg.lam_pl * warm
            lam_smooth = cfg.lam_smooth * warm
            loss = (
                cfg.lam_on * parts["loss_on"]
                + cfg.lam_denoise * parts["loss_denoise"]
                + cfg.lam_recon * parts["loss_recon"]
                + cfg.lam_dist * parts["loss_dist"]
                + cfg.lam_dir * parts["loss_dir"]
                + lam_pl * parts["loss_pl"]
                + lam_smooth * parts["loss_smooth"]
                + cfg.lam_thin * parts["loss_thin"]
            )
            stats["loss_total"] = float(loss.detach().cpu())
            stats["w_loss_on"] = float((cfg.lam_on * parts["loss_on"]).detach().cpu())
            stats["w_loss_denoise"] = float(
                (cfg.lam_denoise * parts["loss_denoise"]).detach().cpu()
            )
            stats["w_loss_recon"] = float(
                (cfg.lam_recon * parts["loss_recon"]).detach().cpu()
            )
            stats["w_loss_dist"] = float(
                (cfg.lam_dist * parts["loss_dist"]).detach().cpu()
            )
            stats["w_loss_dir"] = float((cfg.lam_dir * parts["loss_dir"]).detach().cpu())
            stats["w_loss_pl"] = float((lam_pl * parts["loss_pl"]).detach().cpu())
            stats["w_loss_smooth"] = float(
                (lam_smooth * parts["loss_smooth"]).detach().cpu()
            )
            stats["w_loss_thin"] = float(
                (cfg.lam_thin * parts["loss_thin"]).detach().cpu()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_stats = stats
        if last_stats:
            for k in history:
                history[k].append(last_stats[k])
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch+1:04d}  loss={last_stats['loss_total']:.6f}")

    return model, last_stats, n_hat, quality, thickness, history


def project_points(
    model: nn.Module, x0: torch.Tensor, cfg: Config
) -> Tuple[torch.Tensor, int]:
    x = x0.clone()
    for k in range(cfg.proj_steps):
        x.requires_grad_(True)
        with torch.enable_grad():
            v = energy_from_f(model(x))
            grad = torch.autograd.grad(v.sum(), x)[0]
        grad_norm = torch.norm(grad, dim=1, keepdim=True)
        x_next = x - cfg.proj_alpha * grad
        step_norm = torch.norm(x_next - x, dim=1, keepdim=True)
        if (
            torch.all(grad_norm < cfg.tau_g)
            or torch.all(v < cfg.tau_v)
            or torch.all(step_norm < cfg.tau_x)
        ):
            return x_next.detach(), k + 1
        x = x_next.detach()
    return x.detach(), cfg.proj_steps


def project_trajectory(
    model: nn.Module, x0: torch.Tensor, cfg: Config
) -> Tuple[torch.Tensor, int]:
    traj = []
    x = x0.clone()
    for k in range(cfg.proj_steps):
        x.requires_grad_(True)
        with torch.enable_grad():
            v = energy_from_f(model(x))
            grad = torch.autograd.grad(v.sum(), x)[0]
        x_next = x - cfg.proj_alpha * grad
        if k == 0:
            traj.append(x.detach())
        traj.append(x_next.detach())
        grad_norm = torch.norm(grad, dim=1, keepdim=True)
        step_norm = torch.norm(x_next - x, dim=1, keepdim=True)
        if (
            torch.all(grad_norm < cfg.tau_g)
            or torch.all(v < cfg.tau_v)
            or torch.all(step_norm < cfg.tau_x)
        ):
            return torch.stack(traj, dim=0), k + 1
        x = x_next.detach()
    return torch.stack(traj, dim=0), cfg.proj_steps


def evaluate(
    model: nn.Module, x_train: np.ndarray, grid: np.ndarray, cfg: Config
) -> Dict[str, float]:
    model.eval()
    x_t = torch.from_numpy(x_train).to(cfg.device)
    f = model(x_t)
    v = energy_from_f(f).detach().cpu().numpy().reshape(-1)
    on_mean = float(np.mean(v))

    # Correlation between V and true distance squared for random off-manifold points
    x_anchor = x_train[np.random.choice(len(x_train), size=256, replace=True)]
    delta = np.random.randn(*x_anchor.shape).astype(np.float32) * float(cfg.sigmas[0])
    x_off = x_anchor + delta
    d_true = true_distance(x_off, grid)
    x_off_t = torch.from_numpy(x_off).to(cfg.device)
    v_off = energy_from_f(model(x_off_t)).detach().cpu().numpy().reshape(-1)
    corr = float(np.corrcoef(v_off, d_true ** 2)[0, 1])

    return {"on_mean_v": on_mean, "corr_v_d2": corr}


def train_baseline(
    cfg: Config, mode: str, x: np.ndarray, n_hat: np.ndarray
) -> Tuple[nn.Module, Dict[str, float], Dict[str, List[float]]]:
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_hat)
    idx_t = torch.arange(len(x), dtype=torch.long)

    ds = TensorDataset(x_t, n_t, idx_t)
    use_cuda = cfg.device == "cuda"
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0,
    )

    model = MLP(in_dim=x.shape[1], hidden=cfg.hidden, depth=cfg.depth).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: Dict[str, List[float]] = {
        "loss_on": [],
        "loss_off": [],
        "loss_total": [],
        "w_loss_on": [],
        "w_loss_off": [],
    }
    last_stats: Dict[str, float] = {}
    for epoch in range(cfg.epochs):
        x_ref = x_t.to(cfg.device)
        for xb, nb, idxb in dl:
            xb = xb.to(cfg.device)
            nb = nb.to(cfg.device)
            idxb = idxb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            h_on = model(xb)
            loss_on = (h_on ** 2).mean()

            x_off, s = sample_off_manifold(xb, nb, cfg.sigmas)
            k_accept = effective_k_accept(cfg, x_ref.shape[0])
            mask = filter_off_by_knn(x_off, x_ref, idxb, k_accept)
            if mask.sum() > 0:
                x_off = x_off[mask]
                s = s[mask]
            h_off = model(x_off)
            if mode == "margin":
                loss_off = torch.clamp(cfg.margin - h_off, min=0.0).mean()
            elif mode == "delta":
                target = torch.abs(s)
                loss_off = ((h_off - target) ** 2).mean()
            else:
                raise ValueError(f"unknown baseline mode: {mode}")

            loss = cfg.baseline_w_on * loss_on + cfg.baseline_w_off * loss_off
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_stats = {
                "loss_on": float(loss_on.detach().cpu()),
                "loss_off": float(loss_off.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
                "w_loss_on": float((cfg.baseline_w_on * loss_on).detach().cpu()),
                "w_loss_off": float((cfg.baseline_w_off * loss_off).detach().cpu()),
            }
        if last_stats:
            for k in history:
                history[k].append(last_stats[k])
        if (epoch + 1) % 500 == 0:
            print(f"[{mode}] epoch {epoch+1:04d}  loss={last_stats['loss_total']:.6f}")

    return model, last_stats, history


def plot_contour_and_trajectory(
    model: nn.Module,
    x_train: np.ndarray,
    x0: np.ndarray,
    traj: np.ndarray,
    cfg: Config,
    out_path: str,
    title: str,
    grid: np.ndarray | None = None,
) -> None:
    if x_train.shape[1] == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        if grid is not None:
            m = int(math.sqrt(len(grid)))
            if m * m == len(grid):
                gx = grid[:, 0].reshape(m, m)
                gy = grid[:, 1].reshape(m, m)
                gz = grid[:, 2].reshape(m, m)
                ax.plot_surface(
                    gx, gy, gz, color="lightgray", alpha=0.25, linewidth=0
                )
        ax.scatter(
            x_train[:, 0], x_train[:, 1], x_train[:, 2], s=8, alpha=0.5, c="gray"
        )
        for i in range(traj.shape[1]):
            ax.plot(
                traj[:, i, 0],
                traj[:, i, 1],
                traj[:, i, 2],
                color="red",
                linewidth=0.8,
            )
            ax.scatter(
                traj[:, i, 0],
                traj[:, i, 1],
                traj[:, i, 2],
                color="red",
                s=8,
                alpha=0.8,
            )
            ax.scatter(x0[i, 0], x0[i, 1], x0[i, 2], color="red", s=30)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -3.0, 3.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(cfg.device)
    with torch.no_grad():
        f_grid = model(grid_t).cpu().numpy().reshape(xx.shape)
        v = (0.5 * (f_grid ** 2)).reshape(xx.shape)
    v_max = float(np.percentile(v, 95))
    base = np.linspace(0.0, math.sqrt(max(v_max, 1e-12)), 15)
    levels = base ** 2

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    # Zero level set: f(x)=0, not V(x)=0 (numerically more robust)
    eps = cfg.zero_level_eps
    plt.contourf(
        xx,
        yy,
        f_grid,
        levels=[-eps, eps],
        colors=["dimgray"],
        alpha=0.35,
    )
    plt.scatter(
        x_train[:, 0], x_train[:, 1], s=8, alpha=0.6, label="data", zorder=3
    )
    for i in range(traj.shape[1]):
        plt.scatter(x0[i, 0], x0[i, 1], c="red", s=30, label=None)
        plt.plot(traj[:, i, 0], traj[:, i, 1], "-", color="red", linewidth=0.8)
        plt.scatter(traj[:, i, 0], traj[:, i, 1], c="red", s=4, alpha=0.8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distance_curves(
    traj: np.ndarray, grid: np.ndarray, out_path: str, title: str
) -> None:
    n_steps, n_pts, _ = traj.shape
    final = traj[-1]
    d_learn = []
    d_true = []
    for k in range(n_steps):
        xk = traj[k]
        d_true.append(true_distance(xk, grid))
        d_learn.append(np.linalg.norm(xk - final, axis=1))
    d_true = np.stack(d_true, axis=0)
    d_learn = np.stack(d_learn, axis=0)

    plt.figure(figsize=(8, 5))
    for i in range(n_pts):
        plt.plot(d_true[:, i], label=f"true d (pt{i+1})")
    for i in range(n_pts):
        plt.plot(d_learn[:, i], "--", label=f"learned d (pt{i+1})")
    plt.xlabel("iteration")
    plt.ylabel("distance")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_normal_quality(
    x: np.ndarray,
    quality: np.ndarray,
    out_path: str,
    title: str,
    label: str,
    grid: np.ndarray | None = None,
) -> None:
    if x.shape[1] == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        if grid is not None:
            m = int(math.sqrt(len(grid)))
            if m * m == len(grid):
                gx = grid[:, 0].reshape(m, m)
                gy = grid[:, 1].reshape(m, m)
                gz = grid[:, 2].reshape(m, m)
                ax.plot_surface(
                    gx, gy, gz, color="lightgray", alpha=0.2, linewidth=0
                )
        sc = ax.scatter(
            x[:, 0], x[:, 1], x[:, 2], c=quality, cmap="viridis", s=12
        )
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x[:, 0], x[:, 1], c=quality, cmap="viridis", s=12)
    plt.colorbar(sc, label=label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_v_vs_dtrue(
    model: nn.Module, x_train: np.ndarray, grid: np.ndarray, cfg: Config, out_path: str, title: str
) -> None:
    n = min(512, len(x_train))
    idx = np.random.choice(len(x_train), size=n, replace=True)
    x_anchor = x_train[idx]
    sigma = float(max(cfg.sigmas))
    delta = np.random.randn(*x_anchor.shape).astype(np.float32) * sigma
    x_off = x_anchor + delta
    d_true = true_distance(x_off, grid)
    with torch.no_grad():
        v = energy_from_f(model(torch.from_numpy(x_off).to(cfg.device))).cpu().numpy().reshape(-1)
    plt.figure(figsize=(6, 5))
    plt.scatter(d_true ** 2, v, s=10, alpha=0.6)
    x = d_true ** 2
    y = v
    denom = float(np.dot(x, x)) + 1e-12
    slope = float(np.dot(x, y) / denom)
    resid = y - slope * x
    rel = np.abs(resid) / (x + 1e-6)
    rel_q = float(np.quantile(rel, 0.9))
    max_val = float(max(np.max(x), np.max(y)))
    plt.plot([0, max_val], [0, slope * max_val], "k--", linewidth=1)
    plt.plot([0, max_val], [0, (slope + rel_q) * max_val], "k:", linewidth=1)
    plt.plot([0, max_val], [0, max(slope - rel_q, 0.0) * max_val], "k:", linewidth=1)
    slope_hi = slope + rel_q
    slope_lo = max(slope - rel_q, 0.0)
    plt.text(
        0.02,
        0.98,
        (
            f"slope={slope:.3f}\n"
            f"slope_lo={slope_lo:.3f}\n"
            f"slope_hi={slope_hi:.3f}\n"
            f"cone q=0.9 (rel={rel_q:.3f})"
        ),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    plt.xlabel("d_true^2")
    plt.ylabel("V(x)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_knn_normals(
    x: np.ndarray,
    idx_list: List[int],
    k: int,
    out_path: str,
    title: str,
    cfg: Config,
    grid: np.ndarray | None = None,
) -> None:
    d2 = pairwise_sqdist(x, x)
    k_accept = effective_k_accept(cfg, len(x))
    if x.shape[1] == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        if grid is not None:
            m = int(math.sqrt(len(grid)))
            if m * m == len(grid):
                gx = grid[:, 0].reshape(m, m)
                gy = grid[:, 1].reshape(m, m)
                gz = grid[:, 2].reshape(m, m)
                ax.plot_surface(
                    gx, gy, gz, color="lightgray", alpha=0.2, linewidth=0
                )
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=10, alpha=0.4, color="gray")
    else:
        plt.figure(figsize=(6, 5))
        plt.scatter(x[:, 0], x[:, 1], s=10, alpha=0.4, color="gray")
    for idx in idx_list:
        nn_idx = np.argsort(d2, axis=1)[idx, 1 : k + 1]
        nbrs = x[nn_idx]
        center = nbrs.mean(axis=0, keepdims=True)
        y = nbrs - center
        cov = (y.T @ y) / max(k - 1, 1)
        w, v = np.linalg.eigh(cov)
        nvec = v[:, np.argmin(w)]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)

        if x.shape[1] == 3:
            ax.scatter(nbrs[:, 0], nbrs[:, 1], nbrs[:, 2], s=25, color="blue")
            ax.scatter(x[idx, 0], x[idx, 1], x[idx, 2], s=40, color="red")
        else:
            plt.scatter(nbrs[:, 0], nbrs[:, 1], s=25, color="blue")
            plt.scatter(x[idx, 0], x[idx, 1], s=40, color="red")
        # Sample delta points along the normal direction
        m = 30
        sigmas = np.array(cfg.sigmas, dtype=np.float32)
        sigma = np.random.choice(sigmas, size=(m, 1))
        s = np.random.randn(m, 1).astype(np.float32) * sigma
        delta_pts = x[idx : idx + 1] + s * nvec.reshape(1, -1)
        d2_off = pairwise_sqdist(delta_pts, x)
        nn_idx = np.argsort(d2_off, axis=1)[:, :k_accept]
        mask = (nn_idx == idx).any(axis=1)
        pass_pts = delta_pts[mask]
        fail_pts = delta_pts[~mask]
        if x.shape[1] == 3:
            if len(pass_pts) > 0:
                ax.scatter(
                    pass_pts[:, 0],
                    pass_pts[:, 1],
                    pass_pts[:, 2],
                    s=12,
                    color="green",
                    alpha=0.5,
                )
            if len(fail_pts) > 0:
                ax.scatter(
                    fail_pts[:, 0],
                    fail_pts[:, 1],
                    fail_pts[:, 2],
                    s=12,
                    color="orange",
                    alpha=0.5,
                )
        else:
            if len(pass_pts) > 0:
                plt.scatter(pass_pts[:, 0], pass_pts[:, 1], s=12, color="green", alpha=0.5)
            if len(fail_pts) > 0:
                plt.scatter(fail_pts[:, 0], fail_pts[:, 1], s=12, color="orange", alpha=0.5)
        scale = 0.4
        if x.shape[1] == 3:
            ax.quiver(
                x[idx, 0],
                x[idx, 1],
                x[idx, 2],
                nvec[0],
                nvec[1],
                nvec[2],
                length=scale,
                color="red",
            )
        else:
            plt.arrow(
                x[idx, 0],
                x[idx, 1],
                nvec[0] * scale,
                nvec[1] * scale,
                head_width=0.05,
                head_length=0.08,
                color="red",
                length_includes_head=True,
            )
    if x.shape[1] == 3:
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_knn_filter(
    x: np.ndarray,
    n_hat: np.ndarray,
    cfg: Config,
    out_path: str,
    title: str,
    grid: np.ndarray | None = None,
) -> None:
    k_accept = effective_k_accept(cfg, len(x))
    idx_list = np.arange(len(x))

    if x.shape[1] == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        if grid is not None:
            m = int(math.sqrt(len(grid)))
            if m * m == len(grid):
                gx = grid[:, 0].reshape(m, m)
                gy = grid[:, 1].reshape(m, m)
                gz = grid[:, 2].reshape(m, m)
                ax.plot_surface(
                    gx, gy, gz, color="lightgray", alpha=0.2, linewidth=0
                )
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=8, alpha=0.3, color="gray")
    else:
        plt.figure(figsize=(7, 5))
        plt.scatter(x[:, 0], x[:, 1], s=8, alpha=0.3, color="gray")

    sigmas = np.array(cfg.sigmas, dtype=np.float32)
    for idx in idx_list:
        m = 30
        sigma = np.random.choice(sigmas, size=(m, 1))
        s = np.random.randn(m, 1).astype(np.float32) * sigma
        delta_pts = x[idx : idx + 1] + s * n_hat[idx].reshape(1, -1)
        d2_off = pairwise_sqdist(delta_pts, x)
        nn_idx = np.argsort(d2_off, axis=1)[:, :k_accept]
        mask = (nn_idx == idx).any(axis=1)
        pass_pts = delta_pts[mask]
        fail_pts = delta_pts[~mask]
        if x.shape[1] == 3:
            if len(pass_pts) > 0:
                ax.scatter(
                    pass_pts[:, 0],
                    pass_pts[:, 1],
                    pass_pts[:, 2],
                    s=10,
                    color="green",
                    alpha=0.5,
                )
            if len(fail_pts) > 0:
                ax.scatter(
                    fail_pts[:, 0],
                    fail_pts[:, 1],
                    fail_pts[:, 2],
                    s=10,
                    color="orange",
                    alpha=0.5,
                )
        else:
            if len(pass_pts) > 0:
                plt.scatter(pass_pts[:, 0], pass_pts[:, 1], s=10, color="green", alpha=0.5)
            if len(fail_pts) > 0:
                plt.scatter(fail_pts[:, 0], fail_pts[:, 1], s=10, color="orange", alpha=0.5)

    if x.shape[1] == 3:
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_loss_curves(
    history: Dict[str, List[float]], out_path: str, title: str, cfg: Config
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    alpha = cfg.loss_ema_alpha

    def ema_series(values: List[float]) -> List[float]:
        if not values:
            return []
        ema = [values[0]]
        for val in values[1:]:
            ema.append(alpha * ema[-1] + (1 - alpha) * val)
        return ema

    for k, v in history.items():
        if len(v) == 0:
            continue
        ema = ema_series(v)
        if k.startswith("w_"):
            axes[1].plot(ema, label=k)
        else:
            axes[0].plot(ema, label=k)

    axes[0].set_title("Unweighted")
    axes[1].set_title("Weighted")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(ncol=2, fontsize=7)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(7)
    x0_2d = rng.uniform([-4.0, -3.0], [4.0, 3.0], size=(32, 2)).astype(np.float32)
    x0_3d = rng.uniform([-2.5, -2.5, -2.0], [2.5, 2.5, 2.0], size=(16, 3)).astype(np.float32)
    datasets = [
        # "figure_eight",
        "ellipse",
        # "discontinuous",
        # "noise_only",
        # "sparse_only",
        # "high_freq_knot",
        "hetero_noise",
        "double_valley",
        # "hairpin",
    ]
    methods = [
        "energy",
        # "margin",
        # "delta",
    ]
    output_root = "outputs_levelset_datasets"
    os.makedirs(output_root, exist_ok=True)

    for name in datasets:
        set_seed(cfg.seed)
        x_train, grid = generate_dataset(name, cfg)
        model = None
        stats = {}
        history = {}
        if "energy" in methods:
            model, stats, n_hat, quality, thickness, history = train_with_data(
                cfg, x_train
            )
        else:
            k_use = effective_k(cfg, len(x_train))
            n_hat, quality, thickness = knn_normals_with_quality(x_train, k_use)

        margin_model = None
        margin_stats = {}
        margin_history = {}
        if "margin" in methods:
            margin_model, margin_stats, margin_history = train_baseline(
                cfg, mode="margin", x=x_train, n_hat=n_hat
            )

        delta_model = None
        delta_stats = {}
        delta_history = {}
        if "delta" in methods:
            delta_model, delta_stats, delta_history = train_baseline(
                cfg, mode="delta", x=x_train, n_hat=n_hat
            )

        metrics = evaluate(model, x_train, grid, cfg) if model is not None else {}

        out_dir = os.path.join(output_root, name)
        os.makedirs(out_dir, exist_ok=True)

        x0 = x0_3d if x_train.shape[1] == 3 else x0_2d
        x0_t = torch.from_numpy(x0).to(cfg.device)
        if model is not None:
            traj_t, steps = project_trajectory(model, x0_t, cfg)
            traj = traj_t.cpu().numpy()
        else:
            traj = None
            steps = 0

        if model is not None and traj is not None:
            plot_contour_and_trajectory(
                model,
                x_train,
                x0,
                traj,
                cfg,
                out_path=os.path.join(out_dir, "energy_contour_traj.png"),
                title=f"{name}: Energy Model",
                grid=grid,
            )
            plot_distance_curves(
                traj,
                grid,
                out_path=os.path.join(out_dir, "energy_distance_curves.png"),
                title=f"{name}: Energy Model Distances",
            )
            plot_loss_curves(
                history,
                out_path=os.path.join(out_dir, "energy_loss_curves.png"),
                title=f"{name}: Energy Model Losses",
                cfg=cfg,
            )
            plot_v_vs_dtrue(
                model,
                x_train,
                grid,
                cfg,
                out_path=os.path.join(out_dir, "v_vs_dtrue.png"),
                title=f"{name}: V vs d_true^2",
            )
        plot_normal_quality(
            x_train,
            quality,
            out_path=os.path.join(out_dir, "normal_quality.png"),
            title=f"{name}: Normal Quality",
            label="lambda_min / lambda_max",
            grid=grid,
        )
        plot_normal_quality(
            x_train,
            thickness,
            out_path=os.path.join(out_dir, "normal_thickness.png"),
            title=f"{name}: Normal Thickness",
            label="sqrt(mean(normal_proj^2))",
            grid=grid,
        )
        k_use = effective_k(cfg, len(x_train))
        idx_list = [
            int(len(x_train) * 0.2),
            int(len(x_train) * 0.5),
            int(len(x_train) * 0.8),
        ]
        plot_knn_normals(
            x_train,
            idx_list=idx_list,
            k=k_use,
            out_path=os.path.join(out_dir, "knn_normal.png"),
            title=f"{name}: KNN + Normal",
            cfg=cfg,
            grid=grid,
        )
        plot_knn_filter(
            x_train,
            n_hat,
            cfg,
            out_path=os.path.join(out_dir, "knn_filter.png"),
            title=f"{name}: KNN Filter (pass=green, fail=orange)",
            grid=grid,
        )

        if margin_model is not None:
            margin_traj, _ = project_trajectory(margin_model, x0_t, cfg)
            margin_traj = margin_traj.cpu().numpy()
            plot_contour_and_trajectory(
                margin_model,
                x_train,
                x0,
                margin_traj,
                cfg,
                out_path=os.path.join(out_dir, "margin_contour_traj.png"),
                title=f"{name}: Margin Baseline",
                grid=grid,
            )
            plot_distance_curves(
                margin_traj,
                grid,
                out_path=os.path.join(out_dir, "margin_distance_curves.png"),
                title=f"{name}: Margin Baseline Distances",
            )
            plot_loss_curves(
                margin_history,
                out_path=os.path.join(out_dir, "margin_loss_curves.png"),
                title=f"{name}: Margin Baseline Losses",
                cfg=cfg,
            )

        if delta_model is not None:
            delta_traj, _ = project_trajectory(delta_model, x0_t, cfg)
            delta_traj = delta_traj.cpu().numpy()
            plot_contour_and_trajectory(
                delta_model,
                x_train,
                x0,
                delta_traj,
                cfg,
                out_path=os.path.join(out_dir, "delta_contour_traj.png"),
                title=f"{name}: Delta Baseline",
                grid=grid,
            )
            plot_distance_curves(
                delta_traj,
                grid,
                out_path=os.path.join(out_dir, "delta_distance_curves.png"),
                title=f"{name}: Delta Baseline Distances",
            )
            plot_loss_curves(
                delta_history,
                out_path=os.path.join(out_dir, "delta_loss_curves.png"),
                title=f"{name}: Delta Baseline Losses",
                cfg=cfg,
            )

        print(f"[{name}] final losses:", stats)
        print(f"[{name}] eval metrics:", metrics)
        print(f"[{name}] projection steps:", steps)
        if margin_stats:
            print(f"[{name}] baseline margin losses:", margin_stats)
        if delta_stats:
            print(f"[{name}] baseline delta losses:", delta_stats)


if __name__ == "__main__":
    main()
