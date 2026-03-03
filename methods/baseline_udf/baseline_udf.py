"""Simple UDF baselines (margin/delta) for equality-constraint manifolds."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets.constraint_datasets import set_seed
from evaluator.evaluator import (
    compute_eps_stop,
    resolve_eval_cfg,
    sample_eval_seed_points,
)
from core.dataset_resolve import resolve_dataset
from core.eval_runner import run_eval_metrics
from core.kinematics import (
    is_arm_dataset as _is_arm_dataset,
    planar_fk as _planar_fk,
    spatial_fk_n3 as _spatial_fk_n3,
    spatial_fk_n4 as _spatial_fk_n4,
    workspace_embed_for_eval as shared_workspace_embed_for_eval,
    wrap_np_pi as _wrap_np_pi,
)
from core.projection import (
    project_points_with_steps_numpy,
    project_trajectory_numpy,
    project_trajectory_tensor,
)
from core.mlp import MLP
from core.planner import plan_path
from methods.baseline_udf.plots import (
    plot_contour_and_trajectory,
    plot_contour_only,
    plot_distance_curves,
    plot_knn_normals,
    plot_loss_curves,
    plot_planned_paths,
    plot_planned_paths_off,
    plot_worst_distance_contour,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

@dataclass
class Config:
    seed: int = 721  # random seed
    device: str = "auto"  # "auto", "cpu", or "cuda"

    n_train: int = 512  # training points per dataset (dataset-layer config should override)
    n_grid: int = 4096  # grid points for GT manifold

    # KNN sizes
    knn_norm_estimation_ratio: float = 0.08  # ratio for normal-estimation knn points
    knn_norm_estimation_min_points: int = 4  # min points for normal-estimation knn

    sigmas: Tuple[float, ...] = (0.2)  # off-surface sigma list
    off_sigma_mode: str = "list"  # "list" or "max"
    off_bank_size: int = 16  # off-bank batches cached
    off_bank_oversample: int = 6  # oversample factor for off-bank
    off_bank_chunk: int = 64  # chunk size for off-bank
    off_bank_max_rounds: int = 6  # rounds for off-bank fill
    off_curriculum_enable: bool = False  # enable truncated-Gaussian curriculum on off-bank sampling
    off_curriculum_start_ratio: float = 0.2  # initial |s| cap ratio in [0,1]
    off_curriculum_power: float = 1.0  # progress exponent for cap schedule
    off_curriculum_epochs: int = 0  # curriculum epochs; 0 means use full training epochs

    use_knn_filter: bool = False  # filter off points by knn
    knn_off_data_filter_ratio: float = 0.03  # ratio for off-data filter knn points
    knn_off_data_filter_min_points: int = 1  # min points for off-data filter knn

    plan_steps: int = 64  # planner steps
    # Keep legacy fields for compatibility; planner uses eikonal-style fields first.
    plan_iters: int = 400
    plan_lr: float = 0.05
    plan_smooth_weight: float = 1.0
    plan_manifold_weight: float = 400.0
    plan_opt_steps: int = 1240
    plan_opt_lr: float = 0.01
    plan_opt_lam_smooth: float = 0.2
    plan_lam_manifold: float = 1.0
    plan_lam_len_joint: float = 0.40
    plan_trust_scale: float = 0.8
    plan_method: str = "trajectory_opt"  # "trajectory_opt" or "linear_project"
    plan_off_dist: float = 0.6  # off-manifold target offset

    hidden: int = 128  # model width
    depth: int = 3  # model depth
    lr: float = 3e-4  # optimizer learning rate
    lr_decay_step: int = 0  # <=0 disables LR decay
    lr_decay_gamma: float = 1.0  # multiplicative decay factor
    epochs: int = 2000  # training epochs
    train_log_every: int = 50  # logging interval (epochs)
    batch_size: int = 128  # batch size
    baseline_margin_target: float = 1.0  # target |h(x_off)| for margin baseline
    lam_on: float = 1.0  # on-manifold penalty weight
    lam_off: float = 10.0  # off-manifold penalty weight

    # plot
    loss_ema_alpha: float = 0.85  # loss EMA alpha curve

    eps: float = 1e-8  # numeric eps

    proj_alpha: float = 0.3  # projection step size
    proj_steps: int = 100  # projection iterations
    proj_min_steps: int = 30  # minimum projection iterations before early-stop
    projector: dict = field(default_factory=lambda: {"alpha": 0.3, "steps": 100, "min_steps": 30})

    wandb_enable: bool = False  # enable wandb
    wandb_project: str = "equality constraint learning"  # wandb project
    wandb_entity: str = "pby"  # wandb entity
    wandb_run_name: str = ""  # wandb run name
    ur5_backend: str = "analytic"  # "analytic" or "pybullet"


def pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N, d), b: (M, d)
    a = np.nan_to_num(a, copy=False).astype(np.float64)
    b = np.nan_to_num(b, copy=False).astype(np.float64)
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    d2 = np.maximum(d2, 0.0)
    return d2


def _as_sigmas(sigmas: Tuple[float, ...] | float) -> np.ndarray:
    if isinstance(sigmas, (int, float)):
        return np.array([float(sigmas)], dtype=np.float32)
    return np.array(sigmas, dtype=np.float32)


def effective_knn_norm_estimation_points(cfg: Config, n: int) -> int:
    return max(cfg.knn_norm_estimation_min_points, int(round(cfg.knn_norm_estimation_ratio * n)))


def effective_knn_off_data_filter_points(cfg: Config, n: int) -> int:
    return max(
        cfg.knn_off_data_filter_min_points,
        int(round(cfg.knn_off_data_filter_ratio * n)),
    )


def _local_pca_frame(
    neigh: np.ndarray,
    center: np.ndarray,
    cfg: Config | None = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(neigh) == 0:
        d = center.shape[0]
        return (
            np.zeros((d,), dtype=np.float64),
            np.eye(d, dtype=np.float64),
            np.ones((1,), dtype=np.float64),
            neigh,
        )
    mu = np.mean(neigh, axis=0, keepdims=True)
    xc = neigh - mu
    cov = (xc.T @ xc) / max(len(neigh), 1)
    evals, evecs = np.linalg.eigh(cov)
    w = np.full((len(neigh),), 1.0 / max(len(neigh), 1), dtype=np.float64)
    return evals, evecs, w, neigh


def knn_normals(
    x: np.ndarray, k: int, cfg: Config | None = None
) -> np.ndarray:
    n, d = x.shape
    d2 = pairwise_sqdist(x, x)
    normals = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        nbr_idx = np.argsort(d2[i])[1 : k + 1]
        nbrs = x[nbr_idx]
        _, evecs, _, _ = _local_pca_frame(nbrs, x[i], cfg=cfg)
        nvec = evecs[:, 0]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        normals[i] = nvec.astype(np.float32)
    return normals


def true_distance(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        d2 = pairwise_sqdist(x, grid)
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(np.min(d2, axis=1))
    tree = cKDTree(grid)
    d, _ = tree.query(x, k=1)
    return d


def true_projection(x: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        d2 = pairwise_sqdist(x, grid)
        idx = np.argmin(d2, axis=1)
        d = np.sqrt(np.maximum(d2[np.arange(len(x)), idx], 0.0))
        return grid[idx], d
    tree = cKDTree(grid)
    d, idx = tree.query(x, k=1)
    return grid[idx], d


def sample_off_manifold(
    x_on: torch.Tensor,
    n_hat: torch.Tensor,
    sigmas: Tuple[float, ...] | float,
    sigma_mode: str = "list",
    sigma_per_point: torch.Tensor | None = None,
    r_kappa_per_point: torch.Tensor | None = None,
    r_pos_per_point: torch.Tensor | None = None,
    r_neg_per_point: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sigmas = _as_sigmas(sigmas)
    z90 = 1.645  # P(|N(0,1)| <= z90) ~= 0.90
    # Backward compatibility: old configs may still pass "scale".
    if sigma_mode == "scale":
        sigma_mode = "max"
    # Sample scalar offsets along the normal direction only
    if sigma_per_point is not None:
        if r_pos_per_point is not None and r_neg_per_point is not None:
            r_pos = r_pos_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1)
            r_neg = r_neg_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1)
            z = torch.randn_like(x_on[:, :1])
            s = torch.where(z >= 0.0, z * (r_pos / z90), z * (r_neg / z90))
            s = torch.clamp(s, min=-r_neg, max=r_pos)
        else:
            sigma = sigma_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1) / z90
            s = torch.randn_like(x_on[:, :1]) * sigma
            if r_kappa_per_point is not None:
                r_kappa = r_kappa_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1)
                s = torch.clamp(s, min=-r_kappa, max=r_kappa)
    elif sigma_mode == "max":
        sigma = torch.tensor(
            float(np.max(sigmas)), device=x_on.device, dtype=x_on.dtype
        ).reshape(1, 1)
        s = torch.randn_like(x_on[:, :1]) * sigma
    else:
        sigma_choices = torch.tensor(sigmas, device=x_on.device, dtype=x_on.dtype)
        idx = torch.randint(0, len(sigmas), (x_on.shape[0], 1), device=x_on.device)
        sigma = sigma_choices[idx]
        s = torch.randn_like(x_on[:, :1]) * sigma
    x_off = x_on + s * n_hat
    return x_off, s


def filter_off_by_knn(
    x_off: torch.Tensor, x_ref: torch.Tensor, idx_on: torch.Tensor, knn_off_data_filter_points: int
) -> torch.Tensor:
    d2 = torch.cdist(x_off, x_ref) ** 2
    nn_idx = torch.topk(d2, knn_off_data_filter_points, largest=False).indices
    idx_on = idx_on.view(-1, 1)
    mask = (nn_idx == idx_on).any(dim=1)
    return mask


def precompute_off_bank(
    x: np.ndarray,
    n_hat: np.ndarray,
    cfg: Config,
    sigma_per_point: np.ndarray | None,
    r_kappa_per_point: np.ndarray | None,
    r_pos_per_point: np.ndarray | None = None,
    r_neg_per_point: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n, dim = x.shape
    k = max(1, cfg.off_bank_size)
    off_bank = np.zeros((n, k, dim), dtype=np.float32)
    s_bank = np.zeros((n, k, 1), dtype=np.float32)
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_hat)
    idx_t = torch.arange(n, dtype=torch.long)

    chunk = max(1, cfg.off_bank_chunk)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        x_chunk = x_t[start:end]
        n_chunk = n_t[start:end]
        idx_chunk = idx_t[start:end]
        filled = [0 for _ in range(end - start)]
        rounds = 0
        while min(filled) < k and rounds < cfg.off_bank_max_rounds:
            rounds += 1
            n_samples = k * cfg.off_bank_oversample
            x_rep = x_chunk.repeat_interleave(n_samples, dim=0)
            n_rep = n_chunk.repeat_interleave(n_samples, dim=0)
            idx_rep = idx_chunk.repeat_interleave(n_samples, dim=0)
            sigma_rep = None
            r_kappa_rep = None
            r_pos_rep = None
            r_neg_rep = None
            if sigma_per_point is not None:
                sigma_chunk = sigma_per_point[start:end]
                sigma_rep = torch.from_numpy(np.repeat(sigma_chunk, n_samples))
            if r_kappa_per_point is not None:
                r_chunk = r_kappa_per_point[start:end]
                r_kappa_rep = torch.from_numpy(np.repeat(r_chunk, n_samples))
            if r_pos_per_point is not None:
                rp_chunk = r_pos_per_point[start:end]
                r_pos_rep = torch.from_numpy(np.repeat(rp_chunk, n_samples))
            if r_neg_per_point is not None:
                rn_chunk = r_neg_per_point[start:end]
                r_neg_rep = torch.from_numpy(np.repeat(rn_chunk, n_samples))
            x_off_cand, s_cand = sample_off_manifold(
                x_rep,
                n_rep,
                cfg.sigmas,
                cfg.off_sigma_mode,
                sigma_rep,
                r_kappa_rep,
                r_pos_rep,
                r_neg_rep,
            )
            if cfg.use_knn_filter:
                mask = filter_off_by_knn(
                    x_off_cand,
                    x_t,
                    idx_rep,
                    effective_knn_off_data_filter_points(cfg, len(x)),
                )
            else:
                mask = torch.ones(
                    x_off_cand.shape[0],
                    dtype=torch.bool,
                    device=x_off_cand.device,
                )
            mask = mask.view(end - start, n_samples)
            x_off_cand = x_off_cand.view(end - start, n_samples, dim)
            s_cand = s_cand.view(end - start, n_samples, 1)
            for i in range(end - start):
                if filled[i] >= k:
                    continue
                keep = torch.where(mask[i])[0]
                if keep.numel() == 0:
                    continue
                for j in keep.tolist():
                    off_bank[start + i, filled[i]] = x_off_cand[i, j].numpy()
                    s_bank[start + i, filled[i]] = s_cand[i, j].numpy()
                    filled[i] += 1
                    if filled[i] >= k:
                        break
        for i in range(end - start):
            if filled[i] < k:
                need = k - filled[i]
                sigma_one = None
                r_one = None
                rp_one = None
                rn_one = None
                if sigma_per_point is not None:
                    sigma_one = torch.from_numpy(
                        np.array([sigma_per_point[start + i]], dtype=np.float32)
                    )
                if r_kappa_per_point is not None:
                    r_one = torch.from_numpy(
                        np.array([r_kappa_per_point[start + i]], dtype=np.float32)
                    )
                if r_pos_per_point is not None:
                    rp_one = torch.from_numpy(
                        np.array([r_pos_per_point[start + i]], dtype=np.float32)
                    )
                if r_neg_per_point is not None:
                    rn_one = torch.from_numpy(
                        np.array([r_neg_per_point[start + i]], dtype=np.float32)
                    )
                if cfg.use_knn_filter:
                    # Keep filter semantics in fallback: keep sampling until enough
                    # valid points are found.
                    tries = 0
                    max_tries = max(32, cfg.off_bank_max_rounds * 16)
                    while filled[i] < k and tries < max_tries:
                        tries += 1
                        x_off_fill, s_fill = sample_off_manifold(
                            x_chunk[i : i + 1],
                            n_chunk[i : i + 1],
                            cfg.sigmas,
                            cfg.off_sigma_mode,
                            sigma_one,
                            r_one,
                            rp_one,
                            rn_one,
                        )
                        mask_one = filter_off_by_knn(
                            x_off_fill,
                            x_t,
                            idx_chunk[i : i + 1],
                            effective_knn_off_data_filter_points(cfg, len(x)),
                        )
                        if bool(mask_one.item()):
                            off_bank[start + i, filled[i]] = x_off_fill[0].numpy()
                            s_bank[start + i, filled[i]] = s_fill[0].numpy()
                            filled[i] += 1
                    if filled[i] < k:
                        raise RuntimeError(
                            "off_bank fill failed under knn filter; "
                            "try lowering knn_off_data_filter_ratio/min_points "
                            "or increasing off_bank_oversample/off_bank_max_rounds."
                        )
                else:
                    x_off_fill, s_fill = sample_off_manifold(
                        x_chunk[i : i + 1],
                        n_chunk[i : i + 1],
                        cfg.sigmas,
                        cfg.off_sigma_mode,
                        sigma_one,
                        r_one,
                        rp_one,
                        rn_one,
                    )
                    off_bank[start + i, filled[i] :] = (
                        x_off_fill.repeat(need, 1).numpy()
                    )
                    s_bank[start + i, filled[i] :] = s_fill.repeat(need, 1).numpy()
    return off_bank, s_bank


def _off_curriculum_cap_ratio(cfg: Config, epoch: int) -> float | None:
    if not bool(cfg.off_curriculum_enable):
        return None
    total = int(cfg.off_curriculum_epochs) if int(cfg.off_curriculum_epochs) > 0 else int(cfg.epochs)
    total = max(1, total)
    progress = min(1.0, max(0.0, float(epoch + 1) / float(total)))
    power = max(1e-8, float(cfg.off_curriculum_power))
    start = float(np.clip(cfg.off_curriculum_start_ratio, 0.0, 1.0))
    return start + (1.0 - start) * (progress ** power)


def _sample_off_bank_indices(
    s_bank_rows: np.ndarray, cap_ratio: float | None
) -> np.ndarray:
    # s_bank_rows: (B, K, 1), pick one candidate index per row.
    bsz, ksz = s_bank_rows.shape[0], s_bank_rows.shape[1]
    if cap_ratio is None:
        return np.random.randint(0, ksz, size=bsz)

    s_abs = np.abs(s_bank_rows[..., 0])
    row_max = np.maximum(np.max(s_abs, axis=1), 1e-12)
    caps = row_max * float(cap_ratio)
    keep = s_abs <= caps[:, None]
    bank_idx = np.empty(bsz, dtype=np.int64)
    for i in range(bsz):
        cand = np.where(keep[i])[0]
        if cand.size == 0:
            # Fallback: choose closest-to-manifold candidate.
            bank_idx[i] = int(np.argmin(s_abs[i]))
        else:
            bank_idx[i] = int(cand[np.random.randint(0, cand.size)])
    return bank_idx

def _make_project_fn(cfg: Config):
    def _project_points_for_eval(
        model: nn.Module, x0: np.ndarray, eps_stop: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return project_points_with_steps_numpy(
            model,
            x0,
            device=str(cfg.device),
            proj_steps=int(cfg.proj_steps),
            proj_alpha=float(cfg.proj_alpha),
            proj_min_steps=int(getattr(cfg, "proj_min_steps", 0)),
            f_abs_stop=eps_stop,
        )

    return _project_points_for_eval


def train_baseline(
    cfg: Config, mode: str, x: np.ndarray, n_hat: np.ndarray
) -> Tuple[nn.Module, Dict[str, float], Dict[str, List[float]]]:
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_hat)
    idx_t = torch.arange(len(x), dtype=torch.long)
    sigma_per_point = None
    r_kappa_per_point = None
    r_pos_per_point = None
    r_neg_per_point = None
    off_bank = None
    s_bank = None
    if cfg.off_bank_size > 0:
        off_bank, s_bank = precompute_off_bank(
            x,
            n_hat,
            cfg,
            sigma_per_point,
            r_kappa_per_point,
            r_pos_per_point,
            r_neg_per_point,
        )

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

    model = MLP(in_dim=x.shape[1], hidden=cfg.hidden, depth=cfg.depth, out_dim=1).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = None
    if int(cfg.lr_decay_step) > 0 and float(cfg.lr_decay_gamma) < 1.0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(cfg.lr_decay_step),
            gamma=float(cfg.lr_decay_gamma),
        )

    history: Dict[str, List[float]] = {
        "loss_on": [],
        "loss_off": [],
        "loss_total": [],
        "w_loss_on": [],
        "w_loss_off": [],
    }
    last_stats: Dict[str, float] = {}
    x_ref = x_t.to(cfg.device)
    for epoch in range(cfg.epochs):
        for xb, nb, idxb in dl:
            xb = xb.to(cfg.device)
            nb = nb.to(cfg.device)
            idxb = idxb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            # On-manifold penalty: enforce h(x_on)=0.
            h_on = model(xb)
            loss_on = (h_on ** 2).mean()

            if off_bank is not None and s_bank is not None:
                idx_np = idxb.cpu().numpy()
                cap_ratio = _off_curriculum_cap_ratio(cfg, epoch)
                bank_idx = _sample_off_bank_indices(s_bank[idx_np], cap_ratio)
                x_off = torch.from_numpy(off_bank[idx_np, bank_idx]).to(cfg.device)
                s = torch.from_numpy(s_bank[idx_np, bank_idx]).to(cfg.device)
            else:
                sigma_batch = None
                if sigma_per_point is not None:
                    sigma_batch = torch.from_numpy(
                        sigma_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                r_kappa_batch = None
                if r_kappa_per_point is not None:
                    r_kappa_batch = torch.from_numpy(
                        r_kappa_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                r_pos_batch = None
                if r_pos_per_point is not None:
                    r_pos_batch = torch.from_numpy(
                        r_pos_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                r_neg_batch = None
                if r_neg_per_point is not None:
                    r_neg_batch = torch.from_numpy(
                        r_neg_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                x_off, s = sample_off_manifold(
                    xb,
                    nb,
                    cfg.sigmas,
                    cfg.off_sigma_mode,
                    sigma_batch,
                    r_kappa_batch,
                    r_pos_batch,
                    r_neg_batch,
                )
                if cfg.use_knn_filter:
                    knn_off_data_filter_points = effective_knn_off_data_filter_points(cfg, x_ref.shape[0])
                    mask = filter_off_by_knn(x_off, x_ref, idxb, knn_off_data_filter_points)
                    if mask.sum() > 0:
                        x_off = x_off[mask]
                        s = s[mask]
                    else:
                        # Keep filter semantics: no valid off-point this mini-batch.
                        # Still train on on-manifold term.
                        s = None
            if s is None:
                loss_off = torch.zeros((), device=xb.device)
            else:
                # Off-manifold penalty only (simple baseline):
                # - margin mode: |h(x_off)| -> 1
                # - delta mode:  |h(x_off)| -> |delta|
                h_off = model(x_off)
                if mode == "margin":
                    target = torch.full_like(h_off, float(cfg.baseline_margin_target))
                    loss_off = ((torch.abs(h_off) - target) ** 2).mean()
                elif mode == "delta":
                    target = torch.abs(s)
                    loss_off = ((torch.abs(h_off) - target) ** 2).mean()
                else:
                    raise ValueError(f"unknown baseline mode: {mode}")

            loss = cfg.lam_on * loss_on + cfg.lam_off * loss_off
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_stats = {
                "loss_on": float(loss_on.detach().cpu()),
                "loss_off": float(loss_off.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
                "w_loss_on": float((cfg.lam_on * loss_on).detach().cpu()),
                "w_loss_off": float((cfg.lam_off * loss_off).detach().cpu()),
            }
        if last_stats:
            for k in history:
                history[k].append(last_stats[k])
        if ((epoch + 1) % max(1, int(cfg.train_log_every))) == 0 or epoch == 0 or (epoch + 1) == int(cfg.epochs):
            lr_now = float(opt.param_groups[0]["lr"])
            print(
                f"[train] method={mode} ep={epoch+1:4d}/{cfg.epochs} "
                f"| lr={lr_now:.2e} "
                f"| loss={last_stats['loss_total']:.6f} "
                f"| on={last_stats['loss_on']:.6f} "
                f"| off={last_stats['loss_off']:.6f}"
            )
        if scheduler is not None:
            scheduler.step()

    return model, last_stats, history


def main() -> None:
    cfg = Config()
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(7)
    datasets = [
        "2d_sharp_star",
        # "2d_figure_eight",
        # "2d_ellipse",
        # "2d_discontinuous",
        # "2d_noisy_sine",
        # "2d_sparse_sine",
        # "2d_sine",
        # "2d_looped_spiro",

        # "2d_hetero_noise",

    ]
    methods = ["margin", "delta"]
    output_root = "outputs_levelset_datasets"
    os.makedirs(output_root, exist_ok=True)
    eval_results = []
    wb_run = None
    wb_step = 0
    if cfg.wandb_enable:
        if wandb is None:
            print("wandb not available; disable wandb logging.")
        else:
            wb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name or None,
                config=asdict(cfg),
            )

    for name in datasets:
        set_seed(cfg.seed)
        ds = resolve_dataset(
            name,
            cfg,
            optimize_ur5_train_only=True,
            ur5_backend=str(getattr(cfg, "ur5_backend", "analytic")),
        )
        x_train = ds["x_train"]
        grid = ds["grid"]
        eval_cfg_ref = resolve_eval_cfg(
            cfg,
            method_key="margin",
            dataset_name=name,
        )
        vis_vals_ref = dict(cfg.__dict__)
        vis_vals_ref.update(vars(eval_cfg_ref))
        vis_cfg_ref = SimpleNamespace(**vis_vals_ref)
        x_eval = sample_eval_seed_points(x_train, eval_cfg_ref)
        knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x_train))
        n_hat = knn_normals(x_train, knn_norm_estimation_points, cfg)

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

        out_dir = os.path.join(output_root, name)
        os.makedirs(out_dir, exist_ok=True)

        if x_train.shape[1] == 3:
            n_plot = 16
        else:
            n_plot = 128
        n_plot = min(n_plot, len(x_eval))
        x0 = x_eval[:n_plot]
        x0_t = torch.from_numpy(x0).to(cfg.device)
        plan_pairs = None
        plan_pairs_off = None
        if x_train.shape[1] == 2:
            plan_rng = np.random.default_rng(cfg.seed + 77)
            n_pairs = 4
            replace = len(x_eval) < 2 * n_pairs
            picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
            plan_pairs = [
                (x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]])
                for i in range(n_pairs)
            ]
            off_dirs = plan_rng.normal(size=(n_pairs, 2)).astype(np.float32)
            norms = np.linalg.norm(off_dirs, axis=1, keepdims=True) + 1e-8
            off_dirs = off_dirs / norms
            plan_pairs_off = []
            for i in range(n_pairs):
                x0_p = plan_pairs[i][0]
                x1_on = plan_pairs[i][1]
                x1_off = x1_on + off_dirs[i] * cfg.plan_off_dist
                plan_pairs_off.append((x0_p, x1_off))
        steps = 0
        knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x_train))
        n_train = len(x_train)
        idx_list = [
            min(n_train - 1, int(round((n_train - 1) * r)))
            for r in np.linspace(0.05, 0.95, 15)
        ]
        plot_knn_normals(
            x_train,
            idx_list=idx_list,
            k=knn_norm_estimation_points,
            out_path=os.path.join(out_dir, "knn_normal.png"),
            title=f"{name}: KNN + Normal",
            cfg=vis_cfg_ref,
            grid=grid,
            sigma_per_point=None,
        )
        if margin_model is not None:
            eval_cfg_margin = resolve_eval_cfg(cfg, method_key="margin", dataset_name=name)
            vis_vals_margin = dict(cfg.__dict__)
            vis_vals_margin.update(vars(eval_cfg_margin))
            vis_cfg_margin = SimpleNamespace(**vis_vals_margin)
            margin_eps_used = compute_eps_stop(margin_model, x_train, eval_cfg_margin)
            margin_traj, _ = project_trajectory_tensor(
                margin_model,
                x0_t,
                proj_steps=int(cfg.proj_steps),
                proj_alpha=float(cfg.proj_alpha),
                proj_min_steps=int(getattr(cfg, "proj_min_steps", 0)),
                f_abs_stop=margin_eps_used,
            )
            margin_traj = margin_traj.cpu().numpy()
            plot_contour_and_trajectory(
                margin_model,
                x_train,
                x0,
                margin_traj,
                vis_cfg_margin,
                out_path=os.path.join(out_dir, "margin_contour_traj.png"),
                title=f"{name}: Margin Baseline",
                grid=grid,
                zero_level_eps=margin_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
                project_trajectory_fn=lambda m, x0_seed, c, f_abs_stop: (
                    torch.from_numpy(
                        project_trajectory_numpy(
                            m,
                            x0_seed.detach().cpu().numpy(),
                            device=str(c.device),
                            proj_steps=int(c.proj_steps),
                            proj_alpha=float(c.proj_alpha),
                            proj_min_steps=int(getattr(c, "proj_min_steps", 0)),
                            f_abs_stop=f_abs_stop,
                        )
                    ),
                    int(c.proj_steps),
                ),
            )
            plot_distance_curves(
                margin_traj,
                grid,
                out_path=os.path.join(out_dir, "margin_distance_curves.png"),
                title=f"{name}: Margin Baseline Distances",
            )
            plot_worst_distance_contour(
                margin_model,
                x_train,
                grid,
                x0,
                margin_traj,
                vis_cfg_margin,
                out_path=os.path.join(out_dir, "margin_worst_distance_contour.png"),
                title=f"{name}: Margin Worst Distance (top 5%)",
                zero_level_eps=margin_eps_used,
            )
            plot_loss_curves(
                margin_history,
                out_path=os.path.join(out_dir, "margin_loss_curves.png"),
                title=f"{name}: Margin Baseline Losses",
                cfg=vis_cfg_margin,
            )

        if delta_model is not None:
            eval_cfg_delta = resolve_eval_cfg(cfg, method_key="delta", dataset_name=name)
            vis_vals_delta = dict(cfg.__dict__)
            vis_vals_delta.update(vars(eval_cfg_delta))
            vis_cfg_delta = SimpleNamespace(**vis_vals_delta)
            delta_eps_used = compute_eps_stop(delta_model, x_train, eval_cfg_delta)
            delta_traj, _ = project_trajectory_tensor(
                delta_model,
                x0_t,
                proj_steps=int(cfg.proj_steps),
                proj_alpha=float(cfg.proj_alpha),
                proj_min_steps=int(getattr(cfg, "proj_min_steps", 0)),
                f_abs_stop=delta_eps_used,
            )
            delta_traj = delta_traj.cpu().numpy()
            plot_contour_and_trajectory(
                delta_model,
                x_train,
                x0,
                delta_traj,
                vis_cfg_delta,
                out_path=os.path.join(out_dir, "delta_contour_traj.png"),
                title=f"{name}: Delta Baseline",
                grid=grid,
                zero_level_eps=delta_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
                project_trajectory_fn=lambda m, x0_seed, c, f_abs_stop: (
                    torch.from_numpy(
                        project_trajectory_numpy(
                            m,
                            x0_seed.detach().cpu().numpy(),
                            device=str(c.device),
                            proj_steps=int(c.proj_steps),
                            proj_alpha=float(c.proj_alpha),
                            proj_min_steps=int(getattr(c, "proj_min_steps", 0)),
                            f_abs_stop=f_abs_stop,
                        )
                    ),
                    int(c.proj_steps),
                ),
            )
            plot_distance_curves(
                delta_traj,
                grid,
                out_path=os.path.join(out_dir, "delta_distance_curves.png"),
                title=f"{name}: Delta Baseline Distances",
            )
            if plan_pairs is not None:
                plans_proj = []
                plans_constr = []
                plans_proj_off = []
                plans_constr_off = []
                use_linear = str(getattr(cfg, "plan_method", "trajectory_opt")).lower() in (
                    "linear_project",
                    "linear_proj",
                    "projection",
                )
                for x0_p, x1_p in plan_pairs:
                    x0_p = true_projection(x0_p[None, :], grid)[0][0]
                    x1_p = true_projection(x1_p[None, :], grid)[0][0]
                    planned = plan_path(
                        model=delta_model,
                        x_start=x0_p,
                        x_goal=x1_p,
                        cfg=cfg,
                        planner_name=str(getattr(cfg, "plan_method", "trajectory_opt")),
                        n_waypoints=int(cfg.plan_steps + 1),
                        dataset_name=name,
                        periodic_joint=bool(ds.get("periodic_joint", False)),
                        f_abs_stop=delta_eps_used,
                    )
                    if use_linear:
                        plans_proj.append(planned)
                    else:
                        plans_constr.append(planned)
                if plan_pairs_off is not None:
                    for x0_p, x1_p in plan_pairs_off:
                        x0_p = true_projection(x0_p[None, :], grid)[0][0]
                        planned_off = plan_path(
                            model=delta_model,
                            x_start=x0_p,
                            x_goal=x1_p,
                            cfg=cfg,
                            planner_name=str(getattr(cfg, "plan_method", "trajectory_opt")),
                            n_waypoints=int(cfg.plan_steps + 1),
                            dataset_name=name,
                            periodic_joint=bool(ds.get("periodic_joint", False)),
                            f_abs_stop=delta_eps_used,
                            keep_endpoints=False,
                        )
                        if use_linear:
                            plans_proj_off.append(planned_off)
                        else:
                            plans_constr_off.append(planned_off)
                plot_planned_paths(
                    delta_model,
                    x_train,
                    grid,
                    plans_proj,
                    plans_constr,
                    vis_cfg_delta,
                    out_path=os.path.join(out_dir, "delta_planner_paths.png"),
                    title=f"{name}: Delta Planned Paths",
                    zero_level_eps=delta_eps_used,
                )
                if plans_proj_off or plans_constr_off:
                    plot_planned_paths_off(
                        delta_model,
                        x_train,
                        grid,
                        plans_proj_off,
                        plans_constr_off,
                        vis_cfg_delta,
                        out_path=os.path.join(out_dir, "delta_planner_paths_off.png"),
                        title=f"{name}: Delta Planned Paths (Off-Manifold)",
                        zero_level_eps=delta_eps_used,
                    )
            plot_worst_distance_contour(
                delta_model,
                x_train,
                grid,
                x0,
                delta_traj,
                vis_cfg_delta,
                out_path=os.path.join(out_dir, "delta_worst_distance_contour.png"),
                title=f"{name}: Delta Worst Distance (top 5%)",
                zero_level_eps=delta_eps_used,
            )
            plot_loss_curves(
                delta_history,
                out_path=os.path.join(out_dir, "delta_loss_curves.png"),
                title=f"{name}: Delta Baseline Losses",
                cfg=vis_cfg_delta,
            )

        print(f"[{name}] projection steps:", steps)
        if margin_stats:
            print(f"[{name}] baseline margin losses:", margin_stats)
        if delta_stats:
            print(f"[{name}] baseline delta losses:", delta_stats)

        if margin_model is not None:
            post_fn = _wrap_np_pi if _is_arm_dataset(name) else None
            embed_fn = (
                lambda q, _name=name: shared_workspace_embed_for_eval(
                    _name,
                    q,
                    ur5_use_pybullet_n6=(str(getattr(cfg, "ur5_backend", "analytic")).lower() == "pybullet"),
                )
            ) if _is_arm_dataset(name) else None
            project_fn_margin = _make_project_fn(cfg)
            metrics_map, _, _ = run_eval_metrics(
                cfg=cfg,
                method_key="margin",
                dataset_name=name,
                model=margin_model,
                x_train=x_train,
                project_fn=project_fn_margin,
                embed_fn=embed_fn,
                postprocess_fn=post_fn,
            )
            eval_results.append(
                {
                    "dataset": name,
                    "method": "margin",
                    "metrics": metrics_map,
                }
            )
            if wb_run is not None:
                wb_step += 1
                wandb.log(
                    {f"{name}/margin/{k}": v for k, v in metrics_map.items()},
                    step=wb_step,
                )
        if delta_model is not None:
            post_fn = _wrap_np_pi if _is_arm_dataset(name) else None
            embed_fn = (
                lambda q, _name=name: shared_workspace_embed_for_eval(
                    _name,
                    q,
                    ur5_use_pybullet_n6=(str(getattr(cfg, "ur5_backend", "analytic")).lower() == "pybullet"),
                )
            ) if _is_arm_dataset(name) else None
            project_fn_delta = _make_project_fn(cfg)
            metrics_map, _, _ = run_eval_metrics(
                cfg=cfg,
                method_key="delta",
                dataset_name=name,
                model=delta_model,
                x_train=x_train,
                project_fn=project_fn_delta,
                embed_fn=embed_fn,
                postprocess_fn=post_fn,
            )
            eval_results.append(
                {
                    "dataset": name,
                    "method": "delta",
                    "metrics": metrics_map,
                }
            )
            if wb_run is not None:
                wb_step += 1
                wandb.log(
                    {f"{name}/delta/{k}": v for k, v in metrics_map.items()},
                    step=wb_step,
                )

    if eval_results:
        metrics_out = os.path.join(output_root, "metrics.json")
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        metrics_txt = os.path.join(output_root, "metrics.txt")
        lines = []
        lines.append("=== Evaluation Metrics (per dataset) ===")
        print("\n=== Evaluation Metrics (per dataset) ===")
        for entry in eval_results:
            m = entry["metrics"]
            line1 = (
                f"[eval] {entry['dataset']} | method={entry['method']} "
                f"| proj_dist={m['proj_manifold_dist']:.6f} "
                f"| pred_recall={m['pred_recall']:.6f} "
                f"| pred_FPrate={m['pred_FPrate']:.6f} "
                f"| chamfer={m['bidirectional_chamfer']:.6f} "
                f"| gt->learned={m['gt_to_learned_mean']:.6f} "
                f"| learned->gt={m['learned_to_gt_mean']:.6f} "
                f"| space={m.get('dist_space', 'unknown')}"
            )
            line2 = (
                f"[eval] {entry['dataset']} | method={entry['method']} "
                f"| proj_steps={m['proj_steps']:.2f} "
                f"| proj_true_dist={m['proj_true_dist']:.6f} "
                f"| proj_v_residual={m['proj_v_residual']:.6f} "
                f"| eval_eps={m['eval_eps_used']:.6f} "
                f"| pred_precision={m['pred_precision']:.6f}"
            )
            lines.append(line1)
            lines.append(line2)
            print(line1)
            print(line2)
        lines.append("")
        lines.append("=== Evaluation Metrics (mean over datasets) ===")
        print("\n=== Evaluation Metrics (mean over datasets) ===")
        by_method: Dict[str, List[Dict[str, float]]] = {}
        for entry in eval_results:
            by_method.setdefault(entry["method"], []).append(entry["metrics"])
        for method, items in by_method.items():
            num_keys = []
            for k, v in items[0].items():
                if isinstance(v, (int, float, np.floating)):
                    num_keys.append(k)
            avg = {k: float(np.mean([float(it[k]) for it in items])) for k in num_keys}
            line1 = (
                f"[eval-avg] method={method} "
                f"| proj_dist={avg['proj_manifold_dist']:.6f} "
                f"| pred_recall={avg['pred_recall']:.6f} "
                f"| pred_FPrate={avg['pred_FPrate']:.6f} "
                f"| chamfer={avg['bidirectional_chamfer']:.6f} "
                f"| gt->learned={avg['gt_to_learned_mean']:.6f} "
                f"| learned->gt={avg['learned_to_gt_mean']:.6f}"
            )
            line2 = (
                f"[eval-avg] method={method} "
                f"| proj_steps={avg['proj_steps']:.2f} "
                f"| proj_true_dist={avg['proj_true_dist']:.6f} "
                f"| proj_v_residual={avg['proj_v_residual']:.6f} "
                f"| eval_eps={avg['eval_eps_used']:.6f} "
                f"| pred_precision={avg['pred_precision']:.6f}"
            )
            lines.append(line1)
            lines.append(line2)
            print(line1)
            print(line2)
            if wb_run is not None:
                wb_step += 1
                wandb.log(
                    {f"avg/{method}/{k}": v for k, v in avg.items()},
                    step=wb_step,
                )
        with open(metrics_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    if wb_run is not None:
        wb_run.finish()

if __name__ == "__main__":
    from common.unified_experiment import run_one as _run_one_unified

    p = argparse.ArgumentParser(description="baseline_udf direct wrapper (unified runner)")
    p.add_argument("--method", default="margin,delta", help="margin|delta or comma-separated")
    p.add_argument("--dataset", default="2d_sharp_star", help="single or comma-separated datasets")
    p.add_argument("--outdir", default="outputs_unified")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config-root", default="configs")
    p.add_argument("--override", action="append", default=[], help="dotted key=value override")
    p.add_argument("--legacy", action="store_true", help="run legacy in-file main() flow")
    args = p.parse_args()

    if args.legacy:
        main()
    else:
        methods = [m.strip() for m in str(args.method).split(",") if m.strip()]
        datasets = [d.strip() for d in str(args.dataset).split(",") if d.strip()]
        for m in methods:
            for ds_name in datasets:
                print(f"[run] method={m} dataset={ds_name}")
                result, loaded_paths = _run_one_unified(
                    method=m,
                    dataset=ds_name,
                    out_root=str(args.outdir),
                    seed_override=args.seed,
                    config_root=str(args.config_root),
                    cli_overrides=list(args.override),
                )
                mm = result["metrics"]
                print(f"[cfg] loaded_layers={loaded_paths if loaded_paths else '[]'}")
                print(
                    f"[done] method={m} dataset={ds_name} "
                    f"proj_dist={mm.get('proj_manifold_dist', float('nan')):.6f} "
                    f"recall={mm.get('pred_recall', float('nan')):.6f} "
                    f"FPrate={mm.get('pred_FPrate', float('nan')):.6f}"
                )
