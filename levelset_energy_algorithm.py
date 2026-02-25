"""
Standalone implementation of a level-set energy learning algorithm for
equality-constraint manifolds. This file does not import or interact with
any other project modules.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

NORM_PROJ_BASELINE = {
    "figure_eight": {"mean": 0.120565, "std": 0.107640},
    "ellipse": {"mean": 0.081101, "std": 0.125372},
    "noise_only": {"mean": 0.459975, "std": 0.225468},
    "sparse_only": {"mean": 0.264254, "std": 0.239843},
    "looped_spiro": {"mean": 0.190081, "std": 0.200199},
}

@dataclass
class Config:
    seed: int = 721  # random seed
    device: str = "auto"  # "auto", "cpu", or "cuda"

    n_train: int = 256  # training points per dataset
    n_grid: int = 4096  # grid points for GT manifold

    # KNN sizes
    knn_norm_estimation_ratio: float = 0.08  # ratio for normal-estimation knn points
    knn_norm_estimation_min_points: int = 4  # min points for normal-estimation knn
    use_weighted_pca: bool = False  # use weighted PCA for local frame estimation
    use_trimmed_pca: bool = False  # trim high-residual neighbors before final PCA
    trimmed_pca_keep_ratio: float = 0.9  # keep ratio after trimming (e.g., 0.9 drops top 10%)

    use_radius_knn: bool = False  # radius-based knn for normals/curvature
    radius_knn_k: int = 4  # local spacing k for radius knn
    radius_knn_scale: float = 1.5  # radius multiplier on spacing

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

    use_adaptive_sigma: bool = not True  # enable adaptive sigma
    adp_sigma_scale: float = 1  # global scale in r_kappa
    adp_sigma_kappa_exp: float = 1  # exponent for curvature compression
    adp_sigma_r_min: float = 0.01  # min r_kappa clamp
    adp_sigma_r_max: float = 2  # max r_kappa clamp
    adp_sigma_eps: float = 1e-6  # eps for curvature divide
    adp_sigma_asymmetric_enable: bool = False  # enable asymmetric (+/- normal) radii
    adp_sigma_danger_scale: float = 0.6  # radius scale on dangerous side
    adp_sigma_safe_scale: float = 1.0  # radius scale on safe side
    adp_sigma_nonlocal_offset: int = 1  # extra rank beyond local knn for side test

    loss_denoise_every: int = 4  # denoise loss frequency
    loss_smooth_every: int = 4  # smooth loss frequency

    plan_steps: int = 64  # planner steps
    plan_iters: int = 400  # planner iterations
    plan_lr: float = 0.05  # planner learning rate
    plan_smooth_weight: float = 1.0  # planner smooth weight
    plan_manifold_weight: float = 400.0  # planner manifold weight
    plan_off_dist: float = 0.6  # off-manifold target offset

    hidden: int = 128  # model width
    depth: int = 3  # model depth
    lr: float = 3e-4  # optimizer learning rate
    epochs: int = 2000  # training epochs
    batch_size: int = 128  # batch size
    warmup_epochs: int = 500  # warmup for losses

    lam_on: float = 1  # on-manifold loss weight
    lam_dist: float = 0.3  # distance loss weight
    lam_dir: float = 0.0  # direction loss weight

    lam_pl: float = 0.  # projection loss weight
    beta_pl: float = 1  # projection loss beta

    lam_smooth: float = 0.0  # smoothness loss weight
    smooth_sigma: float = 0.01  # smooth perturb sigma

    lam_thin: float = 0.0  # thinness loss weight
    use_thickness_on: bool = False  # use thickness on on-loss

    lam_denoise: float = 0.0  # denoise loss weight
    denoise_step: float = 1.0  # denoise step size
    lam_recon: float = 0.0  # recon loss weight

    margin: float = 0.5  # margin baseline parameter
    baseline_w_on: float = 1.0  # baseline on weight
    baseline_w_off: float = 10.0  # baseline off weight

    # plot
    loss_ema_alpha: float = 0.85  # loss EMA alpha curve

    eps: float = 1e-8  # numeric eps

    proj_alpha: float = 0.3  # projection step size
    proj_steps: int = 200  # projection iterations
    proj_grad_clip: float = 10.0  # projection grad clip

    eval_seed: int = 123  # eval RNG seed
    eval_n: int = 2048*2  # eval point count
    eval_pad_ratio: float = 0.6  # eval box scale ratio
    eval_eps: float = 0.02  # eval f(x) zero threshold (fixed mode)
    eval_eps_mode: str = "quantile"  # "fixed" or "quantile"
    eval_eps_quantile: float = 90.0  # quantile for adaptive eps
    eval_tau_ratio: float = 0.018  # near-band ratio

    wandb_enable: bool = False  # enable wandb
    wandb_project: str = "equality constraint learning"  # wandb project
    wandb_entity: str = "pby"  # wandb entity
    wandb_run_name: str = ""  # wandb run name


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


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return ((a + math.pi) % (2.0 * math.pi) - math.pi).astype(np.float32)


def _sample_rows(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    if len(arr) >= n:
        idx = np.random.choice(len(arr), size=n, replace=False)
    else:
        idx = np.random.choice(len(arr), size=n, replace=True)
    return arr[idx].astype(np.float32)


def _planar_arm_line_n2(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    l1, l2 = 1.0, 0.8
    y_line = 0.3
    t1 = np.linspace(-math.pi, math.pi, max(4 * cfg.n_grid, 4096), dtype=np.float32)
    rhs = (y_line - l1 * np.sin(t1)) / l2
    valid = np.abs(rhs) <= 1.0
    t1v = t1[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    t2a = _wrap_to_pi(a - t1v)
    t2b = _wrap_to_pi((math.pi - a) - t1v)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(t1v), t2a], axis=1),
            np.stack([_wrap_to_pi(t1v), t2b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(x_all, max(1, cfg.n_grid))
    return x_train, grid


def _planar_arm_line_n3(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    l1, l2, l3 = 1.0, 0.8, 0.6
    y_line = 0.35

    n_cand = max(8 * cfg.n_train, 12000)
    t1 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    t2 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    rhs = (y_line - l1 * np.sin(t1) - l2 * np.sin(t1 + t2)) / l3
    valid = np.abs(rhs) <= 1.0
    t1v = t1[valid]
    t2v = t2[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    t12 = t1v + t2v
    t3a = _wrap_to_pi(a - t12)
    t3b = _wrap_to_pi((math.pi - a) - t12)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(t1v), _wrap_to_pi(t2v), t3a], axis=1),
            np.stack([_wrap_to_pi(t1v), _wrap_to_pi(t2v), t3b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    m = max(24, int(math.sqrt(max(cfg.n_grid, 1024))))
    g1 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g2 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    tg1, tg2 = np.meshgrid(g1, g2)
    t1g = tg1.ravel().astype(np.float32)
    t2g = tg2.ravel().astype(np.float32)
    rhs_g = (y_line - l1 * np.sin(t1g) - l2 * np.sin(t1g + t2g)) / l3
    valid_g = np.abs(rhs_g) <= 1.0
    t1gv = t1g[valid_g]
    t2gv = t2g[valid_g]
    ag = np.arcsin(rhs_g[valid_g]).astype(np.float32)
    t12g = t1gv + t2gv
    t3ga = _wrap_to_pi(ag - t12g)
    t3gb = _wrap_to_pi((math.pi - ag) - t12g)
    grid_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(t1gv), _wrap_to_pi(t2gv), t3ga], axis=1),
            np.stack([_wrap_to_pi(t1gv), _wrap_to_pi(t2gv), t3gb], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(grid_all, max(1, cfg.n_grid))
    return x_train, grid


def _spatial_arm_plane_n4(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    # 4-DoF arm (q1 yaw, q2/q3/q4 pitch chain) constrained to z=z_plane.
    l1, l2, l3 = 1.0, 0.8, 0.6
    z_plane = 0.35

    n_cand = max(12 * cfg.n_train, 24000)
    q1 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    q2 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    q3 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)

    z12 = l1 * np.sin(q2) + l2 * np.sin(q2 + q3)
    rhs = (z_plane - z12) / l3
    valid = np.abs(rhs) <= 1.0
    q1v = q1[valid]
    q2v = q2[valid]
    q3v = q3[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    q23 = q2v + q3v
    q4a = _wrap_to_pi(a - q23)
    q4b = _wrap_to_pi((math.pi - a) - q23)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), _wrap_to_pi(q3v), q4a], axis=1),
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), _wrap_to_pi(q3v), q4b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    # Dense candidate grid for planning/visualization.
    m = max(14, int(round(max(2, cfg.n_grid) ** (1.0 / 3.0))))
    g1 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g2 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g3 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    tg1, tg2, tg3 = np.meshgrid(g1, g2, g3, indexing="ij")
    q1g = tg1.ravel().astype(np.float32)
    q2g = tg2.ravel().astype(np.float32)
    q3g = tg3.ravel().astype(np.float32)
    z12g = l1 * np.sin(q2g) + l2 * np.sin(q2g + q3g)
    rhsg = (z_plane - z12g) / l3
    vg = np.abs(rhsg) <= 1.0
    q1gv = q1g[vg]
    q2gv = q2g[vg]
    q3gv = q3g[vg]
    ag = np.arcsin(rhsg[vg]).astype(np.float32)
    q23g = q2gv + q3gv
    q4ga = _wrap_to_pi(ag - q23g)
    q4gb = _wrap_to_pi((math.pi - ag) - q23g)
    grid_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), _wrap_to_pi(q3gv), q4ga], axis=1),
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), _wrap_to_pi(q3gv), q4gb], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(grid_all, max(1, cfg.n_grid))
    return x_train, grid


def _spatial_arm_plane_n3(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    # 3-DoF arm (q1 yaw, q2/q3 pitch chain) constrained to z=z_plane.
    l1, l2 = 1.0, 0.8
    z_plane = 0.35

    n_cand = max(12 * cfg.n_train, 18000)
    q1 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    q2 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)

    rhs = (z_plane - l1 * np.sin(q2)) / l2
    valid = np.abs(rhs) <= 1.0
    q1v = q1[valid]
    q2v = q2[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    q3a = _wrap_to_pi(a - q2v)
    q3b = _wrap_to_pi((math.pi - a) - q2v)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), q3a], axis=1),
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), q3b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    m = max(26, int(round(max(2, cfg.n_grid) ** 0.5)))
    g1 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g2 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    tg1, tg2 = np.meshgrid(g1, g2, indexing="ij")
    q1g = tg1.ravel().astype(np.float32)
    q2g = tg2.ravel().astype(np.float32)
    rhsg = (z_plane - l1 * np.sin(q2g)) / l2
    vg = np.abs(rhsg) <= 1.0
    q1gv = q1g[vg]
    q2gv = q2g[vg]
    ag = np.arcsin(rhsg[vg]).astype(np.float32)
    q3ga = _wrap_to_pi(ag - q2gv)
    q3gb = _wrap_to_pi((math.pi - ag) - q2gv)
    grid_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), q3ga], axis=1),
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), q3gb], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(grid_all, max(1, cfg.n_grid))
    return x_train, grid


def _spatial_arm_circle_n3(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    # 3-DoF arm (q1 yaw, q2/q3 pitch chain) constrained to workspace circle:
    # x^2 + y^2 = r0^2, z = z0  (codim=2 in 3D joint space).
    l1, l2 = 1.0, 0.8
    r0 = 1.25
    z0 = 0.35

    rho2 = r0 * r0 + z0 * z0
    c3 = (rho2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    c3 = float(np.clip(c3, -1.0, 1.0))
    s3_mag = float(np.sqrt(max(0.0, 1.0 - c3 * c3)))

    def _solve_q2_q3(sign: float) -> Tuple[float, float]:
        s3 = float(sign) * s3_mag
        q3 = math.atan2(s3, c3)
        q2 = math.atan2(z0, r0) - math.atan2(l2 * s3, l1 + l2 * c3)
        q2 = float(_wrap_to_pi(np.array([q2], dtype=np.float32))[0])
        q3 = float(_wrap_to_pi(np.array([q3], dtype=np.float32))[0])
        return q2, q3

    q2a, q3a = _solve_q2_q3(+1.0)
    q2b, q3b = _solve_q2_q3(-1.0)

    def _build(n_target: int) -> np.ndarray:
        n_target = max(1, int(n_target))
        q1 = np.random.uniform(-math.pi, math.pi, size=(n_target,)).astype(np.float32)
        pick_a = np.random.rand(n_target) < 0.5
        q2 = np.where(pick_a, q2a, q2b).astype(np.float32)
        q3 = np.where(pick_a, q3a, q3b).astype(np.float32)
        return np.stack([_wrap_to_pi(q1), _wrap_to_pi(q2), _wrap_to_pi(q3)], axis=1).astype(np.float32)

    x_train = _build(cfg.n_train)
    grid = _build(cfg.n_grid)
    return x_train, grid


def _spatial_arm_up_n6(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    # 6-DoF arm with end-effector tool-axis "upward" orientation.
    # Kinematic orientation chain uses axes: z, y, y, y, x, z.
    # Tool axis is local +x; "up" means world-axis alignment with +z:
    # h1 = a_x = 0, h2 = a_y = 0, and enforce branch a_z > 0.
    # This defines codim=2 manifold in 6D.
    def _rot_x(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    def _rot_y(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    def _rot_z(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _tool_axis_world(q: np.ndarray) -> np.ndarray:
        # q shape: (6,)
        R = np.eye(3, dtype=np.float64)
        axes = ("z", "y", "y", "y", "x", "z")
        for j, ax in enumerate(axes):
            a = float(q[j])
            if ax == "x":
                R = R @ _rot_x(a)
            elif ax == "y":
                R = R @ _rot_y(a)
            else:
                R = R @ _rot_z(a)
        # tool/gripper pointing axis (local +x)
        return R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def _solve_q56_for_up(q14: np.ndarray, n_restart: int = 6) -> np.ndarray | None:
        # Solve h=[a_x,a_y]=0 for q5,q6 with branch a_z>0.
        best_q = None
        best_v = 1e18
        for _ in range(max(1, n_restart)):
            q = np.zeros((6,), dtype=np.float64)
            q[:4] = q14.astype(np.float64)
            q[4] = np.random.uniform(-math.pi, math.pi)
            q[5] = np.random.uniform(-math.pi, math.pi)
            for _it in range(32):
                a = _tool_axis_world(q)
                r = np.array([a[0], a[1]], dtype=np.float64)
                v = float(np.linalg.norm(r))
                if v < best_v and a[2] > 0.0:
                    best_v = v
                    best_q = q.copy()
                if v < 1e-6 and a[2] > 0.0:
                    return _wrap_to_pi(q.astype(np.float32))
                eps = 1e-4
                J = np.zeros((2, 2), dtype=np.float64)
                for k, jidx in enumerate((4, 5)):
                    qd = q.copy()
                    qd[jidx] += eps
                    ad = _tool_axis_world(qd)
                    rd = np.array([ad[0], ad[1]], dtype=np.float64)
                    J[:, k] = (rd - r) / eps
                JTJ = J.T @ J + 1e-6 * np.eye(2, dtype=np.float64)
                step = np.linalg.solve(JTJ, J.T @ r)
                step = np.clip(step, -0.35, 0.35)
                q[4] -= step[0]
                q[5] -= step[1]
                q = _wrap_to_pi(q.astype(np.float32)).astype(np.float64)
        if best_q is not None and best_v < 2e-3:
            return _wrap_to_pi(best_q.astype(np.float32))
        return None

    def _build_set(n_target: int) -> np.ndarray:
        out = []
        trials = 0
        max_trials = max(2000, 80 * n_target)
        while len(out) < n_target and trials < max_trials:
            trials += 1
            q14 = np.random.uniform(-math.pi, math.pi, size=(4,)).astype(np.float32)
            q = _solve_q56_for_up(q14, n_restart=6)
            if q is not None:
                out.append(q.astype(np.float32))
        if len(out) == 0:
            # Fallback to avoid empty dataset.
            q14 = np.random.uniform(-math.pi, math.pi, size=(n_target, 4)).astype(np.float32)
            q56 = np.zeros((n_target, 2), dtype=np.float32)
            return np.concatenate([q14, q56], axis=1).astype(np.float32)
        arr = np.stack(out, axis=0).astype(np.float32)
        return _sample_rows(arr, n_target)

    n_train = max(1, int(cfg.n_train))
    n_grid = max(1, int(cfg.n_grid))
    x_train = _build_set(n_train)
    grid = _build_set(n_grid)
    return x_train, grid


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
        x = np.concatenate([a * np.cos(t), b * np.sin(t)], axis=1).astype(np.float32)
        hole_centers = np.array([0.6, 2.2, 4.0], dtype=np.float32)
        hole_width = 0.25 * 1.15
        mask = np.ones(len(t), dtype=bool)
        for c in hole_centers:
            mask &= np.abs(t.squeeze() - c) > hole_width
        x = x[mask]
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
    if name == "planar_arm_line_n2":
        return _planar_arm_line_n2(cfg)
    if name == "planar_arm_line_n3":
        return _planar_arm_line_n3(cfg)
    if name == "spatial_arm_plane_n3":
        return _spatial_arm_plane_n3(cfg)
    if name == "spatial_arm_circle_n3":
        return _spatial_arm_circle_n3(cfg)
    if name == "spatial_arm_plane_n4":
        return _spatial_arm_plane_n4(cfg)
    if name == "spatial_arm_up_n6":
        return _spatial_arm_up_n6(cfg)
    if name == "noise_only":
        n_sparse = max(128, cfg.n_train // 4)
        t = np.random.uniform(-math.pi, math.pi, size=(n_sparse, 1))
        y = np.sin(t) + 0.1 * np.random.randn(n_sparse, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "sine":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "sparse_only":
        n_sparse = max(45, cfg.n_train // 8)
        t = np.random.uniform(-math.pi, math.pi, size=(n_sparse, 1))
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "discontinuous":
        n_half = cfg.n_train // 2
        t1 = np.random.uniform(-math.pi, -0.7, size=(n_half, 1))
        t2 = np.random.uniform(0.7, math.pi, size=(cfg.n_train - n_half, 1))
        t = np.vstack([t1, t2])
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "looped_spiro":
        t = np.random.uniform(0.0, 2.0 * math.pi, size=(cfg.n_train, 1))
        petals = 4.0
        base = 1.2
        amp = 0.35
        r = base + amp * np.cos(petals * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        x = np.concatenate([x, y], axis=1).astype(np.float32)
        tg = np.linspace(0.0, 2.0 * math.pi, cfg.n_grid).reshape(-1, 1)
        rg = base + amp * np.cos(petals * tg)
        gx = rg * np.cos(tg)
        gy = rg * np.sin(tg)
        grid = np.concatenate([gx, gy], axis=1).astype(np.float32)
        return x, grid
    if name == "sharp_star":
        t = np.random.uniform(0.0, 2.0 * math.pi, size=(cfg.n_train, 1))
        scale = 2.0
        r = 1.0 + 0.35 * np.cos(5 * t) + 0.15 * np.cos(10 * t)
        x = scale * r * np.cos(t)
        y = scale * r * np.sin(t)
        x = np.concatenate([x, y], axis=1).astype(np.float32)
        tg = np.linspace(0.0, 2.0 * math.pi, cfg.n_grid).reshape(-1, 1)
        rg = 1.0 + 0.35 * np.cos(5 * tg) + 0.15 * np.cos(10 * tg)
        gx = scale * rg * np.cos(tg)
        gy = scale * rg * np.sin(tg)
        grid = np.concatenate([gx, gy], axis=1).astype(np.float32)
        return x, grid
    if name == "hetero_noise":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        sigma = 0.02 + 0.3 * np.exp(-0.5 * (t / 0.8) ** 2)
        y = np.sin(t) + sigma * np.random.randn(cfg.n_train, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
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


def _as_sigmas(sigmas: Tuple[float, ...] | float) -> np.ndarray:
    if isinstance(sigmas, (int, float)):
        return np.array([float(sigmas)], dtype=np.float32)
    return np.array(sigmas, dtype=np.float32)


def knn_normals(x: np.ndarray, k: int) -> np.ndarray:
    n, d = x.shape
    d2 = pairwise_sqdist(x, x)
    idx = np.argsort(d2, axis=1)[:, 1 : k + 1]
    normals = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        nbrs = x[idx[i]]
        _, evecs, _, _ = _local_pca_frame(nbrs, x[i], cfg=None)
        nvec = evecs[:, 0]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        normals[i] = nvec.astype(np.float32)
    return normals


def effective_knn_norm_estimation_points(cfg: Config, n: int) -> int:
    return max(cfg.knn_norm_estimation_min_points, int(round(cfg.knn_norm_estimation_ratio * n)))


def effective_knn_off_data_filter_points(cfg: Config, n: int) -> int:
    return max(
        cfg.knn_off_data_filter_min_points,
        int(round(cfg.knn_off_data_filter_ratio * n)),
    )


def _radius_knn_indices(
    d2_row: np.ndarray, k0: int, scale: float
) -> np.ndarray:
    order = np.argsort(d2_row)
    order = order[1:]
    if len(order) == 0:
        return order
    k0 = min(k0, len(order))
    kth = order[k0 - 1]
    spacing = math.sqrt(float(d2_row[kth]))
    radius = scale * spacing
    mask = np.sqrt(d2_row[order]) <= radius
    idx = order[mask]
    if len(idx) == 0:
        idx = order[:k0]
    return idx


def _compute_pca_weights(
    neigh: np.ndarray,
    center: np.ndarray,
    use_weighted: bool,
    eps: float = 1e-12,
) -> np.ndarray:
    if not use_weighted:
        return np.full((len(neigh),), 1.0 / max(len(neigh), 1), dtype=np.float64)
    # Gaussian weights using a robust local scale (median distance).
    d = np.linalg.norm(neigh - center[None, :], axis=1)
    sigma = float(np.median(d)) if len(d) > 0 else 1.0
    sigma = max(sigma, eps)
    w = np.exp(-(d ** 2) / (2.0 * sigma * sigma + eps))
    w = w / (float(np.sum(w)) + eps)
    return w


def _pca_from_weights(
    neigh: np.ndarray, w: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = (w[:, None] * neigh).sum(axis=0, keepdims=True)
    xc = neigh - mu
    cov = (xc.T * w) @ xc
    evals, evecs = np.linalg.eigh(cov)
    return evals, evecs, mu.squeeze(0)


def _local_pca_frame(
    neigh: np.ndarray,
    center: np.ndarray,
    cfg: Config | None = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    use_weighted = bool(cfg.use_weighted_pca) if cfg is not None else False
    use_trimmed = bool(cfg.use_trimmed_pca) if cfg is not None else False
    keep_ratio = (
        float(np.clip(cfg.trimmed_pca_keep_ratio, 0.1, 1.0)) if cfg is not None else 1.0
    )

    if len(neigh) == 0:
        d = center.shape[0]
        return (
            np.zeros((d,), dtype=np.float64),
            np.eye(d, dtype=np.float64),
            np.ones((1,), dtype=np.float64),
            neigh,
        )

    w0 = _compute_pca_weights(neigh, center, use_weighted=use_weighted, eps=eps)
    evals0, evecs0, mu0 = _pca_from_weights(neigh, w0, eps=eps)

    if use_trimmed and len(neigh) >= max(6, center.shape[0] + 2):
        n0 = evecs0[:, 0]
        residual = np.abs((neigh - mu0[None, :]) @ n0)
        keep_n = int(math.ceil(keep_ratio * len(neigh)))
        keep_n = int(np.clip(keep_n, center.shape[0] + 1, len(neigh)))
        keep_idx = np.argsort(residual)[:keep_n]
        neigh = neigh[keep_idx]

    w = _compute_pca_weights(neigh, center, use_weighted=use_weighted, eps=eps)
    evals, evecs, _ = _pca_from_weights(neigh, w, eps=eps)
    return evals, evecs, w, neigh


def knn_normals_with_quality(
    x: np.ndarray, k: int, cfg: Config | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = x.shape
    d2 = pairwise_sqdist(x, x)
    normals = np.zeros((n, d), dtype=np.float32)
    quality = np.zeros((n,), dtype=np.float32)
    thickness = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        if cfg is not None and cfg.use_radius_knn:
            nbr_idx = _radius_knn_indices(
                d2[i], cfg.radius_knn_k, cfg.radius_knn_scale
            )
        else:
            nbr_idx = np.argsort(d2[i])[1 : k + 1]
        nbrs = x[nbr_idx]
        evals, evecs, w_loc, nbrs_used = _local_pca_frame(nbrs, x[i], cfg=cfg)
        nvec = evecs[:, 0]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        normals[i] = nvec.astype(np.float32)
        quality[i] = float(evals[0] / (evals[-1] + 1e-12))
        proj = (nbrs_used - x[i : i + 1]) @ nvec.reshape(-1, 1)
        thickness[i] = float(np.sqrt(np.sum(w_loc * (proj[:, 0] ** 2))))
    return normals, quality, thickness


def compute_adaptive_sigma(
    x: np.ndarray, cfg: Config, knn_norm_estimation_points: int | None = None
) -> np.ndarray | None:
    if x.shape[1] != 2:
        return None
    n = len(x)
    if n < 3:
        return None
    if knn_norm_estimation_points is None:
        knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, n)
    k_n = min(knn_norm_estimation_points, n)
    d2 = pairwise_sqdist(x, x)
    sigmas = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if cfg.use_radius_knn:
            nn_idx = _radius_knn_indices(
                d2[i], cfg.radius_knn_k, cfg.radius_knn_scale
            )
            if len(nn_idx) == 0:
                continue
            nn_idx = np.concatenate(([i], nn_idx))
        else:
            order = np.argsort(d2[i])
            nn_idx = order[:k_n]
        neigh = x[nn_idx]
        center = x[i]
        _, evecs, w_loc, neigh_used = _local_pca_frame(neigh, center, cfg=cfg)
        n_hat = evecs[:, 0]
        t_hat = evecs[:, 1]
        n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
        t_hat = t_hat / (np.linalg.norm(t_hat) + 1e-12)
        dvec = neigh_used - center[None, :]
        u = dvec @ t_hat
        vproj = dvec @ n_hat
        num = float(np.sum(w_loc * (u ** 2) * vproj))
        den = float(np.sum(w_loc * (u ** 4))) + 1e-12
        kappa_hat = abs((num / den) / 0.5)
        # Compress curvature dynamic range with configurable power.
        kappa_eff = max(kappa_hat, 0.0) ** float(cfg.adp_sigma_kappa_exp)
        r_kappa = cfg.adp_sigma_scale / (kappa_eff + cfg.adp_sigma_eps)
        r_kappa = float(np.clip(r_kappa, cfg.adp_sigma_r_min, cfg.adp_sigma_r_max))
        sigmas[i] = r_kappa
    return sigmas


def compute_adaptive_sigma_bounds(
    x: np.ndarray,
    n_hat: np.ndarray,
    sigma_per_point: np.ndarray | None,
    cfg: Config,
    knn_norm_estimation_points: int,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if sigma_per_point is None:
        return None, None
    r_pos = sigma_per_point.astype(np.float32).copy()
    r_neg = sigma_per_point.astype(np.float32).copy()
    if (not cfg.adp_sigma_asymmetric_enable) or x.shape[1] != 2 or len(x) < 3:
        return r_pos, r_neg
    d2 = pairwise_sqdist(x, x)
    order = np.argsort(d2, axis=1)
    rank = max(1, int(knn_norm_estimation_points) + int(cfg.adp_sigma_nonlocal_offset))
    for i in range(len(x)):
        j = int(order[i, min(len(x) - 1, rank)])
        side = float(np.dot(x[j] - x[i], n_hat[i]))
        danger = float(sigma_per_point[i] * cfg.adp_sigma_danger_scale)
        safe = float(sigma_per_point[i] * cfg.adp_sigma_safe_scale)
        if side >= 0.0:
            rp, rn = danger, safe
        else:
            rp, rn = safe, danger
        r_pos[i] = float(np.clip(rp, cfg.adp_sigma_r_min, cfg.adp_sigma_r_max))
        r_neg[i] = float(np.clip(rn, cfg.adp_sigma_r_min, cfg.adp_sigma_r_max))
    return r_pos, r_neg


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


def thickness_weight(
    t: torch.Tensor, cfg: Config
) -> torch.Tensor:
    return torch.ones_like(t)


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




def compute_losses(
    model: nn.Module,
    x_on: torch.Tensor,
    n_hat: torch.Tensor,
    t_on: torch.Tensor,
    x_ref: torch.Tensor,
    idx_on: torch.Tensor,
    cfg: Config,
    x_off: torch.Tensor | None = None,
    s_off: torch.Tensor | None = None,
    step: int = 0,
    sigma_per_point: torch.Tensor | None = None,
    r_kappa_per_point: torch.Tensor | None = None,
    r_pos_per_point: torch.Tensor | None = None,
    r_neg_per_point: torch.Tensor | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    need_loss_denoise = cfg.lam_denoise != 0.0
    need_loss_recon = cfg.lam_recon != 0.0
    need_loss_dist = cfg.lam_dist != 0.0
    need_loss_dir = cfg.lam_dir != 0.0
    need_loss_pl = cfg.lam_pl != 0.0
    need_loss_smooth = cfg.lam_smooth != 0.0
    need_loss_thin = cfg.lam_thin != 0.0

    f_on = model(x_on)
    if cfg.use_thickness_on:
        loss_on = ((f_on ** 2) / (t_on + cfg.eps)).mean()
    else:
        loss_on = (f_on ** 2).mean()
    # Explicit noise model: one-step denoising toward the zero set
    if (
        (need_loss_denoise or need_loss_recon)
        and cfg.loss_denoise_every > 0
        and step % cfg.loss_denoise_every == 0
    ):
        x_on_detached = x_on.detach().requires_grad_(True)
        v_on_det = energy_from_f(model(x_on_detached))
        grad_on = torch.autograd.grad(v_on_det.sum(), x_on_detached, create_graph=True)[0]
        y = x_on_detached - cfg.denoise_step * grad_on
        loss_denoise = (model(y) ** 2).mean() if need_loss_denoise else torch.zeros((), device=x_on.device)
        loss_recon = (
            ((x_on_detached - y) ** 2).sum(dim=1).mean()
            if need_loss_recon
            else torch.zeros((), device=x_on.device)
        )
    else:
        loss_denoise = torch.zeros((), device=x_on.device)
        loss_recon = torch.zeros((), device=x_on.device)

    # No off-surface term active: skip off sampling and all higher-order graphs.
    if not (need_loss_dist or need_loss_dir or need_loss_pl or need_loss_smooth):
        loss_thin = (
            ((f_on ** 2) / (t_on + cfg.eps)).mean()
            if need_loss_thin
            else torch.zeros((), device=x_on.device)
        )
        parts = {
            "loss_on": loss_on,
            "loss_dist": torch.zeros((), device=x_on.device),
            "loss_dir": torch.zeros((), device=x_on.device),
            "loss_pl": torch.zeros((), device=x_on.device),
            "loss_smooth": torch.zeros((), device=x_on.device),
            "loss_denoise": loss_denoise,
            "loss_recon": loss_recon,
            "loss_thin": loss_thin,
        }
        stats = {
            "f_on_abs": float(f_on.abs().mean().item()),
            "f_off_abs": 0.0,
            "cos_abs": 0.0,
            "grad_norm": 0.0,
            "loss_on": float(loss_on.detach().cpu()),
            "loss_denoise": float(loss_denoise.detach().cpu()),
            "loss_recon": float(loss_recon.detach().cpu()),
            "loss_dist": 0.0,
            "loss_dir": 0.0,
            "loss_pl": 0.0,
            "loss_smooth": 0.0,
            "loss_thin": float(loss_thin.detach().cpu()),
        }
        return parts, stats

    if x_off is None or s_off is None:
        x_off, s = sample_off_manifold(
            x_on,
            n_hat,
            cfg.sigmas,
            cfg.off_sigma_mode,
            sigma_per_point,
            r_kappa_per_point,
            r_pos_per_point,
            r_neg_per_point,
        )
        if cfg.use_knn_filter:
            knn_off_data_filter_points = effective_knn_off_data_filter_points(cfg, x_ref.shape[0])
            mask = filter_off_by_knn(x_off, x_ref, idx_on, knn_off_data_filter_points)
            if mask.sum() > 0:
                x_off = x_off[mask]
                s = s[mask]
                n_hat_m = n_hat[mask]
                t_on_m = t_on[mask]
            else:
                # Preserve filter semantics: no valid off-point -> skip off losses this step.
                z = torch.zeros((), device=x_on.device)
                loss_thin = ((f_on ** 2) / (t_on + cfg.eps)).mean()
                parts = {
                    "loss_on": loss_on,
                    "loss_dist": z,
                    "loss_dir": z,
                    "loss_pl": z,
                    "loss_smooth": z,
                    "loss_denoise": loss_denoise,
                    "loss_recon": loss_recon,
                    "loss_thin": loss_thin,
                }
                stats = {
                    "f_on_abs": float(f_on.abs().mean().item()),
                    "f_off_abs": 0.0,
                    "cos_abs": 0.0,
                    "grad_norm": 0.0,
                }
                return parts, stats
        else:
            n_hat_m = n_hat
            t_on_m = t_on
    else:
        x_off = x_off
        s = s_off
        n_hat_m = n_hat
        t_on_m = t_on
    x_off.requires_grad_(True)
    f_off = model(x_off)
    v_off = energy_from_f(f_off)

    w = thickness_weight(t_on_m, cfg)
    loss_dist = (
        (w * (torch.abs(f_off) - torch.abs(s)) ** 2).mean()
        if need_loss_dist
        else torch.zeros((), device=x_off.device)
    )

    grad = None
    grad_norm = None
    cos = None
    if need_loss_dir or need_loss_pl or need_loss_smooth:
        grad = torch.autograd.grad(
            v_off.sum(),
            x_off,
            create_graph=True,
            retain_graph=bool(need_loss_smooth),
        )[0]
        grad_norm = torch.norm(grad, dim=1, keepdim=True) + cfg.eps

    if need_loss_dir:
        cos = (grad * n_hat_m).sum(dim=1, keepdim=True) / grad_norm
        # Downweight direction loss in thick/unstable regions
        loss_dir = (w * torch.clamp(1.0 - torch.abs(cos), min=0.0)).mean()
    else:
        loss_dir = torch.zeros((), device=x_off.device)

    if need_loss_pl:
        loss_pl = torch.clamp(cfg.beta_pl * v_off - (grad_norm ** 2), min=0.0).mean()
    else:
        loss_pl = torch.zeros((), device=x_off.device)

    # Local smoothness: compare gradients at nearby points
    if need_loss_smooth and cfg.loss_smooth_every > 0 and step % cfg.loss_smooth_every == 0:
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
    else:
        loss_smooth = torch.zeros((), device=x_off.device)
    # Thinness: use the same thickness weight to favor thin regions
    loss_thin = (
        ((f_on ** 2) / (t_on + cfg.eps)).mean()
        if need_loss_thin
        else torch.zeros((), device=x_on.device)
    )

    stats = {
        "loss_on": float(loss_on.detach().cpu()),
        "loss_denoise": float(loss_denoise.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_dist": float(loss_dist.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_pl": float(loss_pl.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_thin": float(loss_thin.detach().cpu()),
        "f_on_abs": float(f_on.abs().mean().item()),
        "f_off_abs": float(f_off.abs().mean().detach().cpu().item()),
        "cos_abs": float(torch.abs(cos).mean().detach().cpu().item()) if cos is not None else 0.0,
        "grad_norm": float(torch.mean(grad_norm).detach().cpu().item()) if grad_norm is not None else 0.0,
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
    knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x))
    n_hat, quality, thickness = knn_normals_with_quality(x, knn_norm_estimation_points, cfg)
    sigma_per_point = None
    r_kappa_per_point = None
    r_pos_per_point = None
    r_neg_per_point = None
    if cfg.use_adaptive_sigma:
        sigma_per_point = compute_adaptive_sigma(x, cfg, knn_norm_estimation_points=knn_norm_estimation_points)
        r_kappa_per_point = sigma_per_point
        r_pos_per_point, r_neg_per_point = compute_adaptive_sigma_bounds(
            x, n_hat, sigma_per_point, cfg, knn_norm_estimation_points
        )
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_hat)
    t_t = torch.from_numpy(thickness).unsqueeze(1)
    idx_t = torch.arange(len(x), dtype=torch.long)

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
    x_ref = x_t.to(cfg.device)
    global_step = 0
    for epoch in range(cfg.epochs):
        for xb, nb, tb, idxb in dl:
            xb = xb.to(cfg.device)
            nb = nb.to(cfg.device)
            tb = tb.to(cfg.device)
            idxb = idxb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            if off_bank is not None and s_bank is not None:
                idx_np = idxb.cpu().numpy()
                cap_ratio = _off_curriculum_cap_ratio(cfg, epoch)
                bank_idx = _sample_off_bank_indices(s_bank[idx_np], cap_ratio)
                x_off = torch.from_numpy(off_bank[idx_np, bank_idx]).to(cfg.device)
                s_off = torch.from_numpy(s_bank[idx_np, bank_idx]).to(cfg.device)
                sigma_batch = None
                r_kappa_batch = None
                r_pos_batch = None
                r_neg_batch = None
                if sigma_per_point is not None:
                    sigma_batch = torch.from_numpy(
                        sigma_per_point[idx_np].astype(np.float32)
                    ).to(cfg.device)
                if r_kappa_per_point is not None:
                    r_kappa_batch = torch.from_numpy(
                        r_kappa_per_point[idx_np].astype(np.float32)
                    ).to(cfg.device)
                if r_pos_per_point is not None:
                    r_pos_batch = torch.from_numpy(
                        r_pos_per_point[idx_np].astype(np.float32)
                    ).to(cfg.device)
                if r_neg_per_point is not None:
                    r_neg_batch = torch.from_numpy(
                        r_neg_per_point[idx_np].astype(np.float32)
                    ).to(cfg.device)
                parts, stats = compute_losses(
                    model,
                    xb,
                    nb,
                    tb,
                    x_ref,
                    idxb,
                    cfg,
                    x_off=x_off,
                    s_off=s_off,
                    step=global_step,
                    sigma_per_point=sigma_batch,
                    r_kappa_per_point=r_kappa_batch,
                    r_pos_per_point=r_pos_batch,
                    r_neg_per_point=r_neg_batch,
                )
            else:
                sigma_batch = None
                r_kappa_batch = None
                r_pos_batch = None
                r_neg_batch = None
                if sigma_per_point is not None:
                    sigma_batch = torch.from_numpy(
                        sigma_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                if r_kappa_per_point is not None:
                    r_kappa_batch = torch.from_numpy(
                        r_kappa_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                if r_pos_per_point is not None:
                    r_pos_batch = torch.from_numpy(
                        r_pos_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                if r_neg_per_point is not None:
                    r_neg_batch = torch.from_numpy(
                        r_neg_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                parts, stats = compute_losses(
                    model,
                    xb,
                    nb,
                    tb,
                    x_ref,
                    idxb,
                    cfg,
                    step=global_step,
                    sigma_per_point=sigma_batch,
                    r_kappa_per_point=r_kappa_batch,
                    r_pos_per_point=r_pos_batch,
                    r_neg_per_point=r_neg_batch,
                )
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
            global_step += 1
        if last_stats:
            for k in history:
                history[k].append(last_stats[k])
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch+1:04d}  loss={last_stats['loss_total']:.6f}")

    return model, last_stats, n_hat, quality, thickness, history


def project_points(
    model: nn.Module, x0: torch.Tensor, cfg: Config, f_abs_stop: float | None = None
) -> Tuple[torch.Tensor, int]:
    x = x0.clone()
    for k in range(cfg.proj_steps):
        x.requires_grad_(True)
        with torch.enable_grad():
            f = model(x)
            v = energy_from_f(f)
            grad = torch.autograd.grad(v.sum(), x)[0]
        if cfg.proj_grad_clip > 0:
            grad_norm = torch.norm(grad, dim=1, keepdim=True) + cfg.eps
            scale = torch.clamp(cfg.proj_grad_clip / grad_norm, max=1.0)
            grad = grad * scale
        if f_abs_stop is not None:
            stop_f = torch.abs(f) < f_abs_stop
        else:
            stop_f = torch.zeros_like(v, dtype=torch.bool)
        if torch.all(stop_f):
            return x.detach(), k
        x_next = x - cfg.proj_alpha * grad
        x = x_next.detach()
    return x.detach(), cfg.proj_steps


def project_points_with_steps(
    model: nn.Module, x0: torch.Tensor, cfg: Config, f_abs_stop: float | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x0.clone()
    n = x.shape[0]
    steps = torch.zeros(n, dtype=torch.long, device=x.device)
    active = torch.ones(n, dtype=torch.bool, device=x.device)
    for k in range(cfg.proj_steps):
        x.requires_grad_(True)
        with torch.enable_grad():
            f = model(x)
            v = energy_from_f(f)
            grad = torch.autograd.grad(v.sum(), x)[0]
        if cfg.proj_grad_clip > 0:
            grad_norm = torch.norm(grad, dim=1, keepdim=True) + cfg.eps
            scale = torch.clamp(cfg.proj_grad_clip / grad_norm, max=1.0)
            grad = grad * scale
        if f_abs_stop is not None:
            stop_f = torch.abs(f) < f_abs_stop
        else:
            stop_f = torch.zeros_like(v, dtype=torch.bool)
        converged = stop_f
        converged = converged.squeeze(1)
        newly = active & converged
        steps[newly] = k
        next_active = active & (~converged)
        if not next_active.any():
            active = next_active
            break
        x_next = x - cfg.proj_alpha * grad
        x = torch.where(next_active.unsqueeze(1), x_next, x)
        active = next_active
        x = x.detach()
    steps[active] = cfg.proj_steps
    return x.detach(), steps.detach()


def project_trajectory(
    model: nn.Module, x0: torch.Tensor, cfg: Config, f_abs_stop: float | None = None
) -> Tuple[torch.Tensor, int]:
    traj = []
    x = x0.clone()
    for k in range(cfg.proj_steps):
        x.requires_grad_(True)
        with torch.enable_grad():
            f = model(x)
            v = energy_from_f(f)
            grad = torch.autograd.grad(v.sum(), x)[0]
        if cfg.proj_grad_clip > 0:
            grad_norm = torch.norm(grad, dim=1, keepdim=True) + cfg.eps
            scale = torch.clamp(cfg.proj_grad_clip / grad_norm, max=1.0)
            grad = grad * scale
        if f_abs_stop is not None:
            stop_f = torch.abs(f) < f_abs_stop
        else:
            stop_f = torch.zeros_like(v, dtype=torch.bool)
        if torch.all(stop_f):
            if k == 0:
                traj.append(x.detach())
            return torch.stack(traj, dim=0), k
        x_next = x - cfg.proj_alpha * grad
        if k == 0:
            traj.append(x.detach())
        traj.append(x_next.detach())
        x = x_next.detach()
    return torch.stack(traj, dim=0), cfg.proj_steps


def plan_linear_then_project(
    model: nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    cfg: Config,
    keep_endpoints: bool = False,
    f_abs_stop: float | None = None,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, cfg.plan_steps + 1, dtype=np.float32)[:, None]
    path = (1.0 - t) * x0[None, :] + t * x1[None, :]
    path_t = torch.from_numpy(path).to(cfg.device)
    proj_t, _ = project_points(model, path_t, cfg, f_abs_stop=f_abs_stop)
    proj = proj_t.detach().cpu().numpy()
    if keep_endpoints:
        proj[0] = x0
        proj[-1] = x1
    return proj


def plan_constrained_path(
    model: nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    x0_t = torch.from_numpy(x0[None, :]).to(cfg.device)
    x1_t = torch.from_numpy(x1[None, :]).to(cfg.device)
    t = torch.linspace(0.0, 1.0, steps=cfg.plan_steps + 1, device=cfg.device)[:, None]
    path = (1.0 - t) * x0_t + t * x1_t
    path = path.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([path], lr=cfg.plan_lr)
    for _ in range(cfg.plan_iters):
        opt.zero_grad(set_to_none=True)
        diffs = path[1:] - path[:-1]
        smooth_loss = (diffs ** 2).sum()
        f = model(path)
        manifold_loss = (f ** 2).mean()
        loss = cfg.plan_smooth_weight * smooth_loss + cfg.plan_manifold_weight * manifold_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            path[0].copy_(x0_t[0])
            path[-1].copy_(x1_t[0])
    return path.detach().cpu().numpy()


def plot_planned_paths(
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    plans_proj: List[np.ndarray],
    plans_constr: List[np.ndarray],
    cfg: Config,
    out_path: str,
    title: str,
    zero_level_eps: float,
) -> None:
    if x_train.shape[1] != 2:
        return
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    x_plot_min, x_plot_max = -4.0, 4.0
    y_plot_min, y_plot_max = -3.0, 3.0
    xx, yy = np.meshgrid(
        np.linspace(x_plot_min, x_plot_max, 300),
        np.linspace(y_plot_min, y_plot_max, 300),
    )
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid_eval).to(cfg.device)
    with torch.no_grad():
        f_grid = model(grid_t).cpu().numpy().reshape(xx.shape)
        v = (0.5 * (f_grid ** 2)).reshape(xx.shape)
    v_max = float(np.percentile(v, 95))
    u = np.linspace(0.0, 1.0, 25)
    levels = (u ** 3) * v_max

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    eps = float(zero_level_eps)
    plt.contourf(
        xx,
        yy,
        f_grid,
        levels=[-eps, eps],
        colors=["#ffa500"],
        alpha=0.55,
    )
    plt.scatter(
        x_train[:, 0], x_train[:, 1], s=4, alpha=0.6, label="data", zorder=3
    )
    for i, traj in enumerate(plans_proj):
        label = "projected (on)" if i == 0 else None
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            color="gold",
            linewidth=2.0,
            label=label,
        )
        plt.scatter(traj[0, 0], traj[0, 1], c="gold", s=25, zorder=4)
        plt.scatter(traj[-1, 0], traj[-1, 1], c="gold", s=25, zorder=4)
    for i, traj in enumerate(plans_constr):
        label = "constrained (on)" if i == 0 else None
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            "--",
            color="purple",
            linewidth=2.0,
            label=label,
        )
        plt.scatter(traj[0, 0], traj[0, 1], c="purple", s=25, zorder=4)
        plt.scatter(traj[-1, 0], traj[-1, 1], c="purple", s=25, zorder=4)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_planned_paths_off(
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    plans_proj_off: List[np.ndarray],
    plans_constr_off: List[np.ndarray],
    cfg: Config,
    out_path: str,
    title: str,
    zero_level_eps: float,
) -> None:
    if x_train.shape[1] != 2:
        return
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    x_plot_min, x_plot_max = -4.0, 4.0
    y_plot_min, y_plot_max = -3.0, 3.0
    xx, yy = np.meshgrid(
        np.linspace(x_plot_min, x_plot_max, 300),
        np.linspace(y_plot_min, y_plot_max, 300),
    )
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid_eval).to(cfg.device)
    with torch.no_grad():
        f_grid = model(grid_t).cpu().numpy().reshape(xx.shape)
        v = (0.5 * (f_grid ** 2)).reshape(xx.shape)
    v_max = float(np.percentile(v, 95))
    u = np.linspace(0.0, 1.0, 25)
    levels = (u ** 3) * v_max

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    eps = float(zero_level_eps)
    plt.contourf(
        xx,
        yy,
        f_grid,
        levels=[-eps, eps],
        colors=["#ffa500"],
        alpha=0.55,
    )
    plt.scatter(
        x_train[:, 0], x_train[:, 1], s=4, alpha=0.6, label="data", zorder=3
    )
    for i, traj in enumerate(plans_proj_off):
        label = "projected (off)" if i == 0 else None
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            ":",
            color="gold",
            linewidth=2.0,
            label=label,
        )
        plt.scatter(traj[0, 0], traj[0, 1], c="gold", s=25, zorder=4, marker="o")
        plt.scatter(traj[-1, 0], traj[-1, 1], c="gold", s=40, zorder=4, marker="x")
    for i, traj in enumerate(plans_constr_off):
        label = "constrained (off)" if i == 0 else None
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            "-.",
            color="purple",
            linewidth=2.0,
            label=label,
        )
        plt.scatter(traj[0, 0], traj[0, 1], c="purple", s=25, zorder=4, marker="o")
        plt.scatter(traj[-1, 0], traj[-1, 1], c="purple", s=40, zorder=4, marker="x")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
    delta = np.random.randn(*x_anchor.shape).astype(np.float32) * float(_as_sigmas(cfg.sigmas)[0])
    x_off = x_anchor + delta
    d_true = true_distance(x_off, grid)
    x_off_t = torch.from_numpy(x_off).to(cfg.device)
    v_off = energy_from_f(model(x_off_t)).detach().cpu().numpy().reshape(-1)
    corr = float(np.corrcoef(v_off, d_true ** 2)[0, 1])

    return {"on_mean_v": on_mean, "corr_v_d2": corr}


def sample_eval_points(
    x_train: np.ndarray, grid: np.ndarray, cfg: Config, rng: np.random.Generator
) -> np.ndarray:
    ref = x_train  # we use train data not true data since data might be sparse
    mins, maxs = eval_bounds_from_train(ref, cfg)
    samples = rng.uniform(mins, maxs, size=(cfg.eval_n, ref.shape[1])).astype(np.float32)
    rng.shuffle(samples)
    return samples


def eval_bounds_from_train(x_train: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    mins = x_train.min(axis=0)
    maxs = x_train.max(axis=0)
    span = maxs - mins
    scale = 1.0 + float(cfg.eval_pad_ratio)
    scale = max(scale, 1e-6)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span * scale
    return center - half, center + half


def compute_eval_eps_used(model: nn.Module, x_train: np.ndarray, cfg: Config) -> float:
    x_t = torch.from_numpy(x_train).to(cfg.device)
    with torch.no_grad():
        f_on = model(x_t).detach().cpu().numpy().reshape(-1)
    eps_mode = cfg.eval_eps_mode.lower().strip()
    if eps_mode == "quantile":
        return float(np.percentile(np.abs(f_on), cfg.eval_eps_quantile))
    return float(cfg.eval_eps)


def evaluate_metrics(
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    x_eval: np.ndarray,
    cfg: Config,
) -> Dict[str, float]:
    model.eval()
    x_t = torch.from_numpy(x_train).to(cfg.device)
    with torch.no_grad():
        f_on = model(x_t).detach().cpu().numpy().reshape(-1)
        v_on = 0.5 * (f_on ** 2)
    on_mean_v = float(np.mean(v_on))
    eval_eps_used = compute_eval_eps_used(model, x_train, cfg)

    x_eval_t = torch.from_numpy(x_eval).to(cfg.device)
    with torch.no_grad():
        f_eval = model(x_eval_t).detach().cpu().numpy().reshape(-1)
    v_eval = 0.5 * (f_eval ** 2)

    _, d_true_eval = true_projection(x_eval, grid)
    d2_true = d_true_eval ** 2
    corr = float(np.corrcoef(v_eval, d2_true)[0, 1])
    denom = float(np.dot(d2_true, d2_true)) + 1e-12
    slope = float(np.dot(d2_true, v_eval) / denom)

    proj_t, steps_t = project_points_with_steps(
        model, x_eval_t, cfg, f_abs_stop=eval_eps_used
    )
    proj = proj_t.detach().cpu().numpy()
    steps = steps_t.detach().cpu().numpy()

    proj_true, _ = true_projection(x_eval, grid)
    proj_mask = np.isfinite(proj).all(axis=1)
    if np.any(proj_mask):
        proj_to_trueproj = np.linalg.norm(proj[proj_mask] - proj_true[proj_mask], axis=1)
        proj_final_true_dist = true_distance(proj[proj_mask], grid)
        with torch.no_grad():
            f_proj = (
                model(torch.from_numpy(proj[proj_mask]).to(cfg.device))
                .detach()
                .cpu()
                .numpy()
            )
        proj_residual = np.abs(f_proj).reshape(-1)
        proj_steps_mean = float(np.mean(steps[proj_mask]))
    else:
        proj_to_trueproj = np.array([float("nan")])
        proj_final_true_dist = np.array([float("nan")])
        proj_residual = np.array([float("nan")])
        proj_steps_mean = float("nan")

    tau = cfg.eval_tau_ratio * float(np.mean(grid.max(axis=0) - grid.min(axis=0)))
    near = d_true_eval < tau
    pred_zero = np.abs(f_eval) < eval_eps_used
    coverage = float(np.mean(pred_zero[near])) if np.any(near) else float("nan")
    false_pos = float(np.mean(pred_zero[~near])) if np.any(~near) else float("nan")
    if np.any(pred_zero):
        pred_on_true_dist = float(np.mean(d_true_eval[pred_zero]))
        pred_on_true_ratio = float(np.mean(near[pred_zero]))
    else:
        pred_on_true_dist = float("nan")
        pred_on_true_ratio = float("nan")

    return {
        "on_mean_v": on_mean_v,
        "proj_manifold_dist": float(np.mean(proj_final_true_dist)),
        "proj_v_residual": float(np.mean(proj_residual)),
        "proj_true_dist": float(np.mean(proj_to_trueproj)),
        "corr_v_d2": corr,
        "slope_v_d2": slope,
        "proj_steps": proj_steps_mean,
        "pred_recall": coverage,
        "FPrate": false_pos,
        "pred_manifold_dist": pred_on_true_dist,
        "pred_precision": pred_on_true_ratio,
        "eval_eps_used": eval_eps_used,
    }




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
    if cfg.use_adaptive_sigma:
        knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x))
        sigma_per_point = compute_adaptive_sigma(x, cfg, knn_norm_estimation_points=knn_norm_estimation_points)
        r_kappa_per_point = sigma_per_point
        r_pos_per_point, r_neg_per_point = compute_adaptive_sigma_bounds(
            x, n_hat, sigma_per_point, cfg, knn_norm_estimation_points
        )
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
    x_ref = x_t.to(cfg.device)
    for epoch in range(cfg.epochs):
        for xb, nb, idxb in dl:
            xb = xb.to(cfg.device)
            nb = nb.to(cfg.device)
            idxb = idxb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
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
    zero_level_eps: float,
    grid: np.ndarray | None = None,
    eval_points: np.ndarray | None = None,
    worst_frac: float = 0.002,
) -> None:
    start_marker_size = 14
    worst_x0 = x0
    worst_traj = traj
    worst_idx = np.array([], dtype=np.int64)
    if (
        grid is not None
        and len(grid) > 0
        and eval_points is not None
        and len(eval_points) > 0
    ):
        eval_t = torch.from_numpy(eval_points).to(cfg.device)
        worst_traj_t, _ = project_trajectory(
            model, eval_t, cfg, f_abs_stop=float(zero_level_eps)
        )
        worst_traj = worst_traj_t.detach().cpu().numpy()
        worst_x0 = eval_points
        n_worst = max(1, int(math.ceil(len(eval_points) * float(worst_frac))))
        d_final = true_distance(worst_traj[-1], grid)
        worst_idx = np.argsort(d_final)[-n_worst:]

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
                color="green",
                linewidth=0.8,
            )
            ax.scatter(
                traj[:, i, 0],
                traj[:, i, 1],
                traj[:, i, 2],
                color="green",
                s=8,
                alpha=0.8,
            )
            ax.scatter(
                x0[i, 0], x0[i, 1], x0[i, 2], color="green", s=start_marker_size
            )
        for i in worst_idx:
            ax.plot(
                worst_traj[:, i, 0],
                worst_traj[:, i, 1],
                worst_traj[:, i, 2],
                color="red",
                linewidth=0.9,
            )
            ax.scatter(
                worst_traj[:, i, 0],
                worst_traj[:, i, 1],
                worst_traj[:, i, 2],
                color="red",
                s=8,
                alpha=0.9,
            )
            ax.scatter(
                worst_x0[i, 0],
                worst_x0[i, 1],
                worst_x0[i, 2],
                color="red",
                s=start_marker_size,
            )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    x_plot_min, x_plot_max = -4.0, 4.0
    y_plot_min, y_plot_max = -3.0, 3.0
    xx, yy = np.meshgrid(
        np.linspace(x_plot_min, x_plot_max, 300),
        np.linspace(y_plot_min, y_plot_max, 300),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(cfg.device)
    with torch.no_grad():
        f_grid = model(grid_t).cpu().numpy().reshape(xx.shape)
        v = (0.5 * (f_grid ** 2)).reshape(xx.shape)
    v_max = float(np.percentile(v, 95))
    u = np.linspace(0.0, 1.0, 25)
    levels = (u ** 3) * v_max

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    # Zero level set: f(x)=0, not V(x)=0 (numerically more robust)
    eps = float(zero_level_eps)
    plt.contourf(
        xx,
        yy,
        f_grid,
        levels=[-eps, eps],
        colors=["#ffa500"],
        alpha=0.55,
    )
    plt.scatter(
        x_train[:, 0], x_train[:, 1], s=6, alpha=0.6, label="data", zorder=3
    )
    _draw_sigma_interval_segments_on_contour(x_train, cfg, n_segments=45)
    for i in range(traj.shape[1]):
        plt.scatter(
            x0[i, 0],
            x0[i, 1],
            c="green",
            s=start_marker_size,
            label="random starts" if i == 0 else None,
        )
        plt.plot(
            traj[:, i, 0],
            traj[:, i, 1],
            "-",
            color="green",
            linewidth=0.8,
            label="random traj" if i == 0 else None,
        )
        plt.scatter(traj[:, i, 0], traj[:, i, 1], c="green", s=4, alpha=0.8)
    for j, i in enumerate(worst_idx):
        plt.scatter(
            worst_x0[i, 0],
            worst_x0[i, 1],
            c="red",
            s=start_marker_size,
            label="worst starts (top 5%)" if j == 0 else None,
        )
        plt.plot(
            worst_traj[:, i, 0],
            worst_traj[:, i, 1],
            "-",
            color="red",
            linewidth=0.9,
            label="worst traj (top 5%)" if j == 0 else None,
        )
        plt.scatter(
            worst_traj[:, i, 0], worst_traj[:, i, 1], c="red", s=4, alpha=0.9
        )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _draw_sigma_interval_segments_on_contour(
    x_train: np.ndarray, cfg: Config, n_segments: int = 15
) -> None:
    if x_train.shape[1] != 2 or len(x_train) == 0:
        return

    n_train = len(x_train)
    k = effective_knn_norm_estimation_points(cfg, n_train)
    n_hat, _, _ = knn_normals_with_quality(x_train, k, cfg)

    sigma_per_point = None
    if cfg.use_adaptive_sigma:
        sigma_per_point = compute_adaptive_sigma(
            x_train, cfg, knn_norm_estimation_points=k
        )

    if sigma_per_point is not None:
        r_pos, r_neg = compute_adaptive_sigma_bounds(
            x_train, n_hat, sigma_per_point, cfg, k
        )
    else:
        base = float(np.max(_as_sigmas(cfg.sigmas)))
        r_pos = np.full(n_train, base, dtype=np.float32)
        r_neg = np.full(n_train, base, dtype=np.float32)

    idx_list = [
        min(n_train - 1, int(round((n_train - 1) * r)))
        for r in np.linspace(0.05, 0.95, max(1, int(n_segments)))
    ]
    for idx in idx_list:
        nvec = n_hat[idx]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        p_pos = x_train[idx] + nvec * float(r_pos[idx])
        p_neg = x_train[idx] - nvec * float(r_neg[idx])
        plt.plot(
            [p_neg[0], p_pos[0]],
            [p_neg[1], p_pos[1]],
            color="gray",
            linewidth=1.0,
            alpha=0.95,
            zorder=2,
        )
        plt.scatter(
            [p_neg[0], p_pos[0]],
            [p_neg[1], p_pos[1]],
            s=4,
            color="black",
            alpha=0.95,
            zorder=2,
        )


def plot_contour_only(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Config,
    out_path: str,
    title: str,
    zero_level_eps: float,
    support_vectors: np.ndarray | None = None,
) -> None:
    if x_train.shape[1] != 2:
        return
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    x_plot_min, x_plot_max = -4.0, 4.0
    y_plot_min, y_plot_max = -3.0, 3.0
    xx, yy = np.meshgrid(
        np.linspace(x_plot_min, x_plot_max, 300),
        np.linspace(y_plot_min, y_plot_max, 300),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(cfg.device)
    with torch.no_grad():
        f_grid = model(grid_t).cpu().numpy().reshape(xx.shape)
        v = (0.5 * (f_grid ** 2)).reshape(xx.shape)
    v_max = float(np.percentile(v, 95))
    u = np.linspace(0.0, 1.0, 25)
    levels = (u ** 3) * v_max

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    eps = float(zero_level_eps)
    plt.contourf(
        xx,
        yy,
        f_grid,
        levels=[-eps, eps],
        colors=["#ffa500"],
        alpha=0.55,
    )
    plt.scatter(
        x_train[:, 0], x_train[:, 1], s=4, alpha=0.6, label="data", zorder=3
    )
    if support_vectors is not None and len(support_vectors) > 0:
        plt.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=20,
            c="gold",
            edgecolors="black",
            linewidths=0.4,
            label="sv",
            zorder=4,
        )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
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


def plot_worst_distance_contour(
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    x0: np.ndarray,
    traj: np.ndarray,
    cfg: Config,
    out_path: str,
    title: str,
    zero_level_eps: float,
    frac: float = 0.05,
) -> None:
    if x0.shape[1] == 3:
        return
    n_pts = traj.shape[1]
    k = max(1, int(math.ceil(n_pts * frac)))
    d_final = true_distance(traj[-1], grid)
    idx = np.argsort(d_final)[-k:]

    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    x_plot_min, x_plot_max = -4.0, 4.0
    y_plot_min, y_plot_max = -3.0, 3.0
    xx, yy = np.meshgrid(
        np.linspace(x_plot_min, x_plot_max, 300),
        np.linspace(y_plot_min, y_plot_max, 300),
    )
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid_eval).to(cfg.device)
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
    eps = float(zero_level_eps)
    plt.contourf(
        xx,
        yy,
        f_grid,
        levels=[-eps, eps],
        colors=["#ffa500"],
        alpha=0.55,
    )
    plt.scatter(
        grid[:, 0], grid[:, 1], s=8, alpha=0.4, label="grid", zorder=3
    )
    for i in idx:
        plt.plot(
            traj[:, i, 0],
            traj[:, i, 1],
            "-",
            color="red",
            linewidth=0.9,
        )
        plt.scatter(
            traj[-1, i, 0],
            traj[-1, i, 1],
            c="red",
            s=25,
            zorder=4,
        )
        plt.text(
            traj[-1, i, 0],
            traj[-1, i, 1],
            f"{d_final[i]:.3f}",
            fontsize=7,
            color="red",
        )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.title(title)
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
    sigma = float(np.max(_as_sigmas(cfg.sigmas)))
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
    sigma_per_point: np.ndarray | None = None,
) -> None:
    d2 = pairwise_sqdist(x, x)
    knn_off_data_filter_points = effective_knn_off_data_filter_points(cfg, len(x))
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
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=6, alpha=0.35, color="gray")
    else:
        plt.figure(figsize=(6, 5))
        plt.scatter(x[:, 0], x[:, 1], s=6, alpha=0.35, color="gray")
    for idx in idx_list:
        if cfg.use_radius_knn:
            nn_idx = _radius_knn_indices(
                d2[idx], cfg.radius_knn_k, cfg.radius_knn_scale
            )
        else:
            nn_idx = np.argsort(d2, axis=1)[idx, 1 : k + 1]
        nbrs = x[nn_idx]
        _, evecs, _, _ = _local_pca_frame(nbrs, x[idx], cfg=cfg)
        nvec = evecs[:, 0]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)

        if x.shape[1] == 3:
            ax.scatter(nbrs[:, 0], nbrs[:, 1], nbrs[:, 2], s=6, color="blue")
            ax.scatter(x[idx, 0], x[idx, 1], x[idx, 2], s=9, color="red")
        else:
            plt.scatter(nbrs[:, 0], nbrs[:, 1], s=6, color="blue")
            plt.scatter(x[idx, 0], x[idx, 1], s=9, color="red")
        # Sample delta points along the normal direction
        m = 30
        r_pos_vis = None
        r_neg_vis = None
        if sigma_per_point is not None:
            r = float(sigma_per_point[idx])
            if cfg.use_adaptive_sigma and cfg.adp_sigma_asymmetric_enable and x.shape[1] == 2:
                order = np.argsort(d2[idx])
                rank = min(
                    len(order) - 1,
                    max(1, int(k) + int(cfg.adp_sigma_nonlocal_offset)),
                )
                j = int(order[rank])
                side = float(np.dot(x[j] - x[idx], nvec))
                danger = float(r * cfg.adp_sigma_danger_scale)
                safe = float(r * cfg.adp_sigma_safe_scale)
                if side >= 0.0:
                    r_pos_vis, r_neg_vis = danger, safe
                else:
                    r_pos_vis, r_neg_vis = safe, danger
                r_pos_vis = float(np.clip(r_pos_vis, cfg.adp_sigma_r_min, cfg.adp_sigma_r_max))
                r_neg_vis = float(np.clip(r_neg_vis, cfg.adp_sigma_r_min, cfg.adp_sigma_r_max))
                z = np.random.randn(m, 1).astype(np.float32)
                s = np.where(z >= 0.0, z * (r_pos_vis / 1.645), z * (r_neg_vis / 1.645)).astype(
                    np.float32
                )
                s = np.clip(s, -r_neg_vis, r_pos_vis).astype(np.float32)
            else:
                r_pos_vis = r
                r_neg_vis = r
                s = np.random.randn(m, 1).astype(np.float32) * (r / 1.645)
        else:
            sigmas = _as_sigmas(cfg.sigmas)
            sigma = np.random.choice(sigmas, size=(m, 1))
            s = np.random.randn(m, 1).astype(np.float32) * sigma
            r_pos_vis = 0.35
            r_neg_vis = 0.35
        delta_pts = x[idx : idx + 1] + s * nvec.reshape(1, -1)
        # A point is "pass" only if it satisfies all enabled checks.
        mask = np.ones(len(delta_pts), dtype=bool)
        if cfg.use_adaptive_sigma:
            if r_pos_vis is not None and r_neg_vis is not None:
                sv = s[:, 0]
                mask &= (sv >= -r_neg_vis) & (sv <= r_pos_vis)
            else:
                r_kappa = float(sigma_per_point[idx]) if sigma_per_point is not None else 0.0
                mask &= (np.abs(s)[:, 0] <= r_kappa)
        if cfg.use_knn_filter:
            d2_off = pairwise_sqdist(delta_pts, x)
            nn_idx = np.argsort(d2_off, axis=1)[:, :knn_off_data_filter_points]
            mask &= (nn_idx == idx).any(axis=1)
        pass_pts = delta_pts[mask]
        fail_pts = delta_pts[~mask]
        if x.shape[1] == 3:
            if len(pass_pts) > 0:
                ax.scatter(
                    pass_pts[:, 0],
                    pass_pts[:, 1],
                    pass_pts[:, 2],
                    s=3,
                    color="green",
                    alpha=0.5,
                )
            if len(fail_pts) > 0:
                ax.scatter(
                    fail_pts[:, 0],
                    fail_pts[:, 1],
                    fail_pts[:, 2],
                    s=3,
                    color="orange",
                    alpha=0.5,
                )
        else:
            if len(pass_pts) > 0:
                plt.scatter(
                    pass_pts[:, 0], pass_pts[:, 1], s=3, color="green", alpha=0.45
                )
            if len(fail_pts) > 0:
                plt.scatter(
                    fail_pts[:, 0], fail_pts[:, 1], s=3, color="orange", alpha=0.45
                )
        # Replace arrow/circle with one gray segment showing clipping interval on normal.
        if x.shape[1] != 3:
            rp = float(r_pos_vis) if r_pos_vis is not None else 0.35
            rn = float(r_neg_vis) if r_neg_vis is not None else 0.35
            p_pos = x[idx] + nvec * rp
            p_neg = x[idx] - nvec * rn
            plt.plot(
                [p_neg[0], p_pos[0]],
                [p_neg[1], p_pos[1]],
                color="gray",
                linewidth=1.1,
                alpha=0.95,
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
    knn_off_data_filter_points = effective_knn_off_data_filter_points(cfg, len(x))
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

    sigmas = _as_sigmas(cfg.sigmas)
    for idx in idx_list:
        m = 30
        sigma = np.random.choice(sigmas, size=(m, 1))
        s = np.random.randn(m, 1).astype(np.float32) * sigma
        delta_pts = x[idx : idx + 1] + s * n_hat[idx].reshape(1, -1)
        d2_off = pairwise_sqdist(delta_pts, x)
        nn_idx = np.argsort(d2_off, axis=1)[:, :knn_off_data_filter_points]
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
    datasets = [
        "sharp_star",
        "figure_eight",
        "ellipse",
        "discontinuous",
        "noise_only",
        "sparse_only",
        "sine",
        "looped_spiro",

        # "hetero_noise",

    ]
    methods = [
        "energy",
        # "margin",
        # "delta",
    ]
    output_root = "outputs_levelset_datasets"
    os.makedirs(output_root, exist_ok=True)
    eval_rng = np.random.default_rng(cfg.eval_seed)
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
        x_train, grid = generate_dataset(name, cfg)
        x_eval = sample_eval_points(x_train, grid, cfg, eval_rng)
        model = None
        stats = {}
        history = {}
        if "energy" in methods:
            model, stats, n_hat, quality, thickness, history = train_with_data(
                cfg, x_train
            )
        else:
            knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x_train))
            n_hat, quality, thickness = knn_normals_with_quality(
                x_train, knn_norm_estimation_points, cfg
            )

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
            plan_rng = np.random.default_rng(cfg.eval_seed + 77)
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
        if model is not None:
            proj_eps_used = compute_eval_eps_used(model, x_train, cfg)
            traj_t, steps = project_trajectory(model, x0_t, cfg, f_abs_stop=proj_eps_used)
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
                zero_level_eps=proj_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
            )
            plot_distance_curves(
                traj,
                grid,
                out_path=os.path.join(out_dir, "energy_distance_curves.png"),
                title=f"{name}: Energy Model Distances",
            )
            plot_worst_distance_contour(
                model,
                x_train,
                grid,
                x0,
                traj,
                cfg,
                out_path=os.path.join(out_dir, "energy_worst_distance_contour.png"),
                title=f"{name}: Energy Model Worst Distance (top 5%)",
                zero_level_eps=proj_eps_used,
            )
            if plan_pairs is not None:
                plans_proj = []
                plans_constr = []
                plans_proj_off = []
                plans_constr_off = []
                for x0_p, x1_p in plan_pairs:
                    x0_p = true_projection(x0_p[None, :], grid)[0][0]
                    x1_p = true_projection(x1_p[None, :], grid)[0][0]
                    plans_proj.append(
                        plan_linear_then_project(
                            model, x0_p, x1_p, cfg, f_abs_stop=proj_eps_used
                        )
                    )
                    plans_constr.append(plan_constrained_path(model, x0_p, x1_p, cfg))
                if plan_pairs_off is not None:
                    for x0_p, x1_p in plan_pairs_off:
                        x0_p = true_projection(x0_p[None, :], grid)[0][0]
                        plans_proj_off.append(
                            plan_linear_then_project(
                                model,
                                x0_p,
                                x1_p,
                                cfg,
                                keep_endpoints=False,
                                f_abs_stop=proj_eps_used,
                            )
                        )
                        plans_constr_off.append(
                            plan_constrained_path(model, x0_p, x1_p, cfg)
                        )
                plot_planned_paths(
                    model,
                    x_train,
                    grid,
                    plans_proj,
                    plans_constr,
                    cfg,
                    out_path=os.path.join(out_dir, "energy_planner_paths.png"),
                    title=f"{name}: Planned Paths",
                    zero_level_eps=proj_eps_used,
                )
                if plans_proj_off or plans_constr_off:
                    plot_planned_paths_off(
                        model,
                        x_train,
                        grid,
                        plans_proj_off,
                        plans_constr_off,
                        cfg,
                        out_path=os.path.join(out_dir, "energy_planner_paths_off.png"),
                        title=f"{name}: Planned Paths (Off-Manifold)",
                        zero_level_eps=proj_eps_used,
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
        knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x_train))
        n_train = len(x_train)
        idx_list = [
            min(n_train - 1, int(round((n_train - 1) * r)))
            for r in np.linspace(0.05, 0.95, 15)
        ]
        sigma_per_point = None
        if cfg.use_adaptive_sigma:
            sigma_per_point = compute_adaptive_sigma(x_train, cfg, knn_norm_estimation_points=knn_norm_estimation_points)
        plot_knn_normals(
            x_train,
            idx_list=idx_list,
            k=knn_norm_estimation_points,
            out_path=os.path.join(out_dir, "knn_normal.png"),
            title=f"{name}: KNN + Normal",
            cfg=cfg,
            grid=grid,
            sigma_per_point=sigma_per_point,
        )
        if margin_model is not None:
            margin_eps_used = compute_eval_eps_used(margin_model, x_train, cfg)
            margin_traj, _ = project_trajectory(
                margin_model, x0_t, cfg, f_abs_stop=margin_eps_used
            )
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
                zero_level_eps=margin_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
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
                cfg,
                out_path=os.path.join(out_dir, "margin_worst_distance_contour.png"),
                title=f"{name}: Margin Worst Distance (top 5%)",
                zero_level_eps=margin_eps_used,
            )
            plot_loss_curves(
                margin_history,
                out_path=os.path.join(out_dir, "margin_loss_curves.png"),
                title=f"{name}: Margin Baseline Losses",
                cfg=cfg,
            )

        if delta_model is not None:
            delta_eps_used = compute_eval_eps_used(delta_model, x_train, cfg)
            delta_traj, _ = project_trajectory(
                delta_model, x0_t, cfg, f_abs_stop=delta_eps_used
            )
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
                zero_level_eps=delta_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
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
                for x0_p, x1_p in plan_pairs:
                    x0_p = true_projection(x0_p[None, :], grid)[0][0]
                    x1_p = true_projection(x1_p[None, :], grid)[0][0]
                    plans_proj.append(
                        plan_linear_then_project(
                            delta_model, x0_p, x1_p, cfg, f_abs_stop=delta_eps_used
                        )
                    )
                    plans_constr.append(plan_constrained_path(delta_model, x0_p, x1_p, cfg))
                if plan_pairs_off is not None:
                    for x0_p, x1_p in plan_pairs_off:
                        x0_p = true_projection(x0_p[None, :], grid)[0][0]
                        plans_proj_off.append(
                            plan_linear_then_project(
                                delta_model,
                                x0_p,
                                x1_p,
                                cfg,
                                keep_endpoints=False,
                                f_abs_stop=delta_eps_used,
                            )
                        )
                        plans_constr_off.append(
                            plan_constrained_path(delta_model, x0_p, x1_p, cfg)
                        )
                plot_planned_paths(
                    delta_model,
                    x_train,
                    grid,
                    plans_proj,
                    plans_constr,
                    cfg,
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
                        cfg,
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
                cfg,
                out_path=os.path.join(out_dir, "delta_worst_distance_contour.png"),
                title=f"{name}: Delta Worst Distance (top 5%)",
                zero_level_eps=delta_eps_used,
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

        if model is not None:
            metrics_map = evaluate_metrics(model, x_train, grid, x_eval, cfg)
            eval_results.append(
                {
                    "dataset": name,
                    "method": "energy",
                    "metrics": metrics_map,
                }
            )
            if wb_run is not None:
                wb_step += 1
                wandb.log(
                    {f"{name}/energy/{k}": v for k, v in metrics_map.items()},
                    step=wb_step,
                )
        if margin_model is not None:
            metrics_map = evaluate_metrics(margin_model, x_train, grid, x_eval, cfg)
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
            metrics_map = evaluate_metrics(delta_model, x_train, grid, x_eval, cfg)
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
            line = (
                f"{entry['method']:<8} {entry['dataset']:<14} "
                f"on_mean_v={m['on_mean_v']:.4f} "
                f"proj_manifold_dist={m['proj_manifold_dist']:.4f} "
                f"proj_v_residual={m['proj_v_residual']:.4f} "
                f"proj_true_dist={m['proj_true_dist']:.4f} "
                f"corr_v_d2={m['corr_v_d2']:.4f} "
                f"slope_v_d2={m['slope_v_d2']:.4f} "
                f"proj_steps={m['proj_steps']:.2f} "
                f"pred_recall={m['pred_recall']:.4f} "
                f"FPrate={m['FPrate']:.4f}"
            )
            lines.append(line)
            print(line)
        lines.append("")
        lines.append("=== Evaluation Metrics (mean over datasets) ===")
        print("\n=== Evaluation Metrics (mean over datasets) ===")
        by_method: Dict[str, List[Dict[str, float]]] = {}
        for entry in eval_results:
            by_method.setdefault(entry["method"], []).append(entry["metrics"])
        for method, items in by_method.items():
            keys = items[0].keys()
            avg = {k: float(np.mean([it[k] for it in items])) for k in keys}
            norm_vals = []
            for entry in eval_results:
                if entry["method"] != method:
                    continue
                ds = entry["dataset"]
                base = NORM_PROJ_BASELINE.get(ds)
                if base is None or base["std"] <= 0:
                    continue
                val = entry["metrics"]["proj_manifold_dist"]
                norm_vals.append((val - base["mean"]) / base["std"])
            avg["norm_proj_manifold_dist"] = float(np.mean(norm_vals)) if norm_vals else float("nan")
            line = (
                f"{method:<8} "
                f"on_mean_v={avg['on_mean_v']:.4f} "
                f"proj_manifold_dist={avg['proj_manifold_dist']:.4f} "
                f"proj_v_residual={avg['proj_v_residual']:.4f} "
                f"proj_true_dist={avg['proj_true_dist']:.4f} "
                f"corr_v_d2={avg['corr_v_d2']:.4f} "
                f"slope_v_d2={avg['slope_v_d2']:.4f} "
                f"proj_steps={avg['proj_steps']:.2f} "
                f"pred_recall={avg['pred_recall']:.4f} "
                f"FPrate={avg['FPrate']:.4f} "
                f"norm_proj_manifold_dist={avg['norm_proj_manifold_dist']:.4f}"
            )
            lines.append(line)
            print(line)
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
    main()
