#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Tuple

import numpy as np
import torch

LIFT_3D_VZ_DEFAULTS = {
    "z_amp1": 0.35,
    "z_amp2": 0.20,
    "z_freq1": 1.5,
    "z_freq2": 1.2,
}

TRAJ_3D_CODIM1_BASES = {
    "3d_paraboloid",
    "3d_twosphere",
    "3d_saddle_surface",
    "3d_sphere_surface",
    "3d_torus_surface",
    "3d_planar_arm_line_n3",
    "3d_spatial_arm_plane_n3",
}


@dataclass
class DatasetSpec:
    name: str
    dim: int
    latent_dim_default: int
    train_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_off_sampler: Callable[[int, np.random.Generator], np.ndarray]
    gt_distance_fn: Callable[[np.ndarray], np.ndarray]
    plot_bounds: Tuple[np.ndarray, np.ndarray]


def get_lift_3d_vz_params(cfg=None) -> dict[str, float]:
    p = dict(LIFT_3D_VZ_DEFAULTS)
    if cfg is None:
        return p
    for k in ("z_amp1", "z_amp2", "z_freq1", "z_freq2"):
        if hasattr(cfg, k):
            p[k] = float(getattr(cfg, k))
    return p


def lift_xy_to_3d_var(x2: np.ndarray, cfg=None) -> np.ndarray:
    p = get_lift_3d_vz_params(cfg)
    x = x2[:, 0:1].astype(np.float32)
    y = x2[:, 1:2].astype(np.float32)
    z = (
        float(p["z_amp1"]) * np.sin(float(p["z_freq1"]) * x)
        + float(p["z_amp2"]) * np.cos(float(p["z_freq2"]) * y)
    ).astype(np.float32)
    return np.concatenate([x2.astype(np.float32), z], axis=1)


def lift_xy_to_3d_zero(x2: np.ndarray) -> np.ndarray:
    z = np.zeros((x2.shape[0], 1), dtype=np.float32)
    return np.concatenate([x2.astype(np.float32), z], axis=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_spiral_on(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_spiral_off(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0
    x += rng.uniform(0.4, 1.0, size=n) * rng.choice([-1, 1], size=n)
    y += rng.uniform(0.4, 1.0, size=n) * rng.choice([-1, 1], size=n)
    z += rng.uniform(0.4, 1.0, size=n) * rng.choice([-1, 1], size=n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_sphere_on(
    n: int,
    rng: np.random.Generator,
    radius: float = 1.0,
    center=(0.0, 0.0, 0.0),
) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    pts = radius * v + np.array(center, dtype=float)[None, :]
    return pts.astype(np.float32)


def sample_paraboloid_on(
    n: int,
    rng: np.random.Generator,
    xy_range: float = 1.2,
    z_scale: float = 1.0,
) -> np.ndarray:
    x = rng.uniform(-xy_range, xy_range, size=n)
    y = rng.uniform(-xy_range, xy_range, size=n)
    z = z_scale * (x ** 2 + y ** 2)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_two_sphere_outer_on(n: int, rng: np.random.Generator) -> np.ndarray:
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0
    m = int(n * 2.2) + 64
    pts_a = sample_sphere_on(m, rng, radius=r, center=ca)
    pts_b = sample_sphere_on(m, rng, radius=r, center=cb)
    # Keep only the boundary of union(A, B): remove sphere patches that lie
    # strictly inside the other sphere.
    db_a = np.linalg.norm(pts_a - cb[None, :], axis=1)
    da_b = np.linalg.norm(pts_b - ca[None, :], axis=1)
    keep_a = db_a >= (r - 1e-6)
    keep_b = da_b >= (r - 1e-6)
    pts = np.concatenate([pts_a[keep_a], pts_b[keep_b]], axis=0)
    if pts.shape[0] < n:
        return pts.astype(np.float32)
    idx = rng.choice(pts.shape[0], size=n, replace=False)
    return pts[idx].astype(np.float32)


def sample_square_on(n: int, rng: np.random.Generator, half: float = 1.0) -> np.ndarray:
    edge = rng.integers(0, 4, size=n)
    u = rng.uniform(-half, half, size=n)
    x = np.empty(n)
    y = np.empty(n)
    m = edge == 0
    x[m], y[m] = u[m], half
    m = edge == 1
    x[m], y[m] = u[m], -half
    m = edge == 2
    x[m], y[m] = half, u[m]
    m = edge == 3
    x[m], y[m] = -half, u[m]
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_paraboloid_off(
    n: int,
    rng: np.random.Generator,
    xy_range: float = 1.2,
    z_max: float = 3.0,
) -> np.ndarray:
    x = rng.uniform(-xy_range, xy_range, size=n)
    y = rng.uniform(-xy_range, xy_range, size=n)
    z_surface = x ** 2 + y ** 2
    sign = rng.choice([-1.0, 1.0], size=n)
    delta = rng.uniform(0.35, 1.0, size=n)
    z = np.clip(z_surface + sign * delta, 0.0, z_max)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_two_sphere_outer_off(n: int, rng: np.random.Generator, max_iters: int = 50) -> np.ndarray:
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0
    lo = np.array([-2.5, -2.0, -2.0])
    hi = np.array([+2.5, +2.0, +2.0])
    out = []
    for _ in range(max_iters):
        if len(out) >= n:
            break
        m = int((n - len(out)) * 1.8) + 32
        pts = rng.uniform(lo, hi, size=(m, 3))
        da = np.linalg.norm(pts - ca[None, :], axis=1)
        db = np.linalg.norm(pts - cb[None, :], axis=1)
        dist_to_boundary = np.minimum(np.abs(da - r), np.abs(db - r))
        sel = pts[dist_to_boundary > 0.25]
        out.append(sel)
        out = [np.concatenate(out, axis=0)[:n]]
    if len(out) < n:
        missing = n - len(out)
        out.append(rng.uniform(lo, hi, size=(missing, 3)))
        out = [np.concatenate(out, axis=0)[:n]]
    return out[0].astype(np.float32)


def sample_square_off(n: int, rng: np.random.Generator, half: float = 1.0, max_iters: int = 50) -> np.ndarray:
    out = []
    for _ in range(max_iters):
        if len(out) >= n:
            break
        m = int((n - len(out)) * 1.6) + 16
        pts = rng.uniform(-1.8 * half, 1.8 * half, size=(m, 2))
        linf = np.maximum(np.abs(pts[:, 0]), np.abs(pts[:, 1]))
        sel = pts[np.abs(linf - half) > 0.18 * half]
        out.append(sel)
        out = [np.concatenate(out, axis=0)[:n]]
    if len(out) < n:
        missing = n - len(out)
        out.append(rng.uniform(-1.8 * half, 1.8 * half, size=(missing, 2)))
        out = [np.concatenate(out, axis=0)[:n]]
    return out[0].astype(np.float32)


def gt_dist_spiral(points: np.ndarray, theta_samples: int = 2048) -> np.ndarray:
    theta = np.linspace(0.0, 4.0 * np.pi, num=theta_samples, dtype=np.float32)
    a = 0.25
    curve = np.stack([np.cos(theta), np.sin(theta), a * theta - 2.0], axis=1).astype(np.float32)
    d2 = np.sum((points[:, None, :] - curve[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def gt_dist_paraboloid(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.abs(z - (x ** 2 + y ** 2))


def gt_dist_two_sphere_outer(points: np.ndarray) -> np.ndarray:
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0
    da = np.linalg.norm(points - ca[None, :], axis=1)
    db = np.linalg.norm(points - cb[None, :], axis=1)
    return np.minimum(np.abs(da - r), np.abs(db - r))


def gt_dist_square(points: np.ndarray, half: float = 1.0) -> np.ndarray:
    linf = np.maximum(np.abs(points[:, 0]), np.abs(points[:, 1]))
    return np.abs(linf - half)


def build_datasets() -> Dict[str, DatasetSpec]:
    ds: Dict[str, DatasetSpec] = {}
    ds["3d_spiral"] = DatasetSpec(
        name="3d_spiral",
        dim=3,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_spiral_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_spiral(x),
        plot_bounds=(np.array([-2.2, -2.2, -2.8]), np.array([2.2, 2.2, 2.8])),
    )
    ds["3d_paraboloid"] = DatasetSpec(
        name="3d_paraboloid",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_off_sampler=lambda n, rng: sample_paraboloid_off(n, rng, xy_range=1.2, z_max=3.0),
        gt_distance_fn=lambda x: gt_dist_paraboloid(x),
        plot_bounds=(np.array([-1.8, -1.8, 0.0]), np.array([1.8, 1.8, 3.2])),
    )
    ds["3d_twosphere"] = DatasetSpec(
        name="3d_twosphere",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_two_sphere_outer_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_two_sphere_outer(x),
        plot_bounds=(np.array([-2.8, -2.2, -2.2]), np.array([2.8, 2.2, 2.2])),
    )
    ds["2d_square"] = DatasetSpec(
        name="2d_square",
        dim=2,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_off_sampler=lambda n, rng: sample_square_off(n, rng, half=1.0),
        gt_distance_fn=lambda x: gt_dist_square(x, half=1.0),
        plot_bounds=(np.array([-2.3, -2.3]), np.array([2.3, 2.3])),
    )
    return ds


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


def _planar_arm_line_n2(cfg) -> Tuple[np.ndarray, np.ndarray]:
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


def _planar_arm_line_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
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


def _spatial_arm_plane_n4(cfg) -> Tuple[np.ndarray, np.ndarray]:
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


def _spatial_arm_plane_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
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


def _spatial_arm_ellip_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 3-DoF arm (q1 yaw, q2/q3 pitch chain) constrained to workspace ellipse:
    # x = a cos(t), y = b sin(t), z = z0  (codim=2 in 3D joint space).
    # Keep old dataset alias for backward compatibility.
    l1, l2 = 1.0, 0.8
    a = 1.35
    b = 0.95
    z0 = 0.35

    def _build(n_target: int, *, use_grid: bool = False) -> np.ndarray:
        n_target = max(1, int(n_target))
        if use_grid:
            t = np.linspace(-math.pi, math.pi, n_target, endpoint=False, dtype=np.float32)
        else:
            t = np.random.uniform(-math.pi, math.pi, size=(n_target,)).astype(np.float32)

        x = (a * np.cos(t)).astype(np.float32)
        y = (b * np.sin(t)).astype(np.float32)
        rho = np.sqrt(np.maximum(x * x + y * y, 1e-8)).astype(np.float32)
        q1 = np.arctan2(y, x).astype(np.float32)

        rho2 = rho * rho + z0 * z0
        c3 = (rho2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        c3 = np.clip(c3, -1.0, 1.0).astype(np.float32)
        s3_mag = np.sqrt(np.maximum(0.0, 1.0 - c3 * c3)).astype(np.float32)

        pick_elbow_up = np.random.rand(n_target) < 0.5
        s3 = np.where(pick_elbow_up, s3_mag, -s3_mag).astype(np.float32)
        q3 = np.arctan2(s3, c3).astype(np.float32)
        q2 = (
            np.arctan2(np.full_like(rho, z0, dtype=np.float32), rho)
            - np.arctan2(l2 * s3, l1 + l2 * c3)
        ).astype(np.float32)
        return np.stack([_wrap_to_pi(q1), _wrap_to_pi(q2), _wrap_to_pi(q3)], axis=1).astype(np.float32)

    x_train = _build(cfg.n_train, use_grid=False)
    grid = _build(cfg.n_grid, use_grid=True)
    return x_train, grid


def _spatial_arm_up_n6(cfg) -> Tuple[np.ndarray, np.ndarray]:
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


def _surface_normal_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Wave surface: z = a1*sin(fx*x) + a2*cos(fy*y)
    a1, a2 = 0.55, 0.35
    fx, fy = 1.2, 1.0
    dzdx = a1 * fx * np.cos(fx * x)
    dzdy = -a2 * fy * np.sin(fy * y)
    n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=1).astype(np.float64)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return n.astype(np.float32)


def _rpy_from_rotmat_zyx(R: np.ndarray) -> np.ndarray:
    # R = Rz(yaw) Ry(pitch) Rx(roll)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _workspace_sine_surface_pose_n6(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 6D workspace pose: [x, y, z, roll, pitch, yaw]
    # Constraints:
    # 1) position lies on wave surface z=f(x,y)
    # 2) orientation's local z-axis aligns with surface normal
    # 3) free spin psi around normal remains unconstrained
    def _sample(n: int, seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(int(cfg.seed) + seed_offset)
        x = rng.uniform(-2.0, 2.0, size=(n,)).astype(np.float32)
        y = rng.uniform(-2.0, 2.0, size=(n,)).astype(np.float32)
        a1, a2 = 0.55, 0.35
        fx, fy = 1.2, 1.0
        z = (a1 * np.sin(fx * x) + a2 * np.cos(fy * y)).astype(np.float32)

        nvec = _surface_normal_from_xy(x, y).astype(np.float64)
        t1 = np.stack([np.ones_like(x), np.zeros_like(x), a1 * fx * np.cos(fx * x)], axis=1).astype(np.float64)
        t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
        t2 = np.cross(nvec, t1)
        t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)

        psi = rng.uniform(-math.pi, math.pi, size=(n,)).astype(np.float64)
        c = np.cos(psi)[:, None]
        s = np.sin(psi)[:, None]
        x_axis = c * t1 + s * t2
        y_axis = -s * t1 + c * t2
        z_axis = nvec

        rpy = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            R = np.stack([x_axis[i], y_axis[i], z_axis[i]], axis=1).astype(np.float64)
            rpy[i] = _rpy_from_rotmat_zyx(R)
        rpy = _wrap_to_pi(rpy.astype(np.float32))
        return np.concatenate([x[:, None], y[:, None], z[:, None], rpy], axis=1).astype(np.float32)

    x_train = _sample(max(1, int(cfg.n_train)), seed_offset=0)
    grid = _sample(max(1, int(cfg.n_grid)), seed_offset=1)
    return x_train, grid


def _workspace_sine_surface_pose_n6_traj(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # Build GT reference points with the original sampler first, then generate
    # xyz trajectories on the same sine surface and assign smooth orientation.
    _, grid = _workspace_sine_surface_pose_n6(cfg)
    xyz_grid = grid[:, :3].astype(np.float32)

    n_train = max(1, int(cfg.n_train))
    traj_count = int(max(4, min(96, getattr(cfg, "traj_count", max(12, n_train // 64)))))
    traj_len = int(max(8, getattr(cfg, "traj_len", int(math.ceil(n_train / max(traj_count, 1))))))
    traj_knn = int(max(4, getattr(cfg, "traj_knn", 20)))
    xyz_train = _traj_points_from_grid(
        xyz_grid,
        n_train=n_train,
        seed=int(getattr(cfg, "seed", 0)),
        traj_count=traj_count,
        traj_len=traj_len,
        traj_knn=traj_knn,
    ).astype(np.float32)

    x = xyz_train[:, 0].astype(np.float32)
    y = xyz_train[:, 1].astype(np.float32)
    nvec = _surface_normal_from_xy(x, y).astype(np.float64)
    a1, fx = 0.55, 1.2
    t1 = np.stack([np.ones_like(x), np.zeros_like(x), a1 * fx * np.cos(fx * x)], axis=1).astype(np.float64)
    t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
    t2 = np.cross(nvec, t1)
    t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)

    rng = np.random.default_rng(int(cfg.seed) + 11)
    psi_step_std = float(max(1e-4, getattr(cfg, "traj_psi_step_std", 0.07)))
    psi = np.zeros((n_train,), dtype=np.float64)
    seg_len = max(2, int(math.ceil(n_train / max(traj_count, 1))))
    for k in range(traj_count):
        a = k * seg_len
        b = min(n_train, (k + 1) * seg_len)
        if a >= b:
            continue
        psi0 = rng.uniform(-math.pi, math.pi)
        dpsi = rng.normal(scale=psi_step_std, size=(b - a,)).astype(np.float64)
        psi_seg = psi0 + np.cumsum(dpsi)
        psi[a:b] = psi_seg
    psi = ((psi + math.pi) % (2.0 * math.pi) - math.pi).astype(np.float64)

    c = np.cos(psi)[:, None]
    s = np.sin(psi)[:, None]
    x_axis = c * t1 + s * t2
    y_axis = -s * t1 + c * t2
    z_axis = nvec

    rpy = np.zeros((n_train, 3), dtype=np.float32)
    for i in range(n_train):
        R = np.stack([x_axis[i], y_axis[i], z_axis[i]], axis=1).astype(np.float64)
        rpy[i] = _rpy_from_rotmat_zyx(R)
    rpy = _wrap_to_pi(rpy.astype(np.float32))

    x_train = np.concatenate([xyz_train.astype(np.float32), rpy], axis=1).astype(np.float32)
    return x_train, grid.astype(np.float32)


def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    pts = points.astype(np.float32)
    n = int(pts.shape[0])
    kk = int(max(2, min(k, n)))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(pts.astype(np.float64))
        _, idx = tree.query(pts.astype(np.float64), k=kk)
        if idx.ndim == 1:
            idx = idx[:, None]
        return idx.astype(np.int64)
    except Exception:
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        return idx.astype(np.int64)


def _traj_points_from_grid(
    grid: np.ndarray,
    *,
    n_train: int,
    seed: int,
    traj_count: int,
    traj_len: int,
    traj_knn: int,
) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float32)
    if len(g) == 0:
        return np.zeros((max(1, int(n_train)), g.shape[1] if g.ndim == 2 else 3), dtype=np.float32)

    rng = np.random.default_rng(int(seed) + 971)
    n = max(1, int(n_train))
    n_traj = max(1, int(traj_count))
    t_len = max(2, int(traj_len))
    knn = _knn_indices(g, k=max(2, int(traj_knn)))

    eps = 1e-8
    # Choose diverse starts to improve global coverage.
    starts: list[int] = []
    if len(g) > 0:
        starts.append(int(rng.integers(0, len(g))))
    while len(starts) < n_traj:
        cand = int(rng.integers(0, len(g)))
        if cand in starts:
            continue
        if len(starts) == 0:
            starts.append(cand)
            continue
        d2 = np.sum((g[np.array(starts)] - g[cand : cand + 1]) ** 2, axis=1)
        # Keep starts reasonably far apart.
        if float(np.min(d2)) > 0.03 * float(np.mean(np.var(g, axis=0) + eps)):
            starts.append(cand)
        elif float(rng.uniform()) < 0.08:
            starts.append(cand)

    visit = np.zeros((len(g),), dtype=np.int32)
    seq = []
    for ti in range(n_traj):
        cur = int(starts[ti % len(starts)])
        prev_idx = -1
        prev_dir = None
        for _step in range(t_len):
            seq.append(g[cur].astype(np.float32))
            visit[cur] += 1
            nbr = knn[cur]
            nbr = nbr[nbr != cur]
            if prev_idx >= 0:
                nbr = nbr[nbr != prev_idx]
            if len(nbr) == 0:
                cur = int(rng.integers(0, len(g)))
                prev_idx = -1
                prev_dir = None
                continue

            cand = nbr[: min(14, len(nbr))]
            vec = (g[cand] - g[cur]).astype(np.float32)
            d = np.linalg.norm(vec, axis=1) + eps
            u = vec / d[:, None]
            cov = visit[cand].astype(np.float32)
            # Coverage gain: prefer less visited regions.
            cov_gain = (np.max(cov) - cov) if len(cov) > 0 else np.zeros_like(d)

            if prev_dir is None:
                # First move: prefer local smooth step (short distance).
                score = -d + 0.65 * cov_gain
            else:
                a = np.clip(np.sum(u * prev_dir.reshape(1, -1), axis=1), -1.0, 1.0)
                # Strongly prefer forward continuation, weakly prefer short step.
                score = 3.2 * a - 0.22 * d + 0.90 * cov_gain
                # Occasionally allow controlled turn to avoid dead loops.
                if float(rng.uniform()) < 0.08:
                    score = score + rng.normal(scale=0.35, size=score.shape).astype(np.float32)

            pick = int(np.argmax(score))
            nxt = int(cand[pick])
            new_vec = (g[nxt] - g[cur]).astype(np.float32)
            new_norm = float(np.linalg.norm(new_vec))
            if new_norm > 1e-8:
                new_u = new_vec / new_norm
                if prev_dir is None:
                    prev_dir = new_u.astype(np.float32)
                else:
                    prev_dir = (0.82 * prev_dir + 0.18 * new_u).astype(np.float32)
                    prev_dir = prev_dir / max(float(np.linalg.norm(prev_dir)), 1e-8)
            prev_idx = cur
            cur = nxt

    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((n, g.shape[1]), dtype=np.float32)
    # Keep trajectory ordering in x_train so downstream consumers can visualize
    # contiguous motion segments instead of shuffled manifold points.
    if len(arr) >= n:
        return arr[:n].astype(np.float32)
    rep = int(math.ceil(float(n) / float(len(arr))))
    tiled = np.tile(arr, (rep, 1)).astype(np.float32)
    return tiled[:n].astype(np.float32)


def generate_dataset(name: str, cfg) -> Tuple[np.ndarray, np.ndarray]:
    if name.endswith("_traj"):
        base = str(name)[: -len("_traj")]
        if base in TRAJ_3D_CODIM1_BASES:
            cfg_base = SimpleNamespace(**vars(cfg))
            # Keep dense reference manifold for eval; train set becomes trajectory samples.
            x_base, grid = generate_dataset(base, cfg_base)
            n_train = max(1, int(getattr(cfg, "n_train", len(x_base))))
            traj_count = int(max(4, min(96, getattr(cfg, "traj_count", max(12, n_train // 24)))))
            traj_len = int(max(8, getattr(cfg, "traj_len", int(math.ceil(n_train / max(traj_count, 1))))))
            traj_knn = int(max(4, getattr(cfg, "traj_knn", 16)))
            x_traj = _traj_points_from_grid(
                grid.astype(np.float32),
                n_train=n_train,
                seed=int(getattr(cfg, "seed", 0)),
                traj_count=traj_count,
                traj_len=traj_len,
                traj_knn=traj_knn,
            )
            return x_traj.astype(np.float32), grid.astype(np.float32)

    if name == "3d_spiral":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_spiral_on(int(cfg.n_train), rng_train).astype(np.float32)
        grid = sample_spiral_on(int(cfg.n_grid), rng_grid).astype(np.float32)
        return x, grid
    if name == "3d_paraboloid":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_paraboloid_on(int(cfg.n_train), rng_train, xy_range=1.2, z_scale=1.0).astype(np.float32)
        grid = sample_paraboloid_on(int(cfg.n_grid), rng_grid, xy_range=1.2, z_scale=1.0).astype(np.float32)
        return x, grid
    if name == "3d_twosphere":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_two_sphere_outer_on(int(cfg.n_train), rng_train).astype(np.float32)
        grid = sample_two_sphere_outer_on(int(cfg.n_grid), rng_grid).astype(np.float32)
        return x, grid
    if name == "2d_square":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_square_on(int(cfg.n_train), rng_train, half=1.0).astype(np.float32)
        grid = sample_square_on(int(cfg.n_grid), rng_grid, half=1.0).astype(np.float32)
        return x, grid
    if name == "2d_figure_eight":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        x = np.concatenate([np.sin(t), np.sin(2 * t)], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([np.sin(tg), np.sin(2 * tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_ellipse":
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
    if name == "3d_saddle_surface":
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
    if name == "3d_sphere_surface":
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
    if name == "3d_torus_surface":
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
    if name == "2d_planar_arm_line_n2":
        return _planar_arm_line_n2(cfg)
    if name == "3d_planar_arm_line_n3":
        return _planar_arm_line_n3(cfg)
    if name == "3d_spatial_arm_plane_n3":
        return _spatial_arm_plane_n3(cfg)
    if name in ("3d_spatial_arm_ellip_n3", "3d_spatial_arm_circle_n3"):
        return _spatial_arm_ellip_n3(cfg)
    if name == "6d_spatial_arm_up_n6":
        return _spatial_arm_up_n6(cfg)
    if name == "6d_workspace_sine_surface_pose":
        return _workspace_sine_surface_pose_n6(cfg)
    if name == "6d_workspace_sine_surface_pose_traj":
        return _workspace_sine_surface_pose_n6_traj(cfg)
    if name == "2d_noisy_sine":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        y = np.sin(t) + 0.1 * np.random.randn(cfg.n_train, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_sine":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_sparse_sine":
        n_sparse = max(45, cfg.n_train // 8)
        t = np.random.uniform(-math.pi, math.pi, size=(n_sparse, 1))
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_discontinuous":
        n_half = cfg.n_train // 2
        t1 = np.random.uniform(-math.pi, -0.7, size=(n_half, 1))
        t2 = np.random.uniform(0.7, math.pi, size=(cfg.n_train - n_half, 1))
        t = np.vstack([t1, t2])
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_looped_spiro":
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
    if name == "2d_sharp_star":
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
    if name == "2d_hetero_noise":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        sigma = 0.02 + 0.3 * np.exp(-0.5 * (t / 0.8) ** 2)
        y = np.sin(t) + sigma * np.random.randn(cfg.n_train, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    raise ValueError(f"unknown dataset: {name}")
