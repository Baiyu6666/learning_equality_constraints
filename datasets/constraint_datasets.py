#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch

LIFT_3D_VZ_DEFAULTS = {
    "z_amp1": 0.35,
    "z_amp2": 0.20,
    "z_freq1": 1.5,
    "z_freq2": 1.2,
}


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


def _spatial_arm_circle_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
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


def generate_dataset(name: str, cfg) -> Tuple[np.ndarray, np.ndarray]:
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
    if name == "3d_spatial_arm_circle_n3":
        return _spatial_arm_circle_n3(cfg)
    if name == "4d_spatial_arm_plane_n4":
        return _spatial_arm_plane_n4(cfg)
    if name == "6d_spatial_arm_up_n6":
        return _spatial_arm_up_n6(cfg)
    if name == "2d_noisy_sine":
        n_sparse = max(128, cfg.n_train // 4)
        t = np.random.uniform(-math.pi, math.pi, size=(n_sparse, 1))
        y = np.sin(t) + 0.1 * np.random.randn(n_sparse, 1)
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
    if name == "2d_hairpin":
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
