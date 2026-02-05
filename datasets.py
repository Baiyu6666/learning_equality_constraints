#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass
class DatasetSpec:
    name: str
    dim: int
    latent_dim_default: int
    train_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_off_sampler: Callable[[int, np.random.Generator], np.ndarray]
    gt_distance_fn: Callable[[np.ndarray], np.ndarray]
    # bounding box for plotting / off sampling reference
    plot_bounds: Tuple[np.ndarray, np.ndarray]  # (lo, hi)


def sample_spiral_on(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    True 3D helix curve (1D manifold):
        x = cos(theta)
        y = sin(theta)
        z = a * theta
    """
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25  # controls pitch
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0  # shift to roughly center around z=0
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_spiral_off(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Points not near the helix curve.
    """
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0

    # move points radially away from helix
    dx = rng.uniform(0.4, 1.0, size=n)
    dy = rng.uniform(0.4, 1.0, size=n)
    dz = rng.uniform(0.4, 1.0, size=n)

    x += dx * rng.choice([-1, 1], size=n)
    y += dy * rng.choice([-1, 1], size=n)
    z += dz * rng.choice([-1, 1], size=n)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_sphere_on(
    n: int,
    rng: np.random.Generator,
    radius: float = 1.0,
    center=(0, 0, 0),
) -> np.ndarray:
    # uniform on sphere surface via normal distribution
    v = rng.normal(size=(n, 3))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    pts = radius * v + np.array(center, dtype=float)[None, :]
    return pts.astype(np.float32)


def sample_sphere_off(
    n: int,
    rng: np.random.Generator,
    radius: float = 1.0,
    center=(0, 0, 0),
    max_iters: int = 50,
) -> np.ndarray:
    # sample in a cube and reject near-surface band
    lo = np.array(center) - 1.8 * radius
    hi = np.array(center) + 1.8 * radius
    out = []
    for _ in range(max_iters):
        if len(out) >= n:
            break
        m = int((n - len(out)) * 1.5) + 16
        pts = rng.uniform(lo, hi, size=(m, 3))
        r = np.linalg.norm(pts - np.array(center)[None, :], axis=1)
        # keep points away from surface band [0.85R, 1.15R]
        mask = (r < 0.85 * radius) | (r > 1.15 * radius)
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    if len(out) < n:
        # fallback: fill remaining with uniform samples (may include near-surface points)
        missing = n - len(out)
        pts = rng.uniform(lo, hi, size=(missing, 3))
        out.append(pts)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


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


def sample_paraboloid_off(
    n: int,
    rng: np.random.Generator,
    xy_range: float = 1.2,
    z_max: float = 3.0,
) -> np.ndarray:
    # sample in a box; force z to differ from x^2+y^2 by a margin
    x = rng.uniform(-xy_range, xy_range, size=n)
    y = rng.uniform(-xy_range, xy_range, size=n)
    z = rng.uniform(0.0, z_max, size=n)
    z_surface = x ** 2 + y ** 2
    # push z away from surface
    sign = rng.choice([-1.0, 1.0], size=n)
    delta = rng.uniform(0.35, 1.0, size=n)
    z2 = np.clip(z_surface + sign * delta, 0.0, z_max)
    return np.stack([x, y, z2], axis=1).astype(np.float32)


def sample_two_sphere_outer_on(n: int, rng: np.random.Generator) -> np.ndarray:
    # Two spheres; keep union outer boundary:
    # sphere A: center (-0.8, 0, 0), R=1
    # sphere B: center (+0.8, 0, 0), R=1
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0

    # sample candidates from both surfaces, then keep those not inside the other sphere
    m = int(n * 2.2) + 64
    pts_a = sample_sphere_on(m, rng, radius=r, center=ca)
    pts_b = sample_sphere_on(m, rng, radius=r, center=cb)
    pts = np.concatenate([pts_a, pts_b], axis=0)

    da = np.linalg.norm(pts - ca[None, :], axis=1)
    db = np.linalg.norm(pts - cb[None, :], axis=1)
    # on A surface means da≈R; keep those with db>=R (not inside B)
    # on B surface means db≈R; keep those with da>=R (not inside A)
    keep = (db >= r - 1e-6) | (da >= r - 1e-6)
    pts = pts[keep]

    if pts.shape[0] < n:
        # fallback: just return what we have (rare)
        return pts.astype(np.float32)
    idx = rng.choice(pts.shape[0], size=n, replace=False)
    return pts[idx].astype(np.float32)


def sample_two_sphere_outer_off(n: int, rng: np.random.Generator, max_iters: int = 50) -> np.ndarray:
    # sample around both spheres but avoid the union boundary band
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
        # distance to union boundary roughly min(|da-R|, |db-R|)
        dist_to_boundary = np.minimum(np.abs(da - r), np.abs(db - r))
        # keep those sufficiently far from boundary
        mask = dist_to_boundary > 0.25
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    if len(out) < n:
        missing = n - len(out)
        pts = rng.uniform(lo, hi, size=(missing, 3))
        out.append(pts)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


def sample_circle_on(n: int, rng: np.random.Generator, r: float = 1.0) -> np.ndarray:
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_circle_off(n: int, rng: np.random.Generator, r: float = 1.0, max_iters: int = 50) -> np.ndarray:
    # sample in box and keep away from radius band
    out = []
    for _ in range(max_iters):
        if len(out) >= n:
            break
        m = int((n - len(out)) * 1.6) + 16
        pts = rng.uniform(-1.8, 1.8, size=(m, 2))
        rad = np.linalg.norm(pts, axis=1)
        mask = (rad < 0.75 * r) | (rad > 1.25 * r)
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    if len(out) < n:
        missing = n - len(out)
        pts = rng.uniform(-1.8, 1.8, size=(missing, 2))
        out.append(pts)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


def sample_square_on(n: int, rng: np.random.Generator, half: float = 1.0) -> np.ndarray:
    # boundary of axis-aligned square: x=±half or y=±half
    # choose edges uniformly
    edge = rng.integers(0, 4, size=n)
    u = rng.uniform(-half, half, size=n)
    x = np.empty(n)
    y = np.empty(n)
    # 0: top y=half, x=u
    mask = edge == 0
    x[mask] = u[mask]
    y[mask] = half
    # 1: bottom y=-half
    mask = edge == 1
    x[mask] = u[mask]
    y[mask] = -half
    # 2: right x=half
    mask = edge == 2
    x[mask] = half
    y[mask] = u[mask]
    # 3: left x=-half
    mask = edge == 3
    x[mask] = -half
    y[mask] = u[mask]
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_square_off(n: int, rng: np.random.Generator, half: float = 1.0, max_iters: int = 50) -> np.ndarray:
    # sample in a larger box and avoid boundary band
    out = []
    for _ in range(max_iters):
        if len(out) >= n:
            break
        m = int((n - len(out)) * 1.6) + 16
        pts = rng.uniform(-1.8 * half, 1.8 * half, size=(m, 2))
        # distance to square boundary in L_inf: abs(max(|x|,|y|)-half)
        linf = np.maximum(np.abs(pts[:, 0]), np.abs(pts[:, 1]))
        dist_to_boundary = np.abs(linf - half)
        mask = dist_to_boundary > 0.18 * half
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    if len(out) < n:
        missing = n - len(out)
        pts = rng.uniform(-1.8 * half, 1.8 * half, size=(missing, 2))
        out.append(pts)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


def gt_dist_spiral(points: np.ndarray, theta_samples: int = 2048) -> np.ndarray:
    # Approximate distance to 3D helix by dense sampling along theta.
    theta = np.linspace(0.0, 4.0 * np.pi, num=theta_samples, dtype=np.float32)
    a = 0.25
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0
    curve = np.stack([x, y, z], axis=1).astype(np.float32)
    diff = points[:, None, :] - curve[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def gt_dist_sphere(points: np.ndarray, radius: float = 1.0, center=(0, 0, 0)) -> np.ndarray:
    r = np.linalg.norm(points - np.array(center, dtype=float)[None, :], axis=1)
    return np.abs(r - radius)


def gt_dist_paraboloid(points: np.ndarray) -> np.ndarray:
    # Vertical distance to surface z = x^2 + y^2 (approximation of true distance).
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


def gt_dist_circle(points: np.ndarray, r: float = 1.0) -> np.ndarray:
    rad = np.linalg.norm(points, axis=1)
    return np.abs(rad - r)


def gt_dist_square(points: np.ndarray, half: float = 1.0) -> np.ndarray:
    linf = np.maximum(np.abs(points[:, 0]), np.abs(points[:, 1]))
    return np.abs(linf - half)


def build_datasets() -> Dict[str, DatasetSpec]:
    ds = {}

    ds["spiral3d"] = DatasetSpec(
        name="spiral3d",
        dim=3,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_spiral_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_spiral(x),
        plot_bounds=(np.array([-2.2, -2.2, -2.8]), np.array([2.2, 2.2, 2.8])),
    )

    ds["sphere3d"] = DatasetSpec(
        name="sphere3d",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_sphere_on(n, rng, radius=1.0, center=(0, 0, 0)),
        eval_on_sampler=lambda n, rng: sample_sphere_on(n, rng, radius=1.0, center=(0, 0, 0)),
        eval_off_sampler=lambda n, rng: sample_sphere_off(n, rng, radius=1.0, center=(0, 0, 0)),
        gt_distance_fn=lambda x: gt_dist_sphere(x, radius=1.0, center=(0, 0, 0)),
        plot_bounds=(np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0])),
    )

    ds["paraboloid3d"] = DatasetSpec(
        name="paraboloid3d",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_off_sampler=lambda n, rng: sample_paraboloid_off(n, rng, xy_range=1.2, z_max=3.0),
        gt_distance_fn=lambda x: gt_dist_paraboloid(x),
        plot_bounds=(np.array([-1.8, -1.8, 0.0]), np.array([1.8, 1.8, 3.2])),
    )

    ds["twosphere3d"] = DatasetSpec(
        name="twosphere3d",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_two_sphere_outer_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_two_sphere_outer(x),
        plot_bounds=(np.array([-2.8, -2.2, -2.2]), np.array([2.8, 2.2, 2.2])),
    )

    ds["circle2d"] = DatasetSpec(
        name="circle2d",
        dim=2,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_circle_on(n, rng, r=1.0),
        eval_on_sampler=lambda n, rng: sample_circle_on(n, rng, r=1.0),
        eval_off_sampler=lambda n, rng: sample_circle_off(n, rng, r=1.0),
        gt_distance_fn=lambda x: gt_dist_circle(x, r=1.0),
        plot_bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    )

    ds["square2d"] = DatasetSpec(
        name="square2d",
        dim=2,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_off_sampler=lambda n, rng: sample_square_off(n, rng, half=1.0),
        gt_distance_fn=lambda x: gt_dist_square(x, half=1.0),
        plot_bounds=(np.array([-2.3, -2.3]), np.array([2.3, 2.3])),
    )

    return ds
