#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from utils import recon_error_l2, to_tensor


def plan_projected_path(
    x0: np.ndarray,
    x1: np.ndarray,
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    n_steps: int = 64,
    n_iters: int = 400,
    lr: float = 0.05,
    smooth_weight: float = 1.0,
    hard_project: bool = False,
    manifold_weight: float = 100.0,
    eps: float = 0.0,
) -> np.ndarray:
    """Optimize a path in x-space with optional hard projection constraint."""
    x0_t = to_tensor(x0[None, :], device)
    x1_t = to_tensor(x1[None, :], device)

    t = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=device)[:, None]
    path = (1.0 - t) * x0_t + t * x1_t
    path = path.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([path], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)

        # enforce endpoints softly by overwriting after update
        x = path

        diffs = x[1:] - x[:-1]
        smooth_loss = (diffs ** 2).sum()

        x_proj = project_fn(x)
        err = recon_error_l2(x, x_proj)
        penalty = torch.clamp(err - eps, min=0.0)
        manifold_loss = (penalty ** 2).mean()

        if hard_project:
            loss = smooth_weight * smooth_loss
        else:
            loss = smooth_weight * smooth_loss + manifold_weight * manifold_loss
        loss.backward()
        opt.step()

        with torch.no_grad():
            if hard_project:
                path.copy_(project_fn(path))
            path[0].copy_(x0_t[0])
            path[-1].copy_(x1_t[0])

    return path.detach().cpu().numpy()


def plan_latent_linear_path(
    x0: np.ndarray,
    x1: np.ndarray,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    n_steps: int = 64,
) -> np.ndarray:
    """Linear interpolation in latent, then decode back to x-space."""
    with torch.no_grad():
        z0 = encode_fn(to_tensor(x0[None, :], device))
        z1 = encode_fn(to_tensor(x1[None, :], device))

    t = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=device)[:, None]
    z = (1.0 - t) * z0 + t * z1

    with torch.no_grad():
        x = decode_fn(z)
    return x.detach().cpu().numpy()


def trajectory_metrics(
    x_traj: np.ndarray,
    gt_distance_fn: Callable[[np.ndarray], np.ndarray],
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    threshold: float,
    device: torch.device,
) -> Dict[str, float]:
    gt_dist = gt_distance_fn(x_traj)
    mean_gt_dist = float(np.mean(gt_dist))

    with torch.no_grad():
        xt = to_tensor(x_traj, device)
        x_proj = project_fn(xt)
        err = recon_error_l2(xt, x_proj).cpu().numpy()

    on_mask = err <= threshold
    on_count = int(np.sum(on_mask))
    off_count = int(len(on_mask) - on_count)
    on_ratio = float(on_count / max(1, len(on_mask)))

    return {
        "mean_gt_dist": mean_gt_dist,
        "on_count": on_count,
        "off_count": off_count,
        "on_ratio": on_ratio,
    }


def build_planner_cases(
    x_pairs: List[Tuple[np.ndarray, np.ndarray]],
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    gt_distance_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    device: torch.device,
    n_steps: int,
) -> Dict[str, List[Dict[str, object]]]:
    cases = {"projected": [], "latent": []}

    for x0, x1 in x_pairs:
        traj_proj = plan_projected_path(
            x0=x0,
            x1=x1,
            project_fn=project_fn,
            device=device,
            n_steps=n_steps,
        )
        m_proj = trajectory_metrics(traj_proj, gt_distance_fn, project_fn, threshold, device)
        cases["projected"].append({"x0": x0, "x1": x1, "traj": traj_proj, "metrics": m_proj})

        traj_lat = plan_latent_linear_path(
            x0=x0,
            x1=x1,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            device=device,
            n_steps=n_steps,
        )
        m_lat = trajectory_metrics(traj_lat, gt_distance_fn, project_fn, threshold, device)
        cases["latent"].append({"x0": x0, "x1": x1, "traj": traj_lat, "metrics": m_lat})

    return cases
