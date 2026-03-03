from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _f_and_grad(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f = model(x)
    if f.dim() == 1:
        f = f.unsqueeze(1)
    v = 0.5 * (f ** 2).sum(dim=1, keepdim=True)
    grad = torch.autograd.grad(v.sum(), x)[0]
    return f, grad


def _converged_mask(f: torch.Tensor, f_abs_stop: float | None, min_ready: bool) -> torch.Tensor:
    if f_abs_stop is None:
        return torch.zeros((f.shape[0],), dtype=torch.bool, device=f.device)
    raw = torch.all(torch.abs(f) < float(f_abs_stop), dim=1)
    if not min_ready:
        return torch.zeros_like(raw)
    return raw


def project_points_tensor(
    model: nn.Module,
    x0: torch.Tensor,
    *,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int = 0,
    f_abs_stop: float | None = None,
) -> tuple[torch.Tensor, int]:
    x = x0.clone()
    active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    min_steps = max(0, min(int(proj_min_steps), int(proj_steps)))
    for k in range(int(proj_steps)):
        x.requires_grad_(True)
        with torch.enable_grad():
            f, grad = _f_and_grad(model, x)
        converged = _converged_mask(f, f_abs_stop, min_ready=(k + 1 >= min_steps))
        next_active = active & (~converged)
        if k + 1 >= min_steps and (not next_active.any()):
            return x.detach(), int(k)
        x_next = x - float(proj_alpha) * grad
        x = torch.where(next_active.unsqueeze(1), x_next, x).detach()
        active = next_active
    return x.detach(), int(proj_steps)


def project_points_with_steps_tensor(
    model: nn.Module,
    x0: torch.Tensor,
    *,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int = 0,
    f_abs_stop: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x0.clone()
    n = x.shape[0]
    steps = torch.zeros(n, dtype=torch.long, device=x.device)
    active = torch.ones(n, dtype=torch.bool, device=x.device)
    min_steps = max(0, min(int(proj_min_steps), int(proj_steps)))
    for k in range(int(proj_steps)):
        x.requires_grad_(True)
        with torch.enable_grad():
            f, grad = _f_and_grad(model, x)
        converged = _converged_mask(f, f_abs_stop, min_ready=(k + 1 >= min_steps))
        newly = active & converged
        steps[newly] = int(k)
        next_active = active & (~converged)
        if not next_active.any():
            active = next_active
            break
        x_next = x - float(proj_alpha) * grad
        x = torch.where(next_active.unsqueeze(1), x_next, x).detach()
        active = next_active
    steps[active] = int(proj_steps)
    return x.detach(), steps.detach()


def project_trajectory_tensor(
    model: nn.Module,
    x0: torch.Tensor,
    *,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int = 0,
    f_abs_stop: float | None = None,
) -> tuple[torch.Tensor, int]:
    x = x0.clone()
    traj = [x.detach()]
    active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    min_steps = max(0, min(int(proj_min_steps), int(proj_steps)))
    for k in range(int(proj_steps)):
        x.requires_grad_(True)
        with torch.enable_grad():
            f, grad = _f_and_grad(model, x)
        converged = _converged_mask(f, f_abs_stop, min_ready=(k + 1 >= min_steps))
        next_active = active & (~converged)
        if k + 1 >= min_steps and (not next_active.any()):
            return torch.stack(traj, dim=0), int(k)
        x_next = x - float(proj_alpha) * grad
        x = torch.where(next_active.unsqueeze(1), x_next, x).detach()
        active = next_active
        traj.append(x.detach())
    return torch.stack(traj, dim=0), int(proj_steps)


def project_points_with_steps_numpy(
    model: nn.Module,
    x0: np.ndarray,
    *,
    device: str,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int = 0,
    f_abs_stop: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x0_t = torch.from_numpy(x0.astype(np.float32)).to(device)
    x_end_t, steps_t = project_points_with_steps_tensor(
        model,
        x0_t,
        proj_steps=int(proj_steps),
        proj_alpha=float(proj_alpha),
        proj_min_steps=int(proj_min_steps),
        f_abs_stop=f_abs_stop,
    )
    return (
        x_end_t.detach().cpu().numpy().astype(np.float32),
        steps_t.detach().cpu().numpy().astype(np.float32),
    )


def project_trajectory_numpy(
    model: nn.Module,
    x0: np.ndarray,
    *,
    device: str,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int = 0,
    f_abs_stop: float | None = None,
) -> np.ndarray:
    x0_t = torch.from_numpy(x0.astype(np.float32)).to(device)
    traj_t, _ = project_trajectory_tensor(
        model,
        x0_t,
        proj_steps=int(proj_steps),
        proj_alpha=float(proj_alpha),
        proj_min_steps=int(proj_min_steps),
        f_abs_stop=f_abs_stop,
    )
    return traj_t.detach().cpu().numpy().astype(np.float32)
