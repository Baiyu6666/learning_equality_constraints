from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from core.kinematics import (
    planar_fk as _planar_fk,
    spatial_fk as _spatial_fk,
    wrap_np_pi as _wrap_np_pi,
)
from core.projection import (
    project_points_tensor,
)


def resolve_periodic_mode(
    *,
    periodic_joint: bool | None = None,
    dataset_name: str | None = None,
) -> bool:
    if periodic_joint is not None:
        return bool(periodic_joint)
    if dataset_name is None:
        return False
    return "arm_" in str(dataset_name)


def _wrap_torch_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + np.pi, 2.0 * np.pi) - np.pi


def _angle_delta_torch(a: torch.Tensor, b: torch.Tensor, periodic: bool) -> torch.Tensor:
    d = b - a
    return _wrap_torch_pi(d) if periodic else d


def _angle_delta_np(a: np.ndarray, b: np.ndarray, periodic: bool) -> np.ndarray:
    d = b - a
    if not periodic:
        return d.astype(np.float32)
    return ((d + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _interp_linear_np(a: np.ndarray, b: np.ndarray, s: np.ndarray, periodic: bool) -> np.ndarray:
    d = _angle_delta_np(a, b, periodic=periodic)
    x = a + s * d
    if periodic:
        x = ((x + np.pi) % (2.0 * np.pi) - np.pi)
    return x.astype(np.float32)


def build_linear_path(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    n_waypoints: int,
    periodic: bool,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32).reshape(-1, 1)
    path = _interp_linear_np(
        x_start.reshape(1, -1).astype(np.float32),
        x_goal.reshape(1, -1).astype(np.float32),
        t,
        periodic=periodic,
    )
    path[0] = x_start.astype(np.float32)
    path[-1] = x_goal.astype(np.float32)
    return path


def pick_far_pair_workspace_planar(
    x: np.ndarray,
    lengths: list[float],
    min_ratio: float,
    max_ratio: float,
    tries: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    reach = float(sum(lengths))
    lo = float(min_ratio) * max(reach, 1e-6)
    hi = float(max_ratio) * max(reach, 1e-6)
    if hi < lo:
        hi = lo
    target = 0.5 * (lo + hi)
    best = None
    best_delta = 1e18

    q = x.astype(np.float32)
    ee = _planar_fk(q, lengths)[:, -1, :]
    n = len(q)
    for _ in range(max(1, int(tries))):
        i = int(np.random.randint(0, n))
        j = int(np.random.randint(0, n))
        if i == j:
            continue
        d = float(np.linalg.norm(ee[i] - ee[j]))
        if lo <= d <= hi:
            return q[i], q[j], d, reach
        delta = abs(d - target)
        if delta < best_delta:
            best_delta = delta
            best = (q[i], q[j], d)
    assert best is not None
    return best[0], best[1], float(best[2]), reach


def _smoothstep(s: np.ndarray) -> np.ndarray:
    return (s * s * (3.0 - 2.0 * s)).astype(np.float32)


def _angle_shortest_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _wrap_np_pi(b - a)


def _angle_interp_shortest(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    d = _angle_shortest_delta(a, b)
    return _wrap_np_pi(a + s * d)


def init_path_joint_spline(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    n_waypoints: int,
    mid_noise: float,
) -> np.ndarray:
    q_start = q_start.astype(np.float32)
    q_goal = q_goal.astype(np.float32)
    d = _angle_shortest_delta(q_start, q_goal)
    q_mid = _wrap_np_pi(
        q_start
        + 0.5 * d
        + float(mid_noise) * np.random.randn(*q_start.shape).astype(np.float32)
    )

    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32)
    q = np.zeros((int(n_waypoints), q_start.shape[0]), dtype=np.float32)
    left = t <= 0.5
    right = ~left
    if np.any(left):
        s = _smoothstep((t[left] / 0.5).reshape(-1, 1))
        q[left] = _angle_interp_shortest(q_start.reshape(1, -1), q_mid.reshape(1, -1), s)
    if np.any(right):
        s = _smoothstep(((t[right] - 0.5) / 0.5).reshape(-1, 1))
        q[right] = _angle_interp_shortest(q_mid.reshape(1, -1), q_goal.reshape(1, -1), s)
    q[0] = q_start
    q[-1] = q_goal
    return _wrap_np_pi(q)


def _fk_ee_torch(q: torch.Tensor, lengths: list[float], is_spatial: bool) -> torch.Tensor:
    if not is_spatial:
        ang = torch.cumsum(q, dim=1)
        x = torch.zeros((q.shape[0],), dtype=q.dtype, device=q.device)
        y = torch.zeros((q.shape[0],), dtype=q.dtype, device=q.device)
        for k in range(q.shape[1]):
            lk = float(lengths[k])
            x = x + lk * torch.cos(ang[:, k])
            y = y + lk * torch.sin(ang[:, k])
        return torch.stack([x, y], dim=1)

    if q.shape[1] == 3:
        l1, l2 = float(lengths[0]), float(lengths[1])
        q1 = q[:, 0]
        t2 = q[:, 1]
        t3 = q[:, 1] + q[:, 2]
        r1 = l1 * torch.cos(t2)
        z1 = l1 * torch.sin(t2)
        r2 = r1 + l2 * torch.cos(t3)
        z2 = z1 + l2 * torch.sin(t3)
        x = r2 * torch.cos(q1)
        y = r2 * torch.sin(q1)
        return torch.stack([x, y, z2], dim=1)

    if q.shape[1] == 4:
        l1, l2, l3 = float(lengths[0]), float(lengths[1]), float(lengths[2])
        q1 = q[:, 0]
        t2 = q[:, 1]
        t3 = q[:, 1] + q[:, 2]
        t4 = q[:, 1] + q[:, 2] + q[:, 3]
        r1 = l1 * torch.cos(t2)
        z1 = l1 * torch.sin(t2)
        r2 = r1 + l2 * torch.cos(t3)
        z2 = z1 + l2 * torch.sin(t3)
        r3 = r2 + l3 * torch.cos(t4)
        z3 = z2 + l3 * torch.sin(t4)
        x = r3 * torch.cos(q1)
        y = r3 * torch.sin(q1)
        return torch.stack([x, y, z3], dim=1)

    if q.shape[1] == 6:
        a1 = q[:, 0]
        a2 = q[:, 1]
        a3 = q[:, 1] + q[:, 2]
        a4 = q[:, 1] + q[:, 2] + q[:, 3]
        r1 = float(lengths[0]) * torch.cos(a2)
        z1 = float(lengths[0]) * torch.sin(a2)
        r2 = r1 + float(lengths[1]) * torch.cos(a3)
        z2 = z1 + float(lengths[1]) * torch.sin(a3)
        r3 = r2 + float(lengths[2]) * torch.cos(a4)
        z3 = z2 + float(lengths[2]) * torch.sin(a4)
        r4 = r3 + float(lengths[3]) * torch.cos(a4)
        z4 = z3 + float(lengths[3]) * torch.sin(a4)
        r5 = r4 + float(lengths[4]) * torch.cos(a4 + q[:, 4])
        z5 = z4 + float(lengths[4]) * torch.sin(a4 + q[:, 4])
        r6 = r5 + float(lengths[5]) * torch.cos(a4 + q[:, 4])
        z6 = z5 + float(lengths[5]) * torch.sin(a4 + q[:, 4])
        x = r6 * torch.cos(a1)
        y = r6 * torch.sin(a1)
        return torch.stack([x, y, z6], dim=1)

    raise ValueError(f"unsupported spatial dof={q.shape[1]}")


def init_path_via_workspace_ik(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    lengths: list[float],
    is_spatial: bool,
    use_pybullet_n6: bool,
    n_waypoints: int,
    device: str,
) -> np.ndarray:
    if is_spatial:
        ee_s = _spatial_fk(q_start.reshape(1, -1), lengths, use_pybullet_n6=use_pybullet_n6)[0, -1, :]
        ee_g = _spatial_fk(q_goal.reshape(1, -1), lengths, use_pybullet_n6=use_pybullet_n6)[0, -1, :]
    else:
        ee_s = _planar_fk(q_start.reshape(1, -1), lengths)[0, -1, :]
        ee_g = _planar_fk(q_goal.reshape(1, -1), lengths)[0, -1, :]

    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32).reshape(-1, 1)
    ee_targets = (1.0 - t) * ee_s.reshape(1, -1) + t * ee_g.reshape(1, -1)

    q_path = np.zeros((int(n_waypoints), q_start.shape[0]), dtype=np.float32)
    q_path[0] = q_start.astype(np.float32)
    q_prev = q_start.astype(np.float32).copy()
    for i in range(1, int(n_waypoints) - 1):
        target = torch.tensor(ee_targets[i : i + 1].astype(np.float32), device=device)
        qv = torch.tensor(q_prev.reshape(1, -1), device=device, requires_grad=True)
        opt = torch.optim.Adam([qv], lr=0.03)
        for _ in range(70):
            opt.zero_grad(set_to_none=True)
            ee = _fk_ee_torch(qv, lengths, is_spatial=is_spatial)
            reg = (qv - torch.tensor(q_prev.reshape(1, -1), device=device)).pow(2).mean()
            loss = (ee - target).pow(2).mean() + 0.01 * reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                qv[:] = torch.remainder(qv + np.pi, 2.0 * np.pi) - np.pi
        q_prev = qv.detach().cpu().numpy().reshape(-1).astype(np.float32)
        q_path[i] = q_prev
    q_path[-1] = q_goal.astype(np.float32)
    return _wrap_np_pi(q_path)


def plan_linear_then_project(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    device: str,
    n_waypoints: int,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int = 0,
    f_abs_stop: float | None = None,
    keep_endpoints: bool = False,
    periodic: bool = False,
) -> np.ndarray:
    path = build_linear_path(
        x_start.astype(np.float32),
        x_goal.astype(np.float32),
        n_waypoints=int(n_waypoints),
        periodic=bool(periodic),
    )
    path_t = torch.from_numpy(path).to(device)
    proj_t, _ = project_points_tensor(
        model,
        path_t,
        proj_steps=int(proj_steps),
        proj_alpha=float(proj_alpha),
        proj_min_steps=int(proj_min_steps),
        f_abs_stop=f_abs_stop,
    )
    out = proj_t.detach().cpu().numpy().astype(np.float32)
    if periodic:
        out = ((out + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    if keep_endpoints:
        out[0] = x_start.astype(np.float32)
        out[-1] = x_goal.astype(np.float32)
    return out


def plan_path_optimized(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    device: str,
    n_waypoints: int,
    opt_steps: int,
    opt_lr: float,
    lam_manifold: float,
    lam_len: float,
    lam_smooth: float,
    trust_scale: float,
    periodic: bool,
    init_path: np.ndarray | None = None,
) -> np.ndarray:
    if init_path is None:
        path0 = build_linear_path(
            x_start.astype(np.float32),
            x_goal.astype(np.float32),
            n_waypoints=int(n_waypoints),
            periodic=bool(periodic),
        )
    else:
        path0 = init_path.astype(np.float32)

    q = torch.tensor(path0, device=device, requires_grad=True)
    q0 = torch.tensor(x_start.astype(np.float32), device=device)
    qT = torch.tensor(x_goal.astype(np.float32), device=device)

    with torch.no_grad():
        v0 = _angle_delta_torch(
            torch.tensor(path0[:-1], device=device, dtype=torch.float32),
            torch.tensor(path0[1:], device=device, dtype=torch.float32),
            periodic=bool(periodic),
        )
        mean_step0 = float(torch.mean(torch.linalg.norm(v0, dim=1)).item()) if v0.shape[0] > 0 else 1.0
    trust_delta = float(max(1e-6, float(trust_scale) * max(mean_step0, 1e-6)))

    opt = torch.optim.Adam([q], lr=float(opt_lr))
    for _ in range(int(opt_steps)):
        q_prev = q.detach().clone()
        opt.zero_grad(set_to_none=True)
        f = model(q)
        if f.dim() == 1:
            f = f.unsqueeze(1)
        loss_man = (f ** 2).mean()
        v = _angle_delta_torch(q[:-1], q[1:], periodic=bool(periodic))
        loss_len = (v ** 2).mean()
        if q.shape[0] >= 3:
            dv = _angle_delta_torch(v[:-1], v[1:], periodic=bool(periodic))
            loss_smooth = (dv ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=q.device)
        loss = (
            float(lam_manifold) * loss_man
            + float(lam_smooth) * loss_smooth
            + float(lam_len) * loss_len
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            dq = _angle_delta_torch(q_prev, q, periodic=bool(periodic))
            dn = torch.linalg.norm(dq, dim=1, keepdim=True).clamp_min(1e-9)
            scale = torch.minimum(torch.ones_like(dn), torch.full_like(dn, trust_delta) / dn)
            q[:] = q_prev + dq * scale
            if periodic:
                q[:] = _wrap_torch_pi(q)
            q[0] = q0
            q[-1] = qT
    out = q.detach().cpu().numpy().astype(np.float32)
    if periodic:
        out = ((out + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    return out


def _cfg_val(cfg: Any, names: list[str], default: float) -> float:
    for n in names:
        if hasattr(cfg, n):
            v = getattr(cfg, n)
            if v is not None:
                return float(v)
    return float(default)


def plan_path(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    cfg: Any,
    *,
    planner_name: str,
    n_waypoints: int,
    dataset_name: str | None = None,
    periodic_joint: bool | None = None,
    init_path: np.ndarray | None = None,
    f_abs_stop: float | None = None,
    keep_endpoints: bool = False,
) -> np.ndarray:
    periodic = resolve_periodic_mode(periodic_joint=periodic_joint, dataset_name=dataset_name)
    p = str(planner_name).strip().lower()
    if p in ("linear_project", "linear_proj", "projection"):
        return plan_linear_then_project(
            model,
            x_start,
            x_goal,
            device=str(cfg.device),
            n_waypoints=int(n_waypoints),
            proj_steps=int(_cfg_val(cfg, ["proj_steps"], 100)),
            proj_alpha=float(_cfg_val(cfg, ["proj_alpha"], 0.3)),
            proj_min_steps=int(_cfg_val(cfg, ["proj_min_steps"], 0)),
            f_abs_stop=f_abs_stop,
            keep_endpoints=bool(keep_endpoints),
            periodic=bool(periodic),
        )
    if p in ("trajectory_opt", "opt", "continuous_opt"):
        return plan_path_optimized(
            model,
            x_start,
            x_goal,
            device=str(cfg.device),
            n_waypoints=int(n_waypoints),
            opt_steps=int(_cfg_val(cfg, ["plan_opt_steps", "plan_iters"], 1240)),
            opt_lr=float(_cfg_val(cfg, ["plan_opt_lr", "plan_lr"], 0.01)),
            lam_manifold=float(_cfg_val(cfg, ["plan_lam_manifold", "plan_manifold_weight"], 1.0)),
            lam_len=float(_cfg_val(cfg, ["plan_lam_len_joint"], 0.40)),
            lam_smooth=float(_cfg_val(cfg, ["plan_opt_lam_smooth", "plan_smooth_weight"], 0.2)),
            trust_scale=float(_cfg_val(cfg, ["plan_trust_scale"], 0.8)),
            periodic=bool(periodic),
            init_path=init_path,
        )
    raise ValueError(f"unknown planner_name '{planner_name}'")
