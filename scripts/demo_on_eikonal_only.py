#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import nn

from levelset_energy_algorithm import eval_bounds_from_train, generate_dataset, set_seed
try:
    from ur5_pybullet_utils import UR5PyBulletKinematics
except Exception:
    from scripts.ur5_pybullet_utils import UR5PyBulletKinematics
try:
    from ur5_pybullet_utils import _make_pybullet_friendly_urdf
except Exception:
    from scripts.ur5_pybullet_utils import _make_pybullet_friendly_urdf
try:
    from ur5_pybullet_utils import pick_default_ee_link_index
except Exception:
    from scripts.ur5_pybullet_utils import pick_default_ee_link_index
try:
    from ur5_n6_dataset import sample_ur5_upward_dataset
except Exception:
    from scripts.ur5_n6_dataset import sample_ur5_upward_dataset
try:
    from ur5_n6_dataset import sample_ur5_upward_dataset_analytic
except Exception:
    from scripts.ur5_n6_dataset import sample_ur5_upward_dataset_analytic

BASE_2D_DATASETS = [
    "figure_eight",
    "high_curvature",
    "ellipse",
    "noise_only",
    "sine",
    "sparse_only",
    "discontinuous",
    "looped_spiro",
    "sharp_star",
    "hetero_noise",
    "double_valley",
    "hairpin",
    "planar_arm_line_n2",
]

BASE_3D_DATASETS = [
    "saddle_surface",
    "sphere_surface",
    "torus_surface",
    "planar_arm_line_n3",
    "spatial_arm_plane_n3",
    "spatial_arm_circle_n3",
]

BASE_4D_DATASETS = [
    "spatial_arm_plane_n4",
]

BASE_6D_DATASETS = [
    "spatial_arm_up_n6",
    "spatial_arm_up_n6_py",
]

DEFAULT_DATASETS = [
    # "planar_arm_line_n3",

    # "spatial_arm_plane_n3",
    # "spatial_arm_up_n6_py",
    # "spatial_arm_up_n6",

]
DEFAULT_OUTDIR = "outputs_levelset_datasets/on_eikonal_only"
UR5_LINK_LENGTHS = [0.425, 0.39225, 0.10915, 0.09465, 0.0823, 0.10]


@dataclass
class DemoCfg:
    seed: int = 2116
    device: str = "auto"
    n_train: int = 512
    n_grid: int = 4096
    hidden: int = 128
    depth: int = 3
    lr: float = 3e-4
    epochs: int = 2000
    batch_size: int = 128
    lam_eikonal: float = 0.25
    lam_eikonal_ortho: float = 3.0
    eikonal_near_ratio: float = 0.85
    eikonal_near_std_ratio: float = 0.05
    metric_eval_every: int = 1
    eval_pad_ratio: float = 0.6
    proj_alpha: float = 0.3
    proj_steps: int = 120
    n_traj: int = 64
    zero_eps_quantile: float = 90.0
    # Used for <base>_3d lift.
    z_amp1: float = 0.35
    z_amp2: float = 0.20
    z_freq1: float = 1.5
    z_freq2: float = 1.2
    # For 3D datasets, codim can be >= 1.
    constraint_dim: int = 2
    show_3d_plot: bool = not False
    surface_plot_n: int = 28
    surface_eval_chunk: int = 8192
    surface_max_points: int = 5000
    plot_train_max_points: int = 1200
    plot_traj_max_count: int = 24
    plot_traj_stride: int = 2
    surface_use_marching_cubes: bool = True
    plan_pair_min_ratio: float = 0.15
    plan_pair_max_ratio: float = 0.35
    plan_pair_tries: int = 1200
    plan_init_mode: str = "joint_spline"  # "joint_spline" or "workspace_ik"
    plan_joint_mid_noise: float = 0.0
    plan_lam_manifold: float = 1.0
    plan_lam_len_joint: float = 0.40
    # Planner stabilization against sudden jumps/branch switching.
    plan_opt_steps: int = 1240
    plan_opt_lr: float = 0.01
    plan_opt_lam_smooth: float = 0.2
    plan_trust_scale: float = 0.8  # trust radius = scale * mean step of init path
    plan_anim_fps: int = 6
    plan_anim_stride: int = 1
    plan_save_gif: bool = not False
    plan_pybullet_render: bool = True
    plan_pybullet_real_time_dt: float = 0.06
    n6_workspace_vis_points: int = 90
    train_log_every: int = 50
    eval_chamfer_n_gt: int = 1024
    eval_chamfer_n_seed: int = 2048
    eval_chamfer_near_ratio: float = 0.8
    eval_chamfer_near_noise_std_ratio: float = 0.08
    # UR5 runtime config (avoid env vars).
    ur5_urdf_path: str = "/home/baiyu/PycharmProjects/icrl-master/custom_envs/custom_envs/envs/xmls/UR5+gripper/ur5_gripper.urdf"
    ur5_ee_link_index: int = -1  # <0 means auto-pick
    ur5_tool_axis: str = "x"
    ur5_grasp_offset: float = 0.11
    ur5_grasp_axis_shift: float = 0.0
    ur5_debug_axes: bool = False
    ur5_cylinder_rotate_90: bool = True
    ur5_gripper_close_ratio: float = 0.78


# ----------------------------------------------------------------------
# Config layering:
# 1) BASE_CFG = current default values (from DemoCfg)
# 2) PROFILE_OVERRIDES = run mode (debug/fast/full)
# 3) DATASET_OVERRIDES = per-dataset custom parameters
# ----------------------------------------------------------------------
ACTIVE_PROFILE = "default"
BASE_CFG: dict[str, Any] = asdict(DemoCfg())
PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "default": {},
    "debug": {
        "n_train": 16,
        "epochs": 50,
        "n_traj": 32,
        "train_log_every": 20,
    },
    "fast": {
        "n_train": 256,
        "epochs": 800,
    },
    "full": {},
}

DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    # Examples (edit as needed):
    "spatial_arm_up_n6_py": {
        "n_train": 2048,
        "epochs": 4000,
        "constraint_dim": 2,
        "lr": 2e-4
    },
    "spatial_arm_up_n6": {
        "n_train": 512,
        "epochs": 3000,
        "constraint_dim": 2,
        "lr": 3e-4

    },
}


def build_cfg(dataset_name: str, profile: str = ACTIVE_PROFILE) -> DemoCfg:
    cfg_dict = dict(BASE_CFG)
    prof = PROFILE_OVERRIDES.get(str(profile), {})
    if not prof and str(profile) != "default":
        print(f"[warn] unknown profile '{profile}', fallback to base defaults")
    cfg_dict.update(prof)
    ds_ov = DATASET_OVERRIDES.get(str(dataset_name), {})
    cfg_dict.update(ds_ov)
    return DemoCfg(**cfg_dict)


_UR5_BACKEND: UR5PyBulletKinematics | None = None
_UR5_BACKEND_READY: bool = False
_UR5_URDF_PATH: str = ""
_UR5_EE_LINK_INDEX: int | None = None
_UR5_TOOL_AXIS: str = "x"


def _set_ur5_runtime_from_cfg(cfg: DemoCfg) -> None:
    global _UR5_BACKEND, _UR5_BACKEND_READY, _UR5_URDF_PATH, _UR5_EE_LINK_INDEX, _UR5_TOOL_AXIS
    _UR5_BACKEND = None
    _UR5_BACKEND_READY = False
    _UR5_URDF_PATH = str(cfg.ur5_urdf_path).strip()
    _UR5_EE_LINK_INDEX = None if int(cfg.ur5_ee_link_index) < 0 else int(cfg.ur5_ee_link_index)
    _UR5_TOOL_AXIS = str(cfg.ur5_tool_axis).strip().lower()


def _get_ur5_backend() -> UR5PyBulletKinematics | None:
    global _UR5_BACKEND, _UR5_BACKEND_READY
    if _UR5_BACKEND_READY:
        return _UR5_BACKEND
    _UR5_BACKEND_READY = True
    try:
        _UR5_BACKEND = UR5PyBulletKinematics.from_settings(
            urdf_path=_UR5_URDF_PATH,
            ee_link_index=_UR5_EE_LINK_INDEX,
            tool_axis=_UR5_TOOL_AXIS,
        )
    except Exception as e:
        print(f"[warn] UR5 pybullet backend unavailable: {e}")
        _UR5_BACKEND = None
    return _UR5_BACKEND


def _choose_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _enable_interactive_backend_if_possible() -> bool:
    for name in ("QtAgg", "TkAgg", "Qt5Agg"):
        try:
            plt.switch_backend(name)
            return True
        except Exception:
            continue
    return False


def _uniform_in_box(n: int, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.random.uniform(mins, maxs, size=(n, len(mins))).astype(np.float32)


def _lift_xy_to_3d_var(x2: np.ndarray, cfg: DemoCfg) -> np.ndarray:
    x = x2[:, 0:1].astype(np.float32)
    y = x2[:, 1:2].astype(np.float32)
    z = (
        float(cfg.z_amp1) * np.sin(float(cfg.z_freq1) * x)
        + float(cfg.z_amp2) * np.cos(float(cfg.z_freq2) * y)
    ).astype(np.float32)
    return np.concatenate([x2.astype(np.float32), z], axis=1)


def _lift_xy_to_3d_zero(x2: np.ndarray) -> np.ndarray:
    z = np.zeros((x2.shape[0], 1), dtype=np.float32)
    return np.concatenate([x2.astype(np.float32), z], axis=1)


def _resolve_dataset(name: str, cfg: DemoCfg) -> dict[str, Any]:
    if name in BASE_2D_DATASETS:
        x, _ = generate_dataset(name, cfg)
        labels = ("q1", "q2") if "planar_arm_line_n2" in name else ("x", "y")
        return {"name": name, "x_train": x.astype(np.float32), "data_dim": 2, "axis_labels": labels}
    if name in BASE_3D_DATASETS:
        x, _ = generate_dataset(name, cfg)
        labels = ("q1", "q2", "q3") if ("planar_arm_line_n3" in name or "spatial_arm_plane_n3" in name or "spatial_arm_circle_n3" in name) else ("x", "y", "z")
        return {"name": name, "x_train": x.astype(np.float32), "data_dim": 3, "axis_labels": labels}
    if name in BASE_4D_DATASETS:
        x, _ = generate_dataset(name, cfg)
        return {"name": name, "x_train": x.astype(np.float32), "data_dim": 4, "axis_labels": ("q1", "q2", "q3", "q4")}
    if name in BASE_6D_DATASETS:
        if name == "spatial_arm_up_n6":
            # For demo training we only need x_train; avoid expensive large n_grid sampling here.
            print(f"[data] sampling UR5 n6 train set ... (n_train={cfg.n_train})")
            x, _ = sample_ur5_upward_dataset(
                cfg.n_train,
                1,
                seed=cfg.seed,
                urdf_path=cfg.ur5_urdf_path,
                ee_link_index=(None if int(cfg.ur5_ee_link_index) < 0 else int(cfg.ur5_ee_link_index)),
                tool_axis=cfg.ur5_tool_axis,
            )
            print(f"[data] sampled UR5 n6 train set done: x_train={len(x)}")
        elif name == "spatial_arm_up_n6_py":
            print(f"[data] sampling analytic n6 train set ... (n_train={cfg.n_train})")
            x, _ = sample_ur5_upward_dataset_analytic(cfg.n_train, 1, seed=cfg.seed)
            print(f"[data] sampled analytic n6 train set done: x_train={len(x)}")
        else:
            x, _ = generate_dataset(name, cfg)
        return {
            "name": name,
            "x_train": x.astype(np.float32),
            "data_dim": 6,
            "axis_labels": ("q1", "q2", "q3", "q4", "q5", "q6"),
            "constraint_dim_override": 2,
        }
    if name.endswith("_3d_0"):
        base = name[:-5]
        if base not in BASE_2D_DATASETS:
            raise ValueError(f"unknown lifted dataset base for _3d_0: {base}")
        x2, _ = generate_dataset(base, cfg)
        labels = ("q1", "q2", "q3") if "planar_arm_line_n2" in base else ("x", "y", "z")
        return {"name": name, "x_train": _lift_xy_to_3d_zero(x2), "data_dim": 3, "axis_labels": labels}
    if name.endswith("_3d"):
        base = name[:-3]
        if base not in BASE_2D_DATASETS:
            raise ValueError(f"unknown lifted dataset base for _3d: {base}")
        x2, _ = generate_dataset(base, cfg)
        labels = ("q1", "q2", "q3") if "planar_arm_line_n2" in base else ("x", "y", "z")
        return {"name": name, "x_train": _lift_xy_to_3d_var(x2, cfg), "data_dim": 3, "axis_labels": labels}
    raise ValueError(f"unknown dataset '{name}'")


class MLPConstraint(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _eikonal_multi_constraint(model: nn.Module, x: torch.Tensor, lam_ortho: float) -> torch.Tensor:
    f = model(x)
    if f.dim() == 1:
        f = f.unsqueeze(1)
    k = f.shape[1]
    grads = []
    for i in range(k):
        gi = torch.autograd.grad(f[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        grads.append(gi)
    g = torch.stack(grads, dim=1)  # (B, k, d)
    row_norm = torch.linalg.norm(g, dim=2)
    loss_row = ((row_norm - 1.0) ** 2).mean()
    if k <= 1:
        return loss_row
    gram = torch.matmul(g, g.transpose(1, 2))
    eye = torch.eye(k, device=x.device, dtype=x.dtype).unsqueeze(0)
    loss_ortho = ((gram - eye) ** 2).mean()
    return loss_row + float(lam_ortho) * loss_ortho


def train_on_eikonal_only(
    cfg: DemoCfg, x_train: np.ndarray, constraint_dim: int
) -> tuple[nn.Module, dict[str, np.ndarray]]:
    x_t = torch.from_numpy(x_train)
    ds = torch.utils.data.TensorDataset(x_t)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(cfg.device == "cuda"),
        num_workers=0,
    )
    model = MLPConstraint(
        in_dim=x_train.shape[1],
        hidden=cfg.hidden,
        depth=cfg.depth,
        out_dim=max(1, int(constraint_dim)),
    ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    mins, maxs = eval_bounds_from_train(x_train, cfg)
    mins_t = torch.from_numpy(mins.astype(np.float32)).to(cfg.device)
    maxs_t = torch.from_numpy(maxs.astype(np.float32)).to(cfg.device)
    x_train_dev = x_t.to(cfg.device)
    span_t = (maxs_t - mins_t).clamp_min(1e-6)
    near_ratio = float(min(max(cfg.eikonal_near_ratio, 0.0), 1.0))
    near_std_ratio = float(max(cfg.eikonal_near_std_ratio, 1e-5))

    hist: dict[str, list[np.ndarray | float]] = {
        "epoch": [],
        "f_abs_mean": [],
        "grad_norm_mean": [],
        "ortho_err": [],
    }

    model.train()
    for ep in range(cfg.epochs):
        for (xb,) in dl:
            xb = xb.to(cfg.device)
            opt.zero_grad(set_to_none=True)

            f_on = model(xb)
            loss_on = (f_on ** 2).mean()

            bsz, dim = xb.shape
            n_near = max(0, min(int(round(bsz * near_ratio)), bsz))
            n_box = bsz - n_near
            parts = []
            if n_near > 0:
                idx = torch.randint(0, x_train_dev.shape[0], size=(n_near,), device=cfg.device)
                x_near = x_train_dev[idx]
                x_near = x_near + torch.randn_like(x_near) * (near_std_ratio * span_t)
                x_near = torch.max(torch.min(x_near, maxs_t), mins_t)
                parts.append(x_near)
            if n_box > 0:
                x_box = torch.rand((n_box, dim), device=cfg.device)
                x_box = x_box * (maxs_t - mins_t) + mins_t
                parts.append(x_box)
            xr = torch.cat(parts, dim=0)
            xr = xr[torch.randperm(xr.shape[0], device=cfg.device)]
            xr.requires_grad_(True)

            loss_eik = _eikonal_multi_constraint(model, xr, lam_ortho=float(cfg.lam_eikonal_ortho))
            loss = loss_on + cfg.lam_eikonal * loss_eik
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (ep % max(1, int(cfg.metric_eval_every))) == 0:
            x_eval = x_train_dev.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                f_eval = model(x_eval)
                if f_eval.dim() == 1:
                    f_eval = f_eval.unsqueeze(1)
                k = f_eval.shape[1]
                grads = []
                for i in range(k):
                    gi = torch.autograd.grad(
                        f_eval[:, i].sum(),
                        x_eval,
                        create_graph=False,
                        retain_graph=(i < k - 1),
                    )[0]
                    grads.append(gi)
                g = torch.stack(grads, dim=1)  # (N, k, d)
                grad_norm = torch.linalg.norm(g, dim=2)  # (N, k)
                if k > 1:
                    gram = torch.matmul(g, g.transpose(1, 2))
                    eye = torch.eye(k, device=g.device, dtype=g.dtype).unsqueeze(0)
                    ortho_err = torch.mean(torch.abs(gram - eye)).item()
                else:
                    ortho_err = 0.0
            hist["epoch"].append(float(ep + 1))
            hist["f_abs_mean"].append(torch.mean(torch.abs(f_eval), dim=0).detach().cpu().numpy())
            hist["grad_norm_mean"].append(torch.mean(grad_norm, dim=0).detach().cpu().numpy())
            hist["ortho_err"].append(float(ortho_err))
            if ((ep + 1) % max(1, int(cfg.train_log_every))) == 0 or ep == 0 or (ep + 1) == int(cfg.epochs):
                f_m = np.mean(np.abs(f_eval.detach().cpu().numpy()), axis=0)
                g_m = np.mean(grad_norm.detach().cpu().numpy(), axis=0)
                f_s = ", ".join([f"f{i+1}={v:.4f}" for i, v in enumerate(f_m)])
                g_s = ", ".join([f"|grad f{i+1}|={v:.4f}" for i, v in enumerate(g_m)])
                print(
                    f"[train] ep {ep+1:4d}/{cfg.epochs} | {f_s} | {g_s} | ortho={ortho_err:.5f}"
                )

    out_hist = {
        "epoch": np.asarray(hist["epoch"], dtype=np.float32),
        "f_abs_mean": np.asarray(hist["f_abs_mean"], dtype=np.float32),
        "grad_norm_mean": np.asarray(hist["grad_norm_mean"], dtype=np.float32),
        "ortho_err": np.asarray(hist["ortho_err"], dtype=np.float32),
    }
    return model, out_hist


def _plot_training_diagnostics(hist: dict[str, np.ndarray], out_path: str, title: str) -> None:
    ep = hist["epoch"]
    f_abs = hist["f_abs_mean"]
    gnorm = hist["grad_norm_mean"]
    ortho = hist["ortho_err"]
    k = int(f_abs.shape[1]) if f_abs.ndim == 2 else 1

    plt.figure(figsize=(12, 3.6))

    ax1 = plt.subplot(1, 3, 1)
    for i in range(k):
        ax1.plot(ep, f_abs[:, i], lw=1.6, label=f"|f{i+1}| mean")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("on-data mean")
    ax1.set_title("|f_i| on data")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2 = plt.subplot(1, 3, 2)
    for i in range(k):
        ax2.plot(ep, gnorm[:, i], lw=1.6, label=f"||grad f{i+1}|| mean")
    ax2.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("on-data mean")
    ax2.set_title("Gradient Norm")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(ep, ortho, lw=1.8, color="#ef4444")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("mean |J J^T - I|")
    ax3.set_title("Orthogonality Error")
    ax3.grid(alpha=0.25)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def project_trajectory(model: nn.Module, x0: np.ndarray, cfg: DemoCfg, eps_stop: float) -> np.ndarray:
    x = torch.from_numpy(x0).to(cfg.device)
    traj = [x.detach().cpu().numpy()]
    for _ in range(cfg.proj_steps):
        x.requires_grad_(True)
        with torch.enable_grad():
            f = model(x)
            v = 0.5 * (f ** 2).sum(dim=1, keepdim=True)
            grad = torch.autograd.grad(v.sum(), x)[0]
        f_abs = torch.abs(f) if f.dim() > 1 else torch.abs(f).unsqueeze(1)
        if torch.all(f_abs < eps_stop):
            break
        x = (x - cfg.proj_alpha * grad).detach()
        traj.append(x.detach().cpu().numpy())
    return np.stack(traj, axis=0)


def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.full((len(a),), float("nan"), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(b.astype(np.float64))
        d, _ = tree.query(a.astype(np.float64), k=1)
        return d.astype(np.float32)
    except Exception:
        aa = a.astype(np.float32)
        bb = b.astype(np.float32)
        d2 = np.sum((aa[:, None, :] - bb[None, :, :]) ** 2, axis=2)
        return np.sqrt(np.maximum(np.min(d2, axis=1), 0.0)).astype(np.float32)


def _is_arm_dataset(name: str) -> bool:
    return "arm_" in str(name)


def _workspace_embed_for_eval(name: str, q: np.ndarray, cfg: DemoCfg) -> np.ndarray:
    q = q.astype(np.float32)
    if name == "planar_arm_line_n2":
        ee = _planar_fk(q, [1.0, 0.8])[:, -1, :]
        return ee.astype(np.float32)
    if name == "planar_arm_line_n3":
        ee = _planar_fk(q, [1.0, 0.8, 0.6])[:, -1, :]
        return ee.astype(np.float32)
    if name == "spatial_arm_plane_n3":
        ee = _spatial_fk(q, [1.0, 0.8], use_pybullet_n6=False)[:, -1, :]
        return ee.astype(np.float32)
    if name == "spatial_arm_circle_n3":
        ee = _spatial_fk(q, [1.0, 0.8], use_pybullet_n6=False)[:, -1, :]
        return ee.astype(np.float32)
    if name == "spatial_arm_plane_n4":
        ee = _spatial_fk(q, [1.0, 0.8, 0.6], use_pybullet_n6=False)[:, -1, :]
        return ee.astype(np.float32)
    if name in ("spatial_arm_up_n6", "spatial_arm_up_n6_py"):
        ee = _spatial_fk(q, UR5_LINK_LENGTHS, use_pybullet_n6=(name == "spatial_arm_up_n6"))[:, -1, :]
        tool = _spatial_tool_axis_n6(q, use_pybullet=(name == "spatial_arm_up_n6"))
        return np.concatenate([ee.astype(np.float32), tool.astype(np.float32)], axis=1)
    return q.astype(np.float32)


def _sample_eval_seed_points(x_train: np.ndarray, cfg: DemoCfg) -> np.ndarray:
    n_seed = max(64, int(cfg.eval_chamfer_n_seed))
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    span = np.maximum(maxs - mins, 1e-6).astype(np.float32)
    near_ratio = float(np.clip(cfg.eval_chamfer_near_ratio, 0.0, 1.0))
    n_near = int(round(n_seed * near_ratio))
    n_box = max(0, n_seed - n_near)
    out = []
    if n_near > 0:
        idx = np.random.randint(0, len(x_train), size=n_near)
        x0_near = x_train[idx].astype(np.float32).copy()
        noise_std = float(max(cfg.eval_chamfer_near_noise_std_ratio, 1e-6))
        x0_near = x0_near + np.random.randn(*x0_near.shape).astype(np.float32) * (noise_std * span.reshape(1, -1))
        x0_near = np.clip(x0_near, mins.reshape(1, -1), maxs.reshape(1, -1))
        out.append(x0_near.astype(np.float32))
    if n_box > 0:
        out.append(_uniform_in_box(n_box, mins, maxs))
    if not out:
        return _uniform_in_box(n_seed, mins, maxs)
    return np.concatenate(out, axis=0).astype(np.float32)


def evaluate_bidirectional_chamfer(
    model: nn.Module,
    name: str,
    x_train: np.ndarray,
    cfg: DemoCfg,
) -> dict[str, float]:
    n_gt = min(len(x_train), max(64, int(cfg.eval_chamfer_n_gt)))
    gt_idx = np.random.choice(len(x_train), size=n_gt, replace=False)
    gt_samples = x_train[gt_idx].astype(np.float32)

    with torch.no_grad():
        f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
        if f_on.dim() == 1:
            f_on = f_on.unsqueeze(1)
        h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy().reshape(-1)
    eps_stop = float(np.percentile(np.abs(h_on), cfg.zero_eps_quantile))

    x0 = _sample_eval_seed_points(x_train, cfg)
    if _is_arm_dataset(name):
        x0 = _wrap_np_pi(x0)
    traj = project_trajectory(model, x0, cfg, eps_stop=eps_stop)
    learned_samples = traj[-1].astype(np.float32)
    if _is_arm_dataset(name):
        learned_samples = _wrap_np_pi(learned_samples)

    if _is_arm_dataset(name):
        a = _workspace_embed_for_eval(name, gt_samples, cfg)
        b = _workspace_embed_for_eval(name, learned_samples, cfg)
        dist_space = "workspace"
    else:
        a = gt_samples
        b = learned_samples
        dist_space = "data_space"

    d_gt_to_learned = _nn_dist(a, b)
    d_learned_to_gt = _nn_dist(b, a)
    chamfer = float(np.mean(d_gt_to_learned) + np.mean(d_learned_to_gt))
    return {
        "bidirectional_chamfer": chamfer,
        "gt_to_learned_mean": float(np.mean(d_gt_to_learned)),
        "learned_to_gt_mean": float(np.mean(d_learned_to_gt)),
        "gt_to_learned_p95": float(np.percentile(d_gt_to_learned, 95)),
        "learned_to_gt_p95": float(np.percentile(d_learned_to_gt, 95)),
        "dist_space": dist_space,
        "n_gt": int(len(gt_samples)),
        "n_learned": int(len(learned_samples)),
        "eps_stop": float(eps_stop),
    }


def _plot_constraint_2d(
    model: nn.Module,
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str],
    cfg: DemoCfg,
) -> None:
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    # For joint-space plots, keep a full angular window when possible.
    if axis_labels[0].startswith("q") and axis_labels[1].startswith("q"):
        mins = mins.copy()
        maxs = maxs.copy()
        mins[0] = min(float(mins[0]), -np.pi)
        maxs[0] = max(float(maxs[0]), np.pi)
        mins[1] = min(float(mins[1]), -np.pi)
        maxs[1] = max(float(maxs[1]), np.pi)
    xx, yy = np.meshgrid(np.linspace(float(mins[0]), float(maxs[0]), 360), np.linspace(float(mins[1]), float(maxs[1]), 360))
    grid = np.stack([xx, yy], axis=2).reshape(-1, 2).astype(np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        fg = model(torch.from_numpy(grid).to(device))
        if fg.dim() == 1:
            fg = fg.unsqueeze(1)
        f1 = fg[:, 0].detach().cpu().numpy().reshape(xx.shape)
        h = torch.linalg.norm(fg, dim=1).detach().cpu().numpy().reshape(xx.shape)
        fon = model(torch.from_numpy(x_train).to(device))
        if fon.dim() == 1:
            fon = fon.unsqueeze(1)
        h_on = torch.linalg.norm(fon, dim=1).detach().cpu().numpy()
    eps_h = float(np.percentile(np.abs(h_on), cfg.zero_eps_quantile))
    cap = max(float(np.percentile(h, 95)), eps_h * 1.5, 1e-6)

    plt.figure(figsize=(7.6, 6.2))
    hm = plt.contourf(xx, yy, h, levels=np.linspace(0.0, cap, 30), cmap="viridis")
    plt.colorbar(hm, label="||F||")
    plt.contourf(xx, yy, h, levels=[0.0, eps_h], colors=["#ffa500"], alpha=0.40)
    plt.contour(xx, yy, f1, levels=[0.0], colors=["red"], linewidths=1.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=10, c="gray", alpha=0.65, zorder=3)
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], "-", color="green", linewidth=1.0, alpha=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(float(mins[0]), float(maxs[0]))
    plt.ylim(float(mins[1]), float(maxs[1]))
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_zero_surfaces_3d(
    model: nn.Module,
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str, str],
    cfg: DemoCfg,
) -> None:
    mins = np.min(x_train, axis=0)
    maxs = np.max(x_train, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.15 * span
    mins, maxs = mins - pad, maxs + pad

    n = max(16, int(cfg.surface_plot_n))
    xx, yy, zz = np.meshgrid(
        np.linspace(float(mins[0]), float(maxs[0]), n),
        np.linspace(float(mins[1]), float(maxs[1]), n),
        np.linspace(float(mins[2]), float(maxs[2]), n),
        indexing="ij",
    )
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    device = next(model.parameters()).device

    out_list = []
    with torch.no_grad():
        for s in range(0, len(pts), max(1024, int(cfg.surface_eval_chunk))):
            e = min(len(pts), s + max(1024, int(cfg.surface_eval_chunk)))
            f = model(torch.from_numpy(pts[s:e]).to(device))
            if f.dim() == 1:
                f = f.unsqueeze(1)
            out_list.append(f.detach().cpu().numpy())
    f_all = np.concatenate(out_list, axis=0)
    f1 = f_all[:, 0]
    has_f2 = f_all.shape[1] >= 2
    f2 = f_all[:, 1] if has_f2 else None

    with torch.no_grad():
        f_train = model(torch.from_numpy(x_train).to(device))
        if f_train.dim() == 1:
            f_train = f_train.unsqueeze(1)
        eps1 = max(float(np.percentile(np.abs(f_train[:, 0].detach().cpu().numpy()), 90)), 1e-4)
        if has_f2:
            eps2 = max(float(np.percentile(np.abs(f_train[:, 1].detach().cpu().numpy()), 90)), 1e-4)
            h_on = torch.linalg.norm(f_train[:, :2], dim=1).detach().cpu().numpy()
            eps_h = max(float(np.percentile(h_on, 90)), 1e-6)

    rendered = False
    verts1 = faces1 = verts2 = faces2 = None
    if bool(cfg.surface_use_marching_cubes):
        try:
            from skimage import measure  # type: ignore

            f1_vol = f1.reshape(n, n, n)
            dx = float((maxs[0] - mins[0]) / max(1, n - 1))
            dy = float((maxs[1] - mins[1]) / max(1, n - 1))
            dz = float((maxs[2] - mins[2]) / max(1, n - 1))
            lvl1 = 0.0 if (float(np.min(f1_vol)) <= 0.0 <= float(np.max(f1_vol))) else eps1
            vol1 = f1_vol if lvl1 == 0.0 else np.abs(f1_vol)
            verts1, faces1, _, _ = measure.marching_cubes(vol1, level=lvl1, spacing=(dx, dy, dz))
            verts1 += np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
            if has_f2 and f2 is not None:
                f2_vol = f2.reshape(n, n, n)
                lvl2 = 0.0 if (float(np.min(f2_vol)) <= 0.0 <= float(np.max(f2_vol))) else eps2
                vol2 = f2_vol if lvl2 == 0.0 else np.abs(f2_vol)
                verts2, faces2, _, _ = measure.marching_cubes(vol2, level=lvl2, spacing=(dx, dy, dz))
                verts2 += np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
            rendered = True
        except Exception:
            rendered = False

    p1 = p2 = p12 = None
    if not rendered:
        p1 = pts[np.abs(f1) <= eps1]
        if len(p1) > cfg.surface_max_points:
            idx = np.random.choice(len(p1), size=int(cfg.surface_max_points), replace=False)
            p1 = p1[idx]
        if has_f2 and f2 is not None:
            p2 = pts[np.abs(f2) <= eps2]
            if len(p2) > cfg.surface_max_points:
                idx = np.random.choice(len(p2), size=int(cfg.surface_max_points), replace=False)
                p2 = p2[idx]
            h_grid = np.sqrt(f1 * f1 + f2 * f2)
            p12 = pts[h_grid <= eps_h]
            if len(p12) > 1200:
                idx = np.random.choice(len(p12), size=1200, replace=False)
                p12 = p12[idx]

    train_plot = x_train
    if len(train_plot) > cfg.plot_train_max_points:
        idx = np.random.choice(len(train_plot), size=int(cfg.plot_train_max_points), replace=False)
        train_plot = train_plot[idx]
    step = max(1, int(np.ceil(traj.shape[1] / max(1, int(cfg.plot_traj_max_count)))))
    traj_ids = list(range(0, traj.shape[1], step))
    stride = max(1, int(cfg.plot_traj_stride))

    def _draw_common(ax) -> None:
        ax.scatter(train_plot[:, 0], train_plot[:, 1], train_plot[:, 2], s=5, c="gray", alpha=0.24, label="train")
        for i in traj_ids:
            ax.plot(
                traj[::stride, i, 0], traj[::stride, i, 1], traj[::stride, i, 2],
                "-", color="green", linewidth=1.2 if i == traj_ids[0] else 0.8, alpha=0.65,
                label="traj" if i == traj_ids[0] else None,
            )
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.set_xlim(float(mins[0]), float(maxs[0]))
        ax.set_ylim(float(mins[1]), float(maxs[1]))
        ax.set_zlim(float(mins[2]), float(maxs[2]))
        ax.set_proj_type("persp")
        ax.view_init(elev=28, azim=-42)

    def _add_surface(ax, verts, faces, color: str, alpha: float, label: str) -> None:
        poly = Poly3DCollection(verts[faces], alpha=alpha, facecolor=color, edgecolor=(0, 0, 0, 0.12), linewidth=0.1)
        poly.set_label(label)
        ax.add_collection3d(poly)

    if has_f2:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133, projection="3d")
        if rendered and verts1 is not None and faces1 is not None and verts2 is not None and faces2 is not None:
            _add_surface(ax1, verts1, faces1, "#22d3ee", 0.24, "f1=0")
            _add_surface(ax2, verts2, faces2, "#f472b6", 0.24, "f2=0")
            _add_surface(ax3, verts1, faces1, "#22d3ee", 0.14, "f1=0")
            _add_surface(ax3, verts2, faces2, "#f472b6", 0.14, "f2=0")
        else:
            if p1 is not None and len(p1) > 0:
                ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.2, c="#22d3ee", alpha=0.26, label=f"f1≈0 (p90={eps1:.3g})")
                ax3.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.0, c="#22d3ee", alpha=0.18, label=f"f1≈0 (p90={eps1:.3g})")
            if p2 is not None and len(p2) > 0:
                ax2.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=1.2, c="#f472b6", alpha=0.26, label=f"f2≈0 (p90={eps2:.3g})")
                ax3.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=1.0, c="#f472b6", alpha=0.18, label=f"f2≈0 (p90={eps2:.3g})")
            if p12 is not None and len(p12) > 0:
                ax3.scatter(p12[:, 0], p12[:, 1], p12[:, 2], s=7, c="red", alpha=0.95, label="intersection")
        _draw_common(ax1)
        _draw_common(ax2)
        _draw_common(ax3)
        ax1.set_title("f1")
        ax2.set_title("f2")
        ax3.set_title("f1 + f2 + intersection")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper left", fontsize=8)
        ax3.legend(loc="upper left", fontsize=8)
    else:
        fig = plt.figure(figsize=(7, 6))
        ax1 = fig.add_subplot(111, projection="3d")
        if rendered and verts1 is not None and faces1 is not None:
            _add_surface(ax1, verts1, faces1, "#ef4444", 0.24, "f1=0")
        elif p1 is not None and len(p1) > 0:
            ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.5, c="#ef4444", alpha=0.30, label=f"f1≈0 (p90={eps1:.3g})")
        _draw_common(ax1)
        ax1.set_title("f1 (codim=1)")
        ax1.legend(loc="upper left", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if bool(cfg.show_3d_plot):
        plt.show()
    plt.close(fig)


def _plot_highdim_pca(
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    # Lightweight fallback for data_dim > 3: visualize train manifold and trajectories in PCA-2D.
    x = x_train.astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T  # (D,2)
    emb_train = (x - mu) @ basis

    x0 = traj[0]
    xT = traj[-1]
    emb_start = (x0 - mu) @ basis
    emb_end = (xT - mu) @ basis

    plt.figure(figsize=(7.0, 6.0))
    if len(emb_train) > 2500:
        idx = np.random.choice(len(emb_train), size=2500, replace=False)
        emb_train = emb_train[idx]
    plt.scatter(emb_train[:, 0], emb_train[:, 1], s=5, c="gray", alpha=0.25, label="train")

    step = max(1, int(np.ceil(traj.shape[1] / 32)))
    for i in range(0, traj.shape[1], step):
        tr = (traj[:, i, :] - mu) @ basis
        plt.plot(tr[:, 0], tr[:, 1], "-", color="green", lw=0.9, alpha=0.7)
    plt.scatter(emb_start[:, 0], emb_start[:, 1], s=12, c="royalblue", alpha=0.9, label="traj start")
    plt.scatter(emb_end[:, 0], emb_end[:, 1], s=12, c="crimson", alpha=0.9, label="traj end")
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.title(title + " (PCA-2D)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _pick_far_pair(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        return x[0], x[0]
    m = min(len(x), 192)
    idx = np.random.choice(len(x), size=m, replace=False)
    xs = x[idx]
    d2 = np.sum((xs[:, None, :] - xs[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return xs[i].astype(np.float32), xs[j].astype(np.float32)


def _pick_far_pair_workspace(
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
    ee = _planar_fk(q, lengths)[:, -1, :]  # (N,2)
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


def _plan_path_continuous(
    model: nn.Module,
    q_lin: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    cfg: DemoCfg,
    steps: int | None = None,
    lr: float | None = None,
    lam_smooth: float | None = None,
) -> np.ndarray:
    def _wrap_torch_pi(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + np.pi, 2.0 * np.pi) - np.pi

    def _ang_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Shortest periodic angle delta from a -> b in [-pi, pi].
        return _wrap_torch_pi(b - a)

    steps_v = int(cfg.plan_opt_steps if steps is None else steps)
    lr_v = float(cfg.plan_opt_lr if lr is None else lr)
    lam_smooth_v = float(cfg.plan_opt_lam_smooth if lam_smooth is None else lam_smooth)
    # Relative trust radius from initial path scale (periodic angle metric).
    with torch.no_grad():
        v0 = _ang_delta(
            torch.tensor(q_lin[:-1], device=cfg.device, dtype=torch.float32),
            torch.tensor(q_lin[1:], device=cfg.device, dtype=torch.float32),
        )
        mean_step0 = float(torch.mean(torch.linalg.norm(v0, dim=1)).item()) if v0.shape[0] > 0 else 1.0
    trust_delta = float(max(1e-6, float(cfg.plan_trust_scale) * max(mean_step0, 1e-6)))

    q = torch.tensor(q_lin.astype(np.float32), device=cfg.device, requires_grad=True)
    q0 = torch.tensor(q_start.astype(np.float32), device=cfg.device)
    qT = torch.tensor(q_goal.astype(np.float32), device=cfg.device)

    opt = torch.optim.Adam([q], lr=lr_v)
    for _ in range(steps_v):
        q_prev = q.detach().clone()
        opt.zero_grad(set_to_none=True)
        f = model(q)
        if f.dim() == 1:
            f = f.unsqueeze(1)
        # Keep path on manifold.
        loss_man = (f ** 2).mean()
        # Periodic smoothness on torus:
        # v_t = wrap(q_{t+1}-q_t), smooth v_t over time.
        v = _ang_delta(q[:-1], q[1:])  # (T-1, J)
        loss_len_joint = (v ** 2).mean()
        if q.shape[0] >= 3:
            dv = _ang_delta(v[:-1], v[1:])  # (T-2, J)
            loss_smooth = (dv ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=q.device)
        loss = (
            float(cfg.plan_lam_manifold) * loss_man
            + lam_smooth_v * loss_smooth
            + float(cfg.plan_lam_len_joint) * loss_len_joint
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            # Trust region: bound waypoint update to avoid sudden jumps.
            dq = _ang_delta(q_prev, q)
            dn = torch.linalg.norm(dq, dim=1, keepdim=True).clamp_min(1e-9)
            scale = torch.minimum(torch.ones_like(dn), torch.full_like(dn, trust_delta) / dn)
            q[:] = _wrap_torch_pi(q_prev + dq * scale)
            q[:] = _wrap_torch_pi(q)
            q[0] = q0
            q[-1] = qT
    return q.detach().cpu().numpy().astype(np.float32)


def _planar_fk(q: np.ndarray, lengths: list[float]) -> np.ndarray:
    # q: (T, J), return joints: (T, J+1, 2)
    t, j = q.shape
    joints = np.zeros((t, j + 1, 2), dtype=np.float32)
    ang = np.cumsum(q, axis=1)
    for k in range(j):
        joints[:, k + 1, 0] = joints[:, k, 0] + float(lengths[k]) * np.cos(ang[:, k])
        joints[:, k + 1, 1] = joints[:, k, 1] + float(lengths[k]) * np.sin(ang[:, k])
    return joints


def _wrap_np_pi(a: np.ndarray) -> np.ndarray:
    return ((a + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _smoothstep(s: np.ndarray) -> np.ndarray:
    return (s * s * (3.0 - 2.0 * s)).astype(np.float32)


def _angle_shortest_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _wrap_np_pi(b - a)


def _angle_interp_shortest(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    d = _angle_shortest_delta(a, b)
    return _wrap_np_pi(a + s * d)


def _init_path_joint_spline(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    n_waypoints: int,
    mid_noise: float,
) -> np.ndarray:
    # Pure joint-space init: shortest-arc start->mid->goal with smoothstep blending.
    q_start = q_start.astype(np.float32)
    q_goal = q_goal.astype(np.float32)
    d = _angle_shortest_delta(q_start, q_goal)
    q_mid = _wrap_np_pi(q_start + 0.5 * d + float(mid_noise) * np.random.randn(*q_start.shape).astype(np.float32))

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


def _spatial_fk_n3(q: np.ndarray, lengths: list[float]) -> np.ndarray:
    # q: (T,3): q1 yaw, q2/q3 pitch chain
    t = q.shape[0]
    l1, l2 = float(lengths[0]), float(lengths[1])
    q1 = q[:, 0]
    t2 = q[:, 1]
    t3 = q[:, 1] + q[:, 2]
    r1 = l1 * np.cos(t2)
    z1 = l1 * np.sin(t2)
    r2 = r1 + l2 * np.cos(t3)
    z2 = z1 + l2 * np.sin(t3)
    joints = np.zeros((t, 3, 3), dtype=np.float32)  # base + elbow + ee
    joints[:, 1, 0] = r1 * np.cos(q1)
    joints[:, 1, 1] = r1 * np.sin(q1)
    joints[:, 1, 2] = z1
    joints[:, 2, 0] = r2 * np.cos(q1)
    joints[:, 2, 1] = r2 * np.sin(q1)
    joints[:, 2, 2] = z2
    return joints


def _spatial_fk_n4(q: np.ndarray, lengths: list[float]) -> np.ndarray:
    # q: (T,4): q1 yaw, q2/q3/q4 pitch-chain
    t = q.shape[0]
    l1, l2, l3 = float(lengths[0]), float(lengths[1]), float(lengths[2])
    q1 = q[:, 0]
    t2 = q[:, 1]
    t3 = q[:, 1] + q[:, 2]
    t4 = q[:, 1] + q[:, 2] + q[:, 3]
    r1 = l1 * np.cos(t2)
    z1 = l1 * np.sin(t2)
    r2 = r1 + l2 * np.cos(t3)
    z2 = z1 + l2 * np.sin(t3)
    r3 = r2 + l3 * np.cos(t4)
    z3 = z2 + l3 * np.sin(t4)
    joints = np.zeros((t, 4, 3), dtype=np.float32)  # base + 3 joints(ee)
    joints[:, 1, 0] = r1 * np.cos(q1)
    joints[:, 1, 1] = r1 * np.sin(q1)
    joints[:, 1, 2] = z1
    joints[:, 2, 0] = r2 * np.cos(q1)
    joints[:, 2, 1] = r2 * np.sin(q1)
    joints[:, 2, 2] = z2
    joints[:, 3, 0] = r3 * np.cos(q1)
    joints[:, 3, 1] = r3 * np.sin(q1)
    joints[:, 3, 2] = z3
    return joints


def _spatial_fk_n6(q: np.ndarray, lengths: list[float], use_pybullet: bool = True) -> np.ndarray:
    backend = _get_ur5_backend() if use_pybullet else None
    if backend is not None:
        chain, _, _ = backend.fk_batch(q.astype(np.float32))
        return chain.astype(np.float32)

    # q: (T,6), axes: z, y, y, y, x, z
    t = q.shape[0]
    l = [float(v) for v in lengths]
    out = np.zeros((t, 7, 3), dtype=np.float32)  # base + 6 joints/ee
    for i in range(t):
        qi = q[i]
        R = np.eye(3, dtype=np.float32)
        p = np.zeros((3,), dtype=np.float32)
        out[i, 0] = p
        axes = ("z", "y", "y", "y", "x", "z")
        for j, ax in enumerate(axes):
            a = float(qi[j])
            if ax == "x":
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
            elif ax == "y":
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
            else:
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            R = R @ Rj
            p = p + R @ np.array([l[j], 0.0, 0.0], dtype=np.float32)
            out[i, j + 1] = p
    return out


def _spatial_tool_axis_n6(q: np.ndarray, use_pybullet: bool = True) -> np.ndarray:
    backend = _get_ur5_backend() if use_pybullet else None
    if backend is not None:
        axis = backend.tool_axis_batch(q.astype(np.float32))
        n = np.linalg.norm(axis, axis=1, keepdims=True)
        return (axis / np.maximum(n, 1e-8)).astype(np.float32)

    # Tool/gripper axis in world frame for n6 chain.
    # Convention: tool forward axis is local +x.
    t = q.shape[0]
    out = np.zeros((t, 3), dtype=np.float32)
    axes = ("z", "y", "y", "y", "x", "z")
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(t):
        R = np.eye(3, dtype=np.float32)
        for j, ax in enumerate(axes):
            a = float(q[i, j])
            if ax == "x":
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
            elif ax == "y":
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
            else:
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            R = R @ Rj
        out[i] = (R @ ex).astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-8)


def _spatial_fk(q: np.ndarray, lengths: list[float], use_pybullet_n6: bool = True) -> np.ndarray:
    if q.shape[1] == 3:
        return _spatial_fk_n3(q, lengths)
    if q.shape[1] == 4:
        return _spatial_fk_n4(q, lengths)
    if q.shape[1] == 6:
        return _spatial_fk_n6(q, lengths, use_pybullet=use_pybullet_n6)
    raise ValueError(f"unsupported spatial dof={q.shape[1]}")


def _fk_ee_torch(q: torch.Tensor, lengths: list[float], is_spatial: bool) -> torch.Tensor:
    # q: (B, J) -> ee: (B, 2 or 3)
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
        # Position-only FK for init path (rotation-based chain: z, y, y, y, x, z).
        x = torch.zeros((q.shape[0],), dtype=q.dtype, device=q.device)
        y = torch.zeros((q.shape[0],), dtype=q.dtype, device=q.device)
        z = torch.zeros((q.shape[0],), dtype=q.dtype, device=q.device)
        # Use a lightweight approximation for differentiable initialization:
        # first 4 joints dominate position, wrist joints have shorter links.
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
        z = z6
        return torch.stack([x, y, z], dim=1)
    raise ValueError(f"unsupported spatial dof={q.shape[1]}")


def _init_path_via_workspace_ik(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lengths: list[float],
    is_spatial: bool,
    use_pybullet_n6: bool,
    n_waypoints: int,
    device: str,
) -> np.ndarray:
    # Build a smooth unconstrained joint path by tracking a linear workspace EE path.
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
            # Keep each waypoint close to previous one to avoid branch jumps.
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


def _plot_planar_arm_planning(
    model: nn.Module,
    name: str,
    x_train: np.ndarray,
    out_path: str,
    cfg: DemoCfg,
    render_pybullet: bool = True,
) -> list[np.ndarray]:
    if name == "planar_arm_line_n2":
        lengths = [1.0, 0.8]
        y_line = 0.3
        is_spatial = False
        use_pybullet_n6 = False
    elif name == "planar_arm_line_n3":
        lengths = [1.0, 0.8, 0.6]
        y_line = 0.35
        is_spatial = False
        use_pybullet_n6 = False
    elif name == "spatial_arm_plane_n3":
        lengths = [1.0, 0.8]
        y_line = 0.35  # here means z-plane value
        is_spatial = True
        use_pybullet_n6 = False
    elif name == "spatial_arm_plane_n4":
        lengths = [1.0, 0.8, 0.6]
        y_line = 0.35  # here means z-plane value
        is_spatial = True
        use_pybullet_n6 = False
    elif name == "spatial_arm_up_n6":
        lengths = list(UR5_LINK_LENGTHS)
        y_line = None
        is_spatial = True
        use_pybullet_n6 = True
    elif name == "spatial_arm_up_n6_py":
        lengths = list(UR5_LINK_LENGTHS)
        y_line = None
        is_spatial = True
        use_pybullet_n6 = False
    else:
        return []

    # Sample start/goal from a denser manifold candidate set, then enforce workspace distance.
    try:
        x_dense, grid_dense = generate_dataset(name, cfg)
        cand = grid_dense if (grid_dense is not None and len(grid_dense) >= 2) else x_dense
        if cand.shape[1] != x_train.shape[1]:
            cand = x_train
    except Exception:
        cand = x_train

    cases: list[tuple[np.ndarray, np.ndarray, float]] = []
    lo = float(cfg.plan_pair_min_ratio) * max(float(sum(lengths)), 1e-6)
    hi = float(cfg.plan_pair_max_ratio) * max(float(sum(lengths)), 1e-6)
    if is_spatial:
        q_c = cand.astype(np.float32)
        ee_c = _spatial_fk(q_c, lengths, use_pybullet_n6=use_pybullet_n6)[:, -1, :]  # (N,3)
        target = 0.5 * (lo + hi)
    else:
        q_c = None
        ee_c = None
        target = 0.5 * (lo + hi)
    for _ in range(3):
        if not is_spatial:
            q_start, q_goal, ee_dist, _ = _pick_far_pair_workspace(
                x=cand.astype(np.float32),
                lengths=lengths,
                min_ratio=float(cfg.plan_pair_min_ratio),
                max_ratio=float(cfg.plan_pair_max_ratio),
                tries=int(cfg.plan_pair_tries),
            )
        else:
            best = None
            best_delta = 1e18
            for _t in range(max(1, int(cfg.plan_pair_tries))):
                i = int(np.random.randint(0, len(q_c)))
                j = int(np.random.randint(0, len(q_c)))
                if i == j:
                    continue
                d = float(np.linalg.norm(ee_c[i] - ee_c[j]))
                if lo <= d <= hi:
                    best = (q_c[i], q_c[j], d)
                    break
                delta = abs(d - target)
                if delta < best_delta:
                    best_delta = delta
                    best = (q_c[i], q_c[j], d)
            assert best is not None
            q_start, q_goal, ee_dist = best[0], best[1], float(best[2])
        cases.append((q_start, q_goal, ee_dist))

    # Precompute constraint 0-level overlay once.
    contour_2d = None
    zpts_3d = None
    if x_train.shape[1] == 2:
        mins, maxs = eval_bounds_from_train(x_train, cfg)
        mins = mins.copy()
        maxs = maxs.copy()
        mins[0] = min(float(mins[0]), -np.pi)
        maxs[0] = max(float(maxs[0]), np.pi)
        mins[1] = min(float(mins[1]), -np.pi)
        maxs[1] = max(float(maxs[1]), np.pi)
        q1g, q2g = np.meshgrid(
            np.linspace(float(mins[0]), float(maxs[0]), 260),
            np.linspace(float(mins[1]), float(maxs[1]), 260),
        )
        grid = np.stack([q1g, q2g], axis=2).reshape(-1, 2).astype(np.float32)
        with torch.no_grad():
            fg = model(torch.from_numpy(grid).to(cfg.device))
            if fg.dim() == 1:
                fg = fg.unsqueeze(1)
            f1 = fg[:, 0].detach().cpu().numpy().reshape(q1g.shape)
        contour_2d = (q1g, q2g, f1, mins, maxs)
    elif x_train.shape[1] == 3:
        mins3, maxs3 = eval_bounds_from_train(x_train, cfg)
        q1g, q2g, q3g = np.meshgrid(
            np.linspace(float(mins3[0]), float(maxs3[0]), 34),
            np.linspace(float(mins3[1]), float(maxs3[1]), 34),
            np.linspace(float(mins3[2]), float(maxs3[2]), 34),
            indexing="ij",
        )
        grid3 = np.stack([q1g.ravel(), q2g.ravel(), q3g.ravel()], axis=1).astype(np.float32)
        with torch.no_grad():
            f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
            if f_on.dim() == 1:
                f_on = f_on.unsqueeze(1)
            eps0 = float(np.percentile(np.abs(f_on[:, 0].detach().cpu().numpy()), cfg.zero_eps_quantile))
            vals = []
            chunk = max(2048, int(cfg.surface_eval_chunk))
            for s in range(0, len(grid3), chunk):
                e = min(len(grid3), s + chunk)
                fg = model(torch.from_numpy(grid3[s:e]).to(cfg.device))
                if fg.dim() == 1:
                    fg = fg.unsqueeze(1)
                vals.append(fg[:, 0].detach().cpu().numpy())
            f1g = np.concatenate(vals, axis=0)
        zmask = np.abs(f1g) <= max(eps0, 1e-4)
        zpts_3d = grid3[zmask]
        if len(zpts_3d) > 4500:
            idx = np.random.choice(len(zpts_3d), size=4500, replace=False)
            zpts_3d = zpts_3d[idx]

    fig = plt.figure(figsize=(10.5, 12.0))
    q_paths_render: list[np.ndarray] = []
    for row, (q_start, q_goal, ee_dist) in enumerate(cases):
        if str(cfg.plan_init_mode).lower() == "workspace_ik":
            q_lin = _init_path_via_workspace_ik(
                q_start=q_start.astype(np.float32),
                q_goal=q_goal.astype(np.float32),
                lengths=lengths,
                is_spatial=is_spatial,
                use_pybullet_n6=use_pybullet_n6,
                n_waypoints=140,
                device=str(cfg.device),
            )
        else:
            q_lin = _init_path_joint_spline(
                q_start=q_start.astype(np.float32),
                q_goal=q_goal.astype(np.float32),
                n_waypoints=140,
                mid_noise=float(cfg.plan_joint_mid_noise),
            )
        q_path = _plan_path_continuous(
            model=model,
            q_lin=q_lin.astype(np.float32),
            q_start=q_start,
            q_goal=q_goal,
            cfg=cfg,
        )
        q_paths_render.append(q_path.astype(np.float32))
        q_init = q_lin.astype(np.float32)
        if not is_spatial:
            joints = _planar_fk(q_path, lengths)
            ee = joints[:, -1, :]
        else:
            joints = _spatial_fk(q_path, lengths, use_pybullet_n6=use_pybullet_n6)
            ee = joints[:, -1, :]
            joints_init = _spatial_fk(q_init, lengths, use_pybullet_n6=use_pybullet_n6)
            ee_init = joints_init[:, -1, :]

        left_idx = 2 * row + 1
        right_idx = 2 * row + 2
        if q_path.shape[1] == 2:
            ax1 = fig.add_subplot(3, 2, left_idx)
            assert contour_2d is not None
            q1g, q2g, f1, mins, maxs = contour_2d
            ax1.contour(q1g, q2g, f1, levels=[0.0], colors=["red"], linewidths=1.4, alpha=0.95)
            q_unw = np.unwrap(q_path, axis=0)
            ax1.plot(q_unw[:, 0], q_unw[:, 1], "-", color="#0ea5e9", lw=1.2, alpha=0.85, label="planned")
            ax1.scatter([q_unw[0, 0], q_unw[-1, 0]], [q_unw[0, 1], q_unw[-1, 1]], c=["blue", "red"], s=20, zorder=3)
            ax1.set_xlim(float(mins[0]), float(maxs[0]))
            ax1.set_ylim(float(mins[1]), float(maxs[1]))
            ax1.set_aspect("equal", adjustable="box")
            ax1.set_xlabel("q1")
            ax1.set_ylabel("q2")
            ax1.set_title(f"Joint Path #{row+1} (ee={ee_dist:.2f})", fontsize=10)
            ax1.grid(alpha=0.25)
            ax1.legend(loc="best", fontsize=8)
        elif q_path.shape[1] == 3:
            ax1 = fig.add_subplot(3, 2, left_idx, projection="3d")
            q_unw = np.unwrap(q_path, axis=0)
            if zpts_3d is not None and len(zpts_3d) > 0:
                ax1.scatter(zpts_3d[:, 0], zpts_3d[:, 1], zpts_3d[:, 2], s=1.0, c="red", alpha=0.18)
            ax1.plot(q_unw[:, 0], q_unw[:, 1], q_unw[:, 2], "-", color="#0ea5e9", lw=1.0, alpha=0.8, label="planned")
            ax1.scatter([q_unw[0, 0]], [q_unw[0, 1]], [q_unw[0, 2]], c="blue", s=20)
            ax1.scatter([q_unw[-1, 0]], [q_unw[-1, 1]], [q_unw[-1, 2]], c="red", s=20)
            ax1.set_xlabel("q1")
            ax1.set_ylabel("q2")
            ax1.set_zlabel("q3")
            ax1.set_title(f"Joint Path #{row+1} (ee={ee_dist:.2f})", fontsize=10)
            ax1.legend(loc="best", fontsize=8)
        else:
            # 4D joint path: PCA to 3D for visualization.
            ax1 = fig.add_subplot(3, 2, left_idx, projection="3d")
            q_ref = x_train.astype(np.float32)
            mu = q_ref.mean(axis=0, keepdims=True)
            q0 = q_ref - mu
            _, _, vt = np.linalg.svd(q0, full_matrices=False)
            basis = vt[:3].T  # (4,3)
            emb = (q_path - mu) @ basis
            q_unw = np.unwrap(q_path, axis=0)
            emb_unw = (q_unw - mu) @ basis
            q_init_unw = np.unwrap(q_init, axis=0)
            emb_init = (q_init_unw - mu) @ basis
            emb_s = emb_unw[0:1]
            emb_g = emb_unw[-1:]
            if name in ("spatial_arm_up_n6", "spatial_arm_up_n6_py"):
                ax1.plot(emb_init[:, 0], emb_init[:, 1], emb_init[:, 2], "--", color="gray", lw=1.1, alpha=0.85, label="init (pre-proj)")
            ax1.plot(emb_unw[:, 0], emb_unw[:, 1], emb_unw[:, 2], "-", color="#0ea5e9", lw=1.2, alpha=0.85, label="planned")
            ax1.scatter([emb_s[0, 0]], [emb_s[0, 1]], [emb_s[0, 2]], c="blue", s=20)
            ax1.scatter([emb_g[0, 0]], [emb_g[0, 1]], [emb_g[0, 2]], c="red", s=20)
            ax1.set_xlabel("pc1")
            ax1.set_ylabel("pc2")
            ax1.set_zlabel("pc3")
            ax1.set_title(f"Joint Path #{row+1} (4D path in PCA-3D)", fontsize=10)

        if not is_spatial:
            ax2 = fig.add_subplot(3, 2, right_idx)
            ax2.plot(ee[:, 0], ee[:, 1], "-", color="green", lw=1.6, label="ee trail")
            step = max(1, len(joints) // 12)
            idx_list = list(range(0, len(joints), step))
            if idx_list[-1] != len(joints) - 1:
                idx_list.append(len(joints) - 1)
            n_seg = max(1, len(idx_list) - 1)
            for k, i in enumerate(idx_list):
                c = 0.88 - 0.70 * (k / n_seg)
                arm_color = (c, c, c)
                ax2.plot(joints[i, :, 0], joints[i, :, 1], "-", color=arm_color, alpha=0.95, lw=1.1)
            ax2.plot(joints[0, :, 0], joints[0, :, 1], "-", color="blue", lw=1.3)
            ax2.plot(joints[-1, :, 0], joints[-1, :, 1], "-", color="red", lw=1.3)
            ax2.scatter([ee[0, 0]], [ee[0, 1]], c="blue", s=26, zorder=5)
            ax2.scatter([ee[-1, 0]], [ee[-1, 1]], c="red", s=26, zorder=5)
            ax2.axhline(float(y_line), color="orange", linestyle="--", linewidth=1.7, alpha=0.9, label="GT line")
            reach = float(sum(lengths)) + 0.15
            ax2.set_xlim(-reach, reach)
            ax2.set_ylim(-reach, reach)
            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(alpha=0.25)
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_title(f"Workspace #{row+1}", fontsize=10)
        else:
            ax2 = fig.add_subplot(3, 2, right_idx, projection="3d")
            if name in ("spatial_arm_up_n6", "spatial_arm_up_n6_py"):
                ax2.plot(ee_init[:, 0], ee_init[:, 1], ee_init[:, 2], "--", color="gray", lw=1.2, alpha=0.85, label="init ee")
                ax2.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color="green", lw=1.8, label="planned ee")
            else:
                ax2.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color="green", lw=1.6, label="ee trail")
                step = max(1, len(joints) // 12)
                idx_list = list(range(0, len(joints), step))
                if idx_list[-1] != len(joints) - 1:
                    idx_list.append(len(joints) - 1)
                n_seg = max(1, len(idx_list) - 1)
                for k, i in enumerate(idx_list):
                    c = 0.88 - 0.70 * (k / n_seg)
                    arm_color = (c, c, c)
                    ax2.plot(joints[i, :, 0], joints[i, :, 1], joints[i, :, 2], "-", color=arm_color, alpha=0.95, lw=1.1)
            # GT plane z=z_plane for plane-constraint datasets.
            if y_line is not None:
                xr = np.linspace(np.min(ee[:, 0]) - 0.3, np.max(ee[:, 0]) + 0.3, 12)
                yr = np.linspace(np.min(ee[:, 1]) - 0.3, np.max(ee[:, 1]) + 0.3, 12)
                XX, YY = np.meshgrid(xr, yr)
                ZZ = np.full_like(XX, float(y_line))
                ax2.plot_surface(XX, YY, ZZ, alpha=0.12, color="orange", linewidth=0, shade=False)
            ax2.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], c="blue", s=26)
            ax2.scatter([ee[-1, 0]], [ee[-1, 1]], [ee[-1, 2]], c="red", s=26)
            if name in ("spatial_arm_up_n6", "spatial_arm_up_n6_py") and joints.shape[1] >= 2:
                # Visualize true end-effector tool orientation.
                d = _spatial_tool_axis_n6(q_path.astype(np.float32), use_pybullet=use_pybullet_n6)
                d0 = _spatial_tool_axis_n6(q_init.astype(np.float32), use_pybullet=use_pybullet_n6)
                qstep = max(1, len(ee) // 14)
                qidx = np.arange(0, len(ee), qstep, dtype=int)
                if qidx[-1] != len(ee) - 1:
                    qidx = np.concatenate([qidx, np.array([len(ee) - 1], dtype=int)])
                scale = 0.22
                ax2.quiver(
                    ee_init[qidx, 0], ee_init[qidx, 1], ee_init[qidx, 2],
                    d0[qidx, 0], d0[qidx, 1], d0[qidx, 2],
                    length=scale, normalize=True, color="#9ca3af", linewidth=0.8, alpha=0.75
                )
                ax2.quiver(
                    ee[qidx, 0], ee[qidx, 1], ee[qidx, 2],
                    d[qidx, 0], d[qidx, 1], d[qidx, 2],
                    length=scale, normalize=True, color="#f59e0b", linewidth=1.0, alpha=0.9
                )
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("z")
            ax2.set_title(f"Workspace 3D #{row+1}", fontsize=10)
            if name in ("spatial_arm_up_n6", "spatial_arm_up_n6_py"):
                ax2.legend(loc="best", fontsize=8)

    fig.suptitle(f"{name}: planning on learned manifold (3 cases)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    if bool(cfg.show_3d_plot):
        plt.show()
    plt.close(fig)
    print(f"saved: {out_path}")
    if render_pybullet and name == "spatial_arm_up_n6" and bool(cfg.plan_pybullet_render):
        _render_ur5_pybullet_trajectories(q_paths_render, cfg)

    # Slow animation of workspace motion.
    if not bool(cfg.plan_save_gif):
        return q_paths_render
    out_gif = out_path.replace(".png", "_anim.gif")
    if is_spatial:
        fig2 = plt.figure(figsize=(10.2, 8.4))
        ax = fig2.add_subplot(111, projection="3d")
        # Tighter limits around trajectory for better visibility.
        c = np.mean(ee, axis=0)
        span = np.max(np.ptp(ee, axis=0))
        span = float(max(span, 0.35))
        half = 0.58 * span
        reach = float(sum(lengths)) + 0.15
        ax.set_xlim(float(c[0] - half), float(c[0] + half))
        ax.set_ylim(float(c[1] - half), float(c[1] + half))
        ax.set_zlim(float(c[2] - half), float(c[2] + half))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{name}: workspace motion (animation)")
        # GT plane z=z_plane for plane-constraint datasets.
        if y_line is not None:
            xr = np.linspace(-reach, reach, 12)
            yr = np.linspace(-reach, reach, 12)
            XX, YY = np.meshgrid(xr, yr)
            ZZ = np.full_like(XX, float(y_line))
            ax.plot_surface(XX, YY, ZZ, alpha=0.10, color="orange", linewidth=0, shade=False)
        ax.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], c="blue", s=38)
        ax.scatter([ee[-1, 0]], [ee[-1, 1]], [ee[-1, 2]], c="red", s=38)
        arm_line, = ax.plot([], [], [], "-", color="black", lw=2.0, alpha=0.9)
        trail, = ax.plot([], [], [], "-", color="green", lw=2.0, alpha=0.9)
        dot, = ax.plot([], [], [], "o", color="green", ms=6)
        ori_line = None
        ori = None
        ori_scale = 0.20
        if name in ("spatial_arm_up_n6", "spatial_arm_up_n6_py"):
            ori = _spatial_tool_axis_n6(q_path.astype(np.float32), use_pybullet=use_pybullet_n6)
            ori_line, = ax.plot([], [], [], "-", color="#f59e0b", lw=2.2, alpha=0.95)

        frame_idx = np.arange(0, len(joints), max(1, int(cfg.plan_anim_stride)), dtype=int)
        if frame_idx[-1] != len(joints) - 1:
            frame_idx = np.concatenate([frame_idx, np.array([len(joints) - 1], dtype=int)])

        def _init3():
            arm_line.set_data([], [])
            arm_line.set_3d_properties([])
            trail.set_data([], [])
            trail.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])
            if ori_line is not None:
                ori_line.set_data([], [])
                ori_line.set_3d_properties([])
                return arm_line, trail, dot, ori_line
            return arm_line, trail, dot

        def _update3(k):
            i = int(frame_idx[k])
            arm_line.set_data(joints[i, :, 0], joints[i, :, 1])
            arm_line.set_3d_properties(joints[i, :, 2])
            trail.set_data(ee[: i + 1, 0], ee[: i + 1, 1])
            trail.set_3d_properties(ee[: i + 1, 2])
            dot.set_data([ee[i, 0]], [ee[i, 1]])
            dot.set_3d_properties([ee[i, 2]])
            if ori_line is not None and ori is not None:
                p0 = ee[i]
                p1 = p0 + ori_scale * ori[i]
                ori_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                ori_line.set_3d_properties([p0[2], p1[2]])
                return arm_line, trail, dot, ori_line
            return arm_line, trail, dot

        ani = animation.FuncAnimation(
            fig2,
            _update3,
            init_func=_init3,
            frames=len(frame_idx),
            interval=max(1, int(round(1000.0 / max(1, int(cfg.plan_anim_fps))))),
            blit=False,
        )
        try:
            ani.save(out_gif, writer=animation.PillowWriter(fps=max(1, int(cfg.plan_anim_fps))))
            print(f"saved: {out_gif}")
        except Exception as e:
            print(f"[warn] {name}: failed to save gif: {e}")
        plt.close(fig2)
        return q_paths_render

    fig2, ax = plt.subplots(figsize=(6.2, 6.2))
    reach = float(sum(lengths)) + 0.15
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{name}: workspace motion (animation)")
    ax.scatter([ee[0, 0]], [ee[0, 1]], c="blue", s=42, label="start ee")
    ax.scatter([ee[-1, 0]], [ee[-1, 1]], c="red", s=42, label="goal ee")
    arm_line, = ax.plot([], [], "-", color="black", lw=2.0, alpha=0.9, label="arm")
    trail, = ax.plot([], [], "-", color="green", lw=2.0, alpha=0.9, label="ee trail")
    dot, = ax.plot([], [], "o", color="green", ms=6)
    ax.legend(loc="best", fontsize=8)

    frame_idx = np.arange(0, len(joints), max(1, int(cfg.plan_anim_stride)), dtype=int)
    if frame_idx[-1] != len(joints) - 1:
        frame_idx = np.concatenate([frame_idx, np.array([len(joints) - 1], dtype=int)])

    def _init():
        arm_line.set_data([], [])
        trail.set_data([], [])
        dot.set_data([], [])
        return arm_line, trail, dot

    def _update(k):
        i = int(frame_idx[k])
        arm_line.set_data(joints[i, :, 0], joints[i, :, 1])
        trail.set_data(ee[: i + 1, 0], ee[: i + 1, 1])
        dot.set_data([ee[i, 0]], [ee[i, 1]])
        return arm_line, trail, dot

    ani = animation.FuncAnimation(
        fig2,
        _update,
        init_func=_init,
        frames=len(frame_idx),
        interval=max(1, int(round(1000.0 / max(1, int(cfg.plan_anim_fps))))),
        blit=True,
    )
    try:
        ani.save(out_gif, writer=animation.PillowWriter(fps=max(1, int(cfg.plan_anim_fps))))
        print(f"saved: {out_gif}")
    except Exception as e:
        print(f"[warn] {name}: failed to save gif: {e}")
    plt.close(fig2)
    return q_paths_render


def _plot_projection_value_distribution(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: DemoCfg,
    out_path: str,
    use_pybullet_n6: bool,
) -> None:
    # Same noisy samples before/after projection; compare orientation angle error distributions.
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    span = np.maximum(maxs - mins, 1e-6).astype(np.float32)
    n0 = max(180, int(cfg.n6_workspace_vis_points) * 3)
    idx = np.random.randint(0, len(x_train), size=n0)
    x0 = x_train[idx].astype(np.float32).copy()
    noise_std = float(max(cfg.eikonal_near_std_ratio, 1e-4))
    x0 = x0 + np.random.randn(*x0.shape).astype(np.float32) * (noise_std * span.reshape(1, -1))
    x0 = np.clip(x0, mins.reshape(1, -1), maxs.reshape(1, -1)).astype(np.float32)

    with torch.no_grad():
        f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
        if f_on.dim() == 1:
            f_on = f_on.unsqueeze(1)
        h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy()
    eps_stop = float(np.percentile(np.abs(h_on), cfg.zero_eps_quantile))

    traj = project_trajectory(model, x0, cfg, eps_stop=eps_stop)
    q_end = traj[-1].astype(np.float32)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    a0 = _spatial_tool_axis_n6(x0.astype(np.float32), use_pybullet=use_pybullet_n6)
    a1 = _spatial_tool_axis_n6(q_end.astype(np.float32), use_pybullet=use_pybullet_n6)
    c0 = np.clip(np.sum(a0 * up.reshape(1, 3), axis=1), -1.0, 1.0)
    c1 = np.clip(np.sum(a1 * up.reshape(1, 3), axis=1), -1.0, 1.0)
    ang0 = np.degrees(np.arccos(c0))
    ang1 = np.degrees(np.arccos(c1))

    cap = float(np.percentile(np.concatenate([ang0, ang1], axis=0), 99))
    cap = max(cap, 1.0)
    bins = np.linspace(0.0, cap, 50)
    fig = plt.figure(figsize=(9.2, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(ang0, bins=bins, color="#64748b", alpha=0.9)
    ax1.set_title("before projection")
    ax1.set_xlabel("orientation error (deg)")
    ax1.set_ylabel("count")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(ang1, bins=bins, color="#16a34a", alpha=0.9)
    ax2.set_title("after projection")
    ax2.set_xlabel("orientation error (deg)")
    ax2.grid(alpha=0.25)

    fig.suptitle("Noisy samples: orientation-angle error before/after projection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(
        f"[proj_err] before mean={float(np.mean(ang0)):.3f} deg, after mean={float(np.mean(ang1)):.3f} deg"
    )
    print(f"saved: {out_path}")


def _render_ur5_pybullet_trajectories(q_paths: list[np.ndarray], cfg: DemoCfg) -> None:
    if not q_paths:
        return
    urdf_path = str(cfg.ur5_urdf_path).strip()
    if not urdf_path:
        print("[warn] skip pybullet render: cfg.ur5_urdf_path is empty")
        return
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except Exception as e:
        print(f"[warn] skip pybullet render: {e}")
        return

    cid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=cid)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.7, cameraYaw=48.0, cameraPitch=-28.0, cameraTargetPosition=[0.0, 0.0, 0.5], physicsClientId=cid
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    floor = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.02], useFixedBase=True, physicsClientId=cid)
    p.changeVisualShape(floor, -1, rgbaColor=[0.96, 0.97, 0.99, 1.0], physicsClientId=cid)
    try:
        rid = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)
    except Exception:
        patched = _make_pybullet_friendly_urdf(urdf_path)
        rid = p.loadURDF(patched, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)

    rev = []
    nj = p.getNumJoints(rid, physicsClientId=cid)
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        if int(info[2]) == p.JOINT_REVOLUTE:
            rev.append(int(j))
    if len(rev) < 6:
        print(f"[warn] skip pybullet render: URDF has only {len(rev)} revolute joints")
        p.disconnect(physicsClientId=cid)
        return
    arm = rev[:6]
    ee_idx = int(cfg.ur5_ee_link_index) if int(cfg.ur5_ee_link_index) >= 0 else pick_default_ee_link_index(rid, arm[-1], cid)
    try:
        ee_info = p.getJointInfo(rid, ee_idx, physicsClientId=cid)
        ee_name = ee_info[12].decode("utf-8", errors="ignore")
        print(f"[pybullet] ee link index={ee_idx}, name={ee_name}")
    except Exception:
        pass
    tool_axis_name = str(cfg.ur5_tool_axis).strip().lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(tool_axis_name, 2)
    grasp_offset = float(cfg.ur5_grasp_offset)
    grasp_axis_shift = float(cfg.ur5_grasp_axis_shift)
    debug_axes = bool(cfg.ur5_debug_axes)
    cyl_rotate_90 = bool(cfg.ur5_cylinder_rotate_90)

    cyl_len = 0.22
    cyl_rad = 0.03
    cyl_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(cyl_rad),
        length=float(cyl_len),
        rgbaColor=[0.95, 0.75, 0.15, 0.9],
        physicsClientId=cid,
    )
    cyl_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=int(cyl_vis),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=cid,
    )
    top_line_id = -1
    # Try to anchor object by actual fingertip geometry (3-finger gripper).
    name_to_idx: dict[str, int] = {}
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        lname = info[12].decode("utf-8", errors="ignore")
        name_to_idx[lname] = int(j)
    tip_names = [
        "gripperfinger_1_link_3",
        "gripperfinger_2_link_3",
        "gripperfinger_middle_link_3",
    ]
    tip_idx = [name_to_idx[nm] for nm in tip_names if nm in name_to_idx]
    gripper_joint_idx = None
    gripper_joint_limits = (-1.0, 1.0)
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        jname = info[1].decode("utf-8", errors="ignore")
        if "gripperrobotiq_hand_joint" in jname:
            gripper_joint_idx = int(j)
            lo = float(info[8])
            hi = float(info[9])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                gripper_joint_limits = (lo, hi)
            break
    if debug_axes:
        print("[pybullet] link indices (subset):")
        for nm in ("base_link", "wrist_3_link", "tool0", "ee_link", "gripperpalm",
                   "gripperfinger_1_link_3", "gripperfinger_2_link_3", "gripperfinger_middle_link_3"):
            if nm in name_to_idx:
                print(f"  - {nm}: {name_to_idx[nm]}")
        print(f"[pybullet] using ee_idx={ee_idx}, tool_axis={tool_axis_name}")

    def _axis_vec_from_quat(quat_xyzw: list[float]) -> np.ndarray:
        mat = np.asarray(p.getMatrixFromQuaternion(quat_xyzw), dtype=np.float32).reshape(3, 3)
        v = mat[:, int(axis_idx)]
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _orient_cylinder_quat(ee_quat: list[float]) -> list[float]:
        # PyBullet cylinder local axis is +Z. Rotate it to requested tool axis, then apply ee orientation.
        if axis_idx == 2:  # z
            q_off = [0.0, 0.0, 0.0, 1.0]
        elif axis_idx == 0:  # x
            q_off = p.getQuaternionFromEuler([0.0, float(np.pi / 2.0), 0.0])
        else:  # y
            q_off = p.getQuaternionFromEuler([float(-np.pi / 2.0), 0.0, 0.0])
        _, q = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat, [0.0, 0.0, 0.0], q_off)
        return list(q)

    def _quat_from_axis(axis: np.ndarray, ref_axis: np.ndarray) -> list[float]:
        # Build quaternion whose local +Z aligns with axis, and x is stabilized by ref_axis.
        z = axis / max(float(np.linalg.norm(axis)), 1e-8)
        x0 = ref_axis - np.dot(ref_axis, z) * z
        if np.linalg.norm(x0) < 1e-8:
            x0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x0 = x0 - np.dot(x0, z) * z
        x = x0 / max(float(np.linalg.norm(x0)), 1e-8)
        y = np.cross(z, x)
        y = y / max(float(np.linalg.norm(y)), 1e-8)
        x = np.cross(y, z)
        R = np.stack([x, y, z], axis=1).astype(np.float32)
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return [float(qx), float(qy), float(qz), float(qw)]

    def _apply_cylinder_extra_rotation(quat_xyzw: np.ndarray) -> np.ndarray:
        if not cyl_rotate_90:
            return quat_xyzw
        # Rotate cylinder by +90 deg around its local X so gripper tends to contact curved side.
        q_extra = p.getQuaternionFromEuler([float(np.pi / 2.0), 0.0, 0.0])
        _, q_new = p.multiplyTransforms([0.0, 0.0, 0.0], quat_xyzw.tolist(), [0.0, 0.0, 0.0], q_extra)
        return np.asarray(q_new, dtype=np.float32)

    def _cyl_axis_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
        mat = np.asarray(p.getMatrixFromQuaternion(quat_xyzw.tolist()), dtype=np.float32).reshape(3, 3)
        v = mat[:, 2]  # cylinder local Z
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _tip_center() -> np.ndarray | None:
        if len(tip_idx) < 3:
            return None
        pts = []
        for li in tip_idx[:3]:
            ls = p.getLinkState(rid, li, computeForwardKinematics=True, physicsClientId=cid)
            pts.append(np.asarray(ls[4], dtype=np.float32))
        return (pts[0] + pts[1] + pts[2]) / 3.0

    def _cylinder_pose_from_gripper(ee_pos: np.ndarray, ee_quat: list[float]) -> tuple[np.ndarray, np.ndarray]:
        # Preferred: infer from fingertip geometry.
        tool = _axis_vec_from_quat(ee_quat)
        if len(tip_idx) >= 3:
            pts = []
            for li in tip_idx[:3]:
                ls = p.getLinkState(rid, li, computeForwardKinematics=True, physicsClientId=cid)
                pts.append(np.asarray(ls[4], dtype=np.float32))
            center = (pts[0] + pts[1] + pts[2]) / 3.0
            center = center + tool * float(grasp_axis_shift)
            q = np.asarray(_quat_from_axis(tool, tool), dtype=np.float32)
            q = _apply_cylinder_extra_rotation(q)
            return center, q
        # Fallback: ee frame + offset.
        center = ee_pos + tool * float(grasp_offset)
        q = np.asarray(_quat_from_axis(tool, tool), dtype=np.float32)
        q = _apply_cylinder_extra_rotation(q)
        return center, q

    dt = float(max(cfg.plan_pybullet_real_time_dt, 0.01))
    try:
        # Initialize to first frame before enabling rendering to avoid the startup "disconnected" flash.
        q0 = q_paths[0][0]
        for k, j in enumerate(arm):
            p.resetJointState(rid, j, float(q0[k]), targetVelocity=0.0, physicsClientId=cid)
        if gripper_joint_idx is not None:
            lo, hi = gripper_joint_limits
            r = float(np.clip(cfg.ur5_gripper_close_ratio, 0.0, 1.0))
            qg = lo + r * (hi - lo)
            p.resetJointState(rid, gripper_joint_idx, qg, targetVelocity=0.0, physicsClientId=cid)
        ls0 = p.getLinkState(rid, ee_idx, computeForwardKinematics=True, physicsClientId=cid)
        ee_pos0 = np.asarray(ls0[4], dtype=np.float32)
        ee_quat0 = list(ls0[5])
        cyl_pos0, cyl_quat0 = _cylinder_pose_from_gripper(ee_pos0, ee_quat0)
        # Fix a rigid grasp offset in ee frame to avoid tiny frame-to-frame jitter.
        R0 = np.asarray(p.getMatrixFromQuaternion(ee_quat0), dtype=np.float32).reshape(3, 3)
        tip0 = _tip_center()
        if tip0 is not None:
            local_off = (R0.T @ (tip0 - ee_pos0)).astype(np.float32)
            local_off = local_off + np.array([0.0, 0.0, float(grasp_axis_shift)], dtype=np.float32)
        else:
            local_off = np.array([0.0, 0.0, float(grasp_offset)], dtype=np.float32)
        q_ref = np.asarray(_orient_cylinder_quat(ee_quat0), dtype=np.float32)
        q_ref = _apply_cylinder_extra_rotation(q_ref)
        _, ee_inv = p.invertTransform([0.0, 0.0, 0.0], ee_quat0)
        _, q_offset = p.multiplyTransforms([0.0, 0.0, 0.0], ee_inv, [0.0, 0.0, 0.0], q_ref.tolist())
        q_offset = np.asarray(q_offset, dtype=np.float32)
        cyl_pos0 = ee_pos0 + (R0 @ local_off).astype(np.float32)
        _, q0w = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat0, [0.0, 0.0, 0.0], q_offset.tolist())
        p.resetBasePositionAndOrientation(cyl_id, cyl_pos0.tolist(), list(q0w), physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=cid)
        for cidx, q_path in enumerate(q_paths):
            print(f"[pybullet] rendering case {cidx + 1}/{len(q_paths)} ...")
            for i in range(q_path.shape[0]):
                q = q_path[i]
                for k, j in enumerate(arm):
                    p.resetJointState(rid, j, float(q[k]), targetVelocity=0.0, physicsClientId=cid)
                if gripper_joint_idx is not None:
                    lo, hi = gripper_joint_limits
                    r = float(np.clip(cfg.ur5_gripper_close_ratio, 0.0, 1.0))
                    qg = lo + r * (hi - lo)
                    p.resetJointState(rid, gripper_joint_idx, qg, targetVelocity=0.0, physicsClientId=cid)
                ls = p.getLinkState(rid, ee_idx, computeForwardKinematics=True, physicsClientId=cid)
                ee_pos = np.asarray(ls[4], dtype=np.float32)
                ee_quat = list(ls[5])  # xyzw
                R = np.asarray(p.getMatrixFromQuaternion(ee_quat), dtype=np.float32).reshape(3, 3)
                cyl_pos = ee_pos + (R @ local_off).astype(np.float32)
                _, q_cur = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat, [0.0, 0.0, 0.0], q_offset.tolist())
                cyl_quat = np.asarray(q_cur, dtype=np.float32)
                axis = _cyl_axis_from_quat(cyl_quat)
                p.resetBasePositionAndOrientation(cyl_id, cyl_pos.tolist(), cyl_quat.tolist(), physicsClientId=cid)

                p0 = cyl_pos + axis * (0.5 * cyl_len)
                p1 = cyl_pos + axis * (0.5 * cyl_len + 0.18)
                top_line_id = p.addUserDebugLine(
                    p0.tolist(),
                    p1.tolist(),
                    [1.0, 0.1, 0.1],
                    lineWidth=4.0,
                    lifeTime=0.0,
                    replaceItemUniqueId=int(top_line_id),
                    physicsClientId=cid,
                )
                if debug_axes:
                    mat = np.asarray(p.getMatrixFromQuaternion(ee_quat), dtype=np.float32).reshape(3, 3)
                    o = ee_pos
                    lx = o + 0.10 * mat[:, 0]
                    ly = o + 0.10 * mat[:, 1]
                    lz = o + 0.10 * mat[:, 2]
                    p.addUserDebugLine(o.tolist(), lx.tolist(), [1, 0, 0], 2.0, 1e-3, physicsClientId=cid)
                    p.addUserDebugLine(o.tolist(), ly.tolist(), [0, 1, 0], 2.0, 1e-3, physicsClientId=cid)
                    p.addUserDebugLine(o.tolist(), lz.tolist(), [0, 0, 1], 2.0, 1e-3, physicsClientId=cid)
                time.sleep(dt)
            time.sleep(0.6)
    finally:
        p.disconnect(physicsClientId=cid)


def run_dataset(name: str, cfg: DemoCfg, outdir: str) -> None:
    set_seed(cfg.seed)
    _set_ur5_runtime_from_cfg(cfg)
    print(f"[run] start dataset={name}")
    ds = _resolve_dataset(name, cfg)
    x_train = ds["x_train"]
    data_dim = int(ds["data_dim"])
    axis_labels = ds["axis_labels"]
    print(f"[run] dataset ready: data_dim={data_dim}, n_train={len(x_train)}")

    codim_override = ds.get("constraint_dim_override", None)
    codim = int(codim_override) if codim_override is not None else (1 if data_dim == 2 else max(1, int(cfg.constraint_dim)))
    if data_dim == 2 and int(cfg.constraint_dim) != 1:
        print(f"[info] {name}: forcing constraint_dim=1 for 2D data")

    model, train_hist = train_on_eikonal_only(cfg, x_train, constraint_dim=codim)
    print(f"[run] training finished: dataset={name}")
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{name}_on_eikonal_model.pt")
    torch.save(
        {
            "dataset": str(name),
            "model_state": model.state_dict(),
            "in_dim": int(x_train.shape[1]),
            "constraint_dim": int(codim),
            "hidden": int(cfg.hidden),
            "depth": int(cfg.depth),
            "x_train": x_train.astype(np.float32),
            "cfg": asdict(cfg),
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")

    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x0 = _uniform_in_box(cfg.n_traj, mins, maxs)

    with torch.no_grad():
        f_on_t = model(torch.from_numpy(x_train).to(cfg.device))
        if f_on_t.dim() == 1:
            f_on_t = f_on_t.unsqueeze(1)
        f_on_norm = torch.linalg.norm(f_on_t, dim=1).detach().cpu().numpy().reshape(-1)
    eps_stop = float(np.percentile(np.abs(f_on_norm), cfg.zero_eps_quantile))
    traj = project_trajectory(model, x0, cfg, eps_stop=eps_stop)

    eval_metrics = evaluate_bidirectional_chamfer(model, name, x_train, cfg)
    out_eval = os.path.join(outdir, f"{name}_on_eikonal_eval.json")
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    print(
        f"[eval] {name} | chamfer={eval_metrics['bidirectional_chamfer']:.6f} "
        f"| gt->learned={eval_metrics['gt_to_learned_mean']:.6f} "
        f"| learned->gt={eval_metrics['learned_to_gt_mean']:.6f} "
        f"| space={eval_metrics['dist_space']}"
    )
    print(f"saved: {out_eval}")

    out_diag = os.path.join(outdir, f"{name}_on_eikonal_training_diag.png")
    _plot_training_diagnostics(train_hist, out_diag, title=f"{name}: training diagnostics (on-data)")
    print(f"saved: {out_diag}")

    planned_paths: list[np.ndarray] = []

    if data_dim == 2:
        out_path = os.path.join(outdir, f"{name}_on_eikonal_contour_traj.png")
        _plot_constraint_2d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{name}: on-loss + eikonal",
            axis_labels=(axis_labels[0], axis_labels[1]),
            cfg=cfg,
        )
        print(f"saved: {out_path}")
        if name == "planar_arm_line_n2":
            out_plan = os.path.join(outdir, f"{name}_on_eikonal_planning_demo.png")
            planned_paths = _plot_planar_arm_planning(model, name, x_train, out_plan, cfg, render_pybullet=False)
        return

    if data_dim == 3:
        out_path = os.path.join(outdir, f"{name}_on_eikonal_zero_surfaces_3d.png")
        _plot_zero_surfaces_3d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{name}: on-loss + eikonal",
            axis_labels=(axis_labels[0], axis_labels[1], axis_labels[2]),
            cfg=cfg,
        )
        print(f"saved: {out_path}")
    else:
        out_path = os.path.join(outdir, f"{name}_on_eikonal_pca_traj.png")
        _plot_highdim_pca(
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{name}: on-loss + eikonal",
        )
        print(f"saved: {out_path}")
    if name in ("planar_arm_line_n3", "spatial_arm_plane_n3", "spatial_arm_plane_n4", "spatial_arm_up_n6", "spatial_arm_up_n6_py"):
        out_plan = os.path.join(outdir, f"{name}_on_eikonal_planning_demo.png")
        planned_paths = _plot_planar_arm_planning(model, name, x_train, out_plan, cfg, render_pybullet=False)
    if name in ("spatial_arm_up_n6", "spatial_arm_up_n6_py"):
        out_dist = os.path.join(outdir, f"{name}_on_eikonal_proj_value_distribution.png")
        _plot_projection_value_distribution(
            model,
            x_train,
            cfg,
            out_dist,
            use_pybullet_n6=(name == "spatial_arm_up_n6"),
        )
    # Keep pybullet render as the very last step after all figures are saved.
    if name == "spatial_arm_up_n6" and bool(cfg.plan_pybullet_render) and len(planned_paths) > 0:
        _render_ur5_pybullet_trajectories(planned_paths, cfg)


def main() -> None:
    interactive_checked = False
    interactive_ok = False
    for name in DEFAULT_DATASETS:
        cfg = build_cfg(str(name), profile=ACTIVE_PROFILE)
        cfg.device = _choose_device(str(cfg.device))
        if bool(cfg.show_3d_plot):
            if not interactive_checked:
                interactive_ok = _enable_interactive_backend_if_possible()
                interactive_checked = True
            if not interactive_ok:
                print("[warn] interactive matplotlib backend unavailable; disable show_3d_plot")
                cfg.show_3d_plot = False
        if cfg.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        print(f"[cfg] dataset={name}, profile={ACTIVE_PROFILE}, n_train={cfg.n_train}, epochs={cfg.epochs}")
        run_dataset(str(name), cfg, DEFAULT_OUTDIR)


if __name__ == "__main__":
    main()
