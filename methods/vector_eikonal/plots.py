from __future__ import annotations

import math
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import nn

from evaluator.evaluator import DEFAULT_EVAL_CFG, eval_bounds_from_train
from datasets.constraint_datasets import generate_dataset
from datasets.ur5_pybullet_utils import (
    UR5_LINK_LENGTHS,
    _make_pybullet_friendly_urdf,
    pick_default_ee_link_index,
    resolve_ur5_kinematics_cfg,
    resolve_ur5_render_cfg,
)
from core.kinematics import (
    is_arm_dataset,
    planar_fk,
    spatial_fk,
    spatial_tool_axis_n6,
)
from core.planner import (
    init_path_joint_spline,
    init_path_via_workspace_ik,
    pick_far_pair_workspace_planar,
    plan_path,
)
from core.projection import project_trajectory_numpy
from common.plot_common import plot_contour_traj_2d

N6_WORKSPACE_VIS_POINTS_DEFAULT = 90
ZERO_EPS_QUANTILE_DEFAULT = float(DEFAULT_EVAL_CFG["zero_eps_quantile"])

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

def _plot_constraint_2d(
    model: nn.Module,
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str],
    cfg: Any,
) -> None:
    plot_contour_traj_2d(
        model=model,
        x_train=x_train,
        traj=traj,
        out_path=out_path,
        title=title,
        axis_labels=axis_labels,
        cfg=cfg,
        line_color="green",
    )


def _plot_zero_surfaces_3d(
    model: nn.Module,
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str, str],
    cfg: Any,
    intersection_points: np.ndarray | None = None,
) -> None:
    # Reuse evaluator bounds so near-constant axes (e.g., 3d_0z_*) also get a usable visual span.
    mins, maxs = eval_bounds_from_train(x_train, cfg)

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
        eps_q = float(getattr(cfg, "zero_eps_quantile", ZERO_EPS_QUANTILE_DEFAULT))
        eps1 = max(float(np.percentile(np.abs(f_train[:, 0].detach().cpu().numpy()), eps_q)), 1e-4)
        if has_f2:
            eps2 = max(float(np.percentile(np.abs(f_train[:, 1].detach().cpu().numpy()), eps_q)), 1e-4)
            h_on = torch.linalg.norm(f_train[:, :2], dim=1).detach().cpu().numpy()
            eps_h = max(float(np.percentile(h_on, eps_q)), 1e-6)

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
    if has_f2 and f2 is not None:
        if intersection_points is not None and len(intersection_points) > 0:
            cand = intersection_points.astype(np.float32)
            with torch.no_grad():
                fc = model(torch.from_numpy(cand).to(device))
                if fc.dim() == 1:
                    fc = fc.unsqueeze(1)
                hc = torch.linalg.norm(fc[:, :2], dim=1).detach().cpu().numpy()
            p12 = cand[hc <= eps_h]
        else:
            h_grid = np.sqrt(f1 * f1 + f2 * f2)
            p12 = pts[h_grid <= eps_h]
        if len(p12) > 1200:
            idx = np.random.choice(len(p12), size=1200, replace=False)
            p12 = p12[idx]
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
            if p12 is not None and len(p12) > 0:
                ax3.scatter(
                    p12[:, 0], p12[:, 1], p12[:, 2],
                    s=2.0, c="red", alpha=0.9,
                    label=f"intersection (q{eps_q:.0f})",
                )
        else:
            if p1 is not None and len(p1) > 0:
                ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.2, c="#22d3ee", alpha=0.26, label=f"f1≈0 (q{eps_q:.0f}={eps1:.3g})")
                ax3.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.0, c="#22d3ee", alpha=0.18, label=f"f1≈0 (q{eps_q:.0f}={eps1:.3g})")
            if p2 is not None and len(p2) > 0:
                ax2.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=1.2, c="#f472b6", alpha=0.26, label=f"f2≈0 (q{eps_q:.0f}={eps2:.3g})")
                ax3.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=1.0, c="#f472b6", alpha=0.18, label=f"f2≈0 (q{eps_q:.0f}={eps2:.3g})")
            if p12 is not None and len(p12) > 0:
                ax3.scatter(
                    p12[:, 0], p12[:, 1], p12[:, 2],
                    s=2.0, c="red", alpha=0.9,
                    label=f"intersection (q{eps_q:.0f})",
                )
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
            ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.5, c="#ef4444", alpha=0.30, label=f"f1≈0 (q{eps_q:.0f}={eps1:.3g})")
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


def _rpy_zyx_to_local_z(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    # R = Rz(yaw) Ry(pitch) Rx(roll), local z-axis in world is R[:,2].
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    z_x = cy * sp * cr + sy * sr
    z_y = sy * sp * cr - cy * sr
    z_z = cp * cr
    z = np.stack([z_x, z_y, z_z], axis=1).astype(np.float32)
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z


def _plot_workspace_pose_orientation_3d(
    x_train: np.ndarray,
    eval_proj: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    # eval_proj expected shape (N,6): [x,y,z,roll,pitch,yaw]
    if eval_proj is None or len(eval_proj) == 0 or eval_proj.shape[1] < 6:
        return
    pts = eval_proj[:, :3].astype(np.float32)
    rpy = eval_proj[:, 3:6].astype(np.float32)
    dirs = _rpy_zyx_to_local_z(rpy[:, 0], rpy[:, 1], rpy[:, 2])

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    # Background wave surface for visual inspection.
    xx, yy = np.meshgrid(np.linspace(-2.0, 2.0, 80), np.linspace(-2.0, 2.0, 80))
    a1, a2 = 0.55, 0.35
    fx, fy = 1.2, 1.0
    zz = a1 * np.sin(fx * xx) + a2 * np.cos(fy * yy)
    ax.plot_surface(xx, yy, zz, rstride=2, cstride=2, alpha=0.18, linewidth=0.0, color="#22d3ee")

    train_plot = x_train[:, :3] if x_train.shape[1] >= 3 else x_train
    if len(train_plot) > 2500:
        idx = np.random.choice(len(train_plot), size=2500, replace=False)
        train_plot = train_plot[idx]
    ax.scatter(train_plot[:, 0], train_plot[:, 1], train_plot[:, 2], s=4, c="gray", alpha=0.20, label="train")

    m = min(len(pts), 500)
    if len(pts) > m:
        idx = np.random.choice(len(pts), size=m, replace=False)
        pts_q = pts[idx]
        dirs_q = dirs[idx]
    else:
        pts_q = pts
        dirs_q = dirs
    ax.scatter(pts_q[:, 0], pts_q[:, 1], pts_q[:, 2], s=8, c="#ef4444", alpha=0.75, label="eval proj")

    ax.quiver(
        pts_q[:, 0],
        pts_q[:, 1],
        pts_q[:, 2],
        dirs_q[:, 0],
        dirs_q[:, 1],
        dirs_q[:, 2],
        length=0.20,
        normalize=True,
        color="#111827",
        linewidths=0.7,
        alpha=0.75,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    z_all = np.concatenate([pts[:, 2], zz.reshape(-1)], axis=0)
    pad = 0.2
    ax.set_zlim(float(np.min(z_all) - pad), float(np.max(z_all) + pad))
    ax.view_init(elev=24, azim=-52)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _workspace_surface_z_and_normal_from_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Wave surface used by 6d_workspace_sine_surface_pose.
    a1, a2 = 0.55, 0.35
    fx, fy = 1.2, 1.0
    z = (a1 * np.sin(fx * x) + a2 * np.cos(fy * y)).astype(np.float32)
    dzdx = a1 * fx * np.cos(fx * x)
    dzdy = -a2 * fy * np.sin(fy * y)
    n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=1).astype(np.float32)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return z, n


def _plot_workspace_pose_projection_error_distributions(
    x_before: np.ndarray,
    x_after: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    # Inputs expected shape (N,6): [x,y,z,roll,pitch,yaw].
    if (
        x_before is None
        or x_after is None
        or len(x_before) == 0
        or len(x_after) == 0
        or x_before.shape[1] < 6
        or x_after.shape[1] < 6
    ):
        return

    n = int(min(len(x_before), len(x_after)))
    xb = x_before[:n].astype(np.float32, copy=False)
    xa = x_after[:n].astype(np.float32, copy=False)
    finite = np.isfinite(xb).all(axis=1) & np.isfinite(xa).all(axis=1)
    if not np.any(finite):
        return
    xb = xb[finite]
    xa = xa[finite]

    zb_true, nb_true = _workspace_surface_z_and_normal_from_xy(xb[:, 0], xb[:, 1])
    za_true, na_true = _workspace_surface_z_and_normal_from_xy(xa[:, 0], xa[:, 1])
    pos_err_before = np.abs(xb[:, 2] - zb_true)
    pos_err_after = np.abs(xa[:, 2] - za_true)

    zb_axis = _rpy_zyx_to_local_z(xb[:, 3], xb[:, 4], xb[:, 5])
    za_axis = _rpy_zyx_to_local_z(xa[:, 3], xa[:, 4], xa[:, 5])
    cos_b = np.clip(np.sum(zb_axis * nb_true, axis=1), -1.0, 1.0)
    cos_a = np.clip(np.sum(za_axis * na_true, axis=1), -1.0, 1.0)
    ang_err_before = np.degrees(np.arccos(cos_b))
    ang_err_after = np.degrees(np.arccos(cos_a))

    pos_cap = float(np.percentile(np.concatenate([pos_err_before, pos_err_after], axis=0), 99))
    pos_cap = max(pos_cap, 1e-4)
    pos_bins = np.linspace(0.0, pos_cap, 50)

    ang_cap = float(np.percentile(np.concatenate([ang_err_before, ang_err_after], axis=0), 99))
    ang_cap = max(ang_cap, 1.0)
    ang_bins = np.linspace(0.0, ang_cap, 50)

    fig = plt.figure(figsize=(10.0, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(pos_err_before, bins=pos_bins, color="#64748b", alpha=0.72, label="before")
    ax1.hist(pos_err_after, bins=pos_bins, color="#16a34a", alpha=0.58, label="after")
    ax1.set_xlabel("position distance to true manifold")
    ax1.set_ylabel("count")
    ax1.set_title("|z - z_true(x,y)|")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(ang_err_before, bins=ang_bins, color="#64748b", alpha=0.72, label="before")
    ax2.hist(ang_err_after, bins=ang_bins, color="#16a34a", alpha=0.58, label="after")
    ax2.set_xlabel("orientation angle error (deg)")
    ax2.set_title("angle(local z-axis, true normal)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(
        "[workspace_pose_err] "
        f"pos_mean: {float(np.mean(pos_err_before)):.5f}->{float(np.mean(pos_err_after)):.5f}, "
        f"ang_mean_deg: {float(np.mean(ang_err_before)):.3f}->{float(np.mean(ang_err_after)):.3f}"
    )
    print(f"saved: {out_path}")


def _plot_ur5_eval_projection_workspace_orientation_3d(
    q_train: np.ndarray,
    q_eval_proj: np.ndarray,
    out_path: str,
    title: str,
    use_pybullet_n6: bool,
) -> None:
    if q_eval_proj is None or len(q_eval_proj) == 0 or q_eval_proj.shape[1] < 6:
        return

    def _ee_and_dir(q: np.ndarray, use_pybullet: bool) -> tuple[np.ndarray, np.ndarray]:
        joints = spatial_fk(q.astype(np.float32), list(UR5_LINK_LENGTHS), use_pybullet_n6=use_pybullet)
        ee = joints[:, -1, :].astype(np.float32)
        d = spatial_tool_axis_n6(q.astype(np.float32), use_pybullet=use_pybullet).astype(np.float32)
        return ee, d

    qtr = q_train.astype(np.float32)
    qpr = q_eval_proj.astype(np.float32)

    try:
        tr_pos, tr_dir = _ee_and_dir(qtr, use_pybullet=bool(use_pybullet_n6))
        pr_pos, pr_dir = _ee_and_dir(qpr, use_pybullet=bool(use_pybullet_n6))
    except Exception as e:
        # Keep visualization available even when pybullet backend is unavailable in headless env.
        print(f"[warn] UR5 workspace vis fallback to analytic kinematics: {e}")
        tr_pos, tr_dir = _ee_and_dir(qtr, use_pybullet=False)
        pr_pos, pr_dir = _ee_and_dir(qpr, use_pybullet=False)

    if len(tr_pos) > 1400:
        idx = np.random.choice(len(tr_pos), size=1400, replace=False)
        tr_pos = tr_pos[idx]
        tr_dir = tr_dir[idx]
    if len(pr_pos) > 900:
        idx = np.random.choice(len(pr_pos), size=900, replace=False)
        pr_pos = pr_pos[idx]
        pr_dir = pr_dir[idx]

    fig = plt.figure(figsize=(10.4, 4.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.scatter(tr_pos[:, 0], tr_pos[:, 1], tr_pos[:, 2], s=4, c="#9ca3af", alpha=0.20, label="train workspace")
    ax1.scatter(pr_pos[:, 0], pr_pos[:, 1], pr_pos[:, 2], s=8, c="#ef4444", alpha=0.72, label="eval proj workspace")
    qstep = max(1, len(pr_pos) // 90)
    qidx = np.arange(0, len(pr_pos), qstep, dtype=int)
    if len(qidx) > 0 and qidx[-1] != len(pr_pos) - 1:
        qidx = np.concatenate([qidx, np.array([len(pr_pos) - 1], dtype=int)])
    if len(qidx) > 0:
        ax1.quiver(
            pr_pos[qidx, 0], pr_pos[qidx, 1], pr_pos[qidx, 2],
            pr_dir[qidx, 0], pr_dir[qidx, 1], pr_dir[qidx, 2],
            length=0.18, normalize=True, color="#111827", linewidths=0.7, alpha=0.75
        )
    all_pos = np.concatenate([tr_pos, pr_pos], axis=0)
    mins = np.min(all_pos, axis=0)
    maxs = np.max(all_pos, axis=0)
    ctr = 0.5 * (mins + maxs)
    half = 0.55 * float(max(np.max(maxs - mins), 1e-3))
    ax1.set_xlim(float(ctr[0] - half), float(ctr[0] + half))
    ax1.set_ylim(float(ctr[1] - half), float(ctr[1] + half))
    ax1.set_zlim(float(ctr[2] - half), float(ctr[2] + half))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Workspace Position + Tool Axis")
    ax1.legend(loc="best", fontsize=8)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ax2.scatter(tr_dir[:, 0], tr_dir[:, 1], tr_dir[:, 2], s=5, c="#9ca3af", alpha=0.22, label="train tool axis")
    ax2.scatter(pr_dir[:, 0], pr_dir[:, 1], pr_dir[:, 2], s=8, c="#ef4444", alpha=0.75, label="eval proj tool axis")
    ax2.quiver([0.0], [0.0], [0.0], [up[0]], [up[1]], [up[2]], color="#16a34a", linewidths=2.0, length=1.0)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_zlim(-1.05, 1.05)
    ax2.set_xlabel("dx")
    ax2.set_ylabel("dy")
    ax2.set_zlabel("dz")
    ax2.set_title("Tool Axis on Unit Sphere")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_planar_arm_planning(
    model: nn.Module,
    name: str,
    x_train: np.ndarray,
    out_path: str,
    cfg: Any,
    render_pybullet: bool = True,
) -> list[np.ndarray]:
    if name == "2d_planar_arm_line_n2":
        lengths = [1.0, 0.8]
        y_line = 0.3
        is_spatial = False
        use_pybullet_n6 = False
    elif name == "3d_planar_arm_line_n3":
        lengths = [1.0, 0.8, 0.6]
        y_line = 0.35
        is_spatial = False
        use_pybullet_n6 = False
    elif name == "3d_spatial_arm_plane_n3":
        lengths = [1.0, 0.8]
        y_line = 0.35  # here means z-plane value
        is_spatial = True
        use_pybullet_n6 = False
    elif name == "3d_spatial_arm_circle_n3":
        lengths = [1.0, 0.8]
        y_line = None
        is_spatial = True
        use_pybullet_n6 = False
    elif name == "6d_spatial_arm_up_n6":
        lengths = list(UR5_LINK_LENGTHS)
        y_line = None
        is_spatial = True
        use_pybullet_n6 = True
    elif name == "6d_spatial_arm_up_n6_py":
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
        ee_c = spatial_fk(q_c, lengths, use_pybullet_n6=use_pybullet_n6)[:, -1, :]  # (N,3)
        target = 0.5 * (lo + hi)
    else:
        q_c = None
        ee_c = None
        target = 0.5 * (lo + hi)
    for _ in range(3):
        if not is_spatial:
            q_start, q_goal, ee_dist, _ = pick_far_pair_workspace_planar(
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
            eps_q = float(getattr(cfg, "zero_eps_quantile", ZERO_EPS_QUANTILE_DEFAULT))
            eps0 = float(np.percentile(np.abs(f_on[:, 0].detach().cpu().numpy()), eps_q))
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
            q_lin = init_path_via_workspace_ik(
                q_start=q_start.astype(np.float32),
                q_goal=q_goal.astype(np.float32),
                lengths=lengths,
                is_spatial=is_spatial,
                use_pybullet_n6=use_pybullet_n6,
                n_waypoints=140,
                device=str(cfg.device),
            )
        else:
            q_lin = init_path_joint_spline(
                q_start=q_start.astype(np.float32),
                q_goal=q_goal.astype(np.float32),
                n_waypoints=140,
                mid_noise=float(cfg.plan_joint_mid_noise),
            )
        q_path = plan_path(
            model=model,
            x_start=q_start.astype(np.float32),
            x_goal=q_goal.astype(np.float32),
            cfg=cfg,
            planner_name=str(getattr(cfg, "plan_method", "trajectory_opt")),
            n_waypoints=int(q_lin.shape[0]),
            dataset_name=str(name),
            periodic_joint=bool(is_arm_dataset(name)),
            init_path=q_lin.astype(np.float32),
        )
        q_paths_render.append(q_path.astype(np.float32))
        q_init = q_lin.astype(np.float32)
        if not is_spatial:
            joints = planar_fk(q_path, lengths)
            ee = joints[:, -1, :]
        else:
            joints = spatial_fk(q_path, lengths, use_pybullet_n6=use_pybullet_n6)
            ee = joints[:, -1, :]
            joints_init = spatial_fk(q_init, lengths, use_pybullet_n6=use_pybullet_n6)
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
            if name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
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
            if name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
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
            if name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py") and joints.shape[1] >= 2:
                # Visualize true end-effector tool orientation.
                d = spatial_tool_axis_n6(q_path.astype(np.float32), use_pybullet=use_pybullet_n6)
                d0 = spatial_tool_axis_n6(q_init.astype(np.float32), use_pybullet=use_pybullet_n6)
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
            if name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
                ax2.legend(loc="best", fontsize=8)

    fig.suptitle(f"{name}: planning on learned manifold (3 cases)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    if bool(cfg.show_3d_plot):
        plt.show()
    plt.close(fig)
    print(f"saved: {out_path}")
    if render_pybullet and name == "6d_spatial_arm_up_n6" and bool(cfg.plan_pybullet_render):
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
        if name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
            ori = spatial_tool_axis_n6(q_path.astype(np.float32), use_pybullet=use_pybullet_n6)
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
    cfg: Any,
    out_path: str,
    use_pybullet_n6: bool,
) -> None:
    # Same noisy samples before/after projection; compare orientation angle error distributions.
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    span = np.maximum(maxs - mins, 1e-6).astype(np.float32)
    n0 = max(180, N6_WORKSPACE_VIS_POINTS_DEFAULT * 3)
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
    eps_q = float(getattr(cfg, "zero_eps_quantile", ZERO_EPS_QUANTILE_DEFAULT))
    eps_stop = float(np.percentile(np.abs(h_on), eps_q))

    traj = project_trajectory_numpy(
        model,
        x0,
        device=str(cfg.device),
        proj_steps=int(cfg.proj_steps),
        proj_alpha=float(cfg.proj_alpha),
        proj_min_steps=int(getattr(cfg, "proj_min_steps", 0)),
        f_abs_stop=eps_stop,
    )
    q_end = traj[-1].astype(np.float32)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    a0 = spatial_tool_axis_n6(x0.astype(np.float32), use_pybullet=use_pybullet_n6)
    a1 = spatial_tool_axis_n6(q_end.astype(np.float32), use_pybullet=use_pybullet_n6)
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


def _plot_ur5_projection_error_distribution_from_pairs(
    q_eval: np.ndarray,
    q_eval_proj: np.ndarray,
    out_path: str,
    title: str,
    use_pybullet_n6: bool,
) -> None:
    # Compare orientation angle to +z before/after projection using matched eval pairs.
    if (
        q_eval is None
        or q_eval_proj is None
        or len(q_eval) == 0
        or len(q_eval_proj) == 0
        or q_eval.shape[1] < 6
        or q_eval_proj.shape[1] < 6
    ):
        return

    n = int(min(len(q_eval), len(q_eval_proj)))
    qb = q_eval[:n].astype(np.float32, copy=False)
    qa = q_eval_proj[:n].astype(np.float32, copy=False)
    finite = np.isfinite(qb).all(axis=1) & np.isfinite(qa).all(axis=1)
    if not np.any(finite):
        return
    qb = qb[finite]
    qa = qa[finite]

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ab = spatial_tool_axis_n6(qb, use_pybullet=use_pybullet_n6)
    aa = spatial_tool_axis_n6(qa, use_pybullet=use_pybullet_n6)
    cb = np.clip(np.sum(ab * up.reshape(1, 3), axis=1), -1.0, 1.0)
    ca = np.clip(np.sum(aa * up.reshape(1, 3), axis=1), -1.0, 1.0)
    ang_before = np.degrees(np.arccos(cb))
    ang_after = np.degrees(np.arccos(ca))

    cap = float(np.percentile(np.concatenate([ang_before, ang_after], axis=0), 99))
    cap = max(cap, 1.0)
    bins = np.linspace(0.0, cap, 50)

    fig = plt.figure(figsize=(9.2, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(ang_before, bins=bins, color="#64748b", alpha=0.9)
    ax1.set_title("before projection")
    ax1.set_xlabel("orientation error (deg)")
    ax1.set_ylabel("count")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(ang_after, bins=bins, color="#16a34a", alpha=0.9)
    ax2.set_title("after projection")
    ax2.set_xlabel("orientation error (deg)")
    ax2.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(
        f"[ur5_proj_err] before mean={float(np.mean(ang_before)):.3f} deg, "
        f"after mean={float(np.mean(ang_after)):.3f} deg"
    )
    print(f"saved: {out_path}")


def _render_ur5_pybullet_trajectories(q_paths: list[np.ndarray], cfg: Any) -> None:
    if not q_paths:
        return
    kin_ov = {}
    for k, a in (("urdf_path", "ur5_urdf_path"), ("ee_link_index", "ur5_ee_link_index"), ("tool_axis", "ur5_tool_axis")):
        if hasattr(cfg, a):
            kin_ov[k] = getattr(cfg, a)
    ren_ov = {}
    for k, a in (
        ("grasp_offset", "ur5_grasp_offset"),
        ("grasp_axis_shift", "ur5_grasp_axis_shift"),
        ("debug_axes", "ur5_debug_axes"),
        ("cylinder_rotate_90", "ur5_cylinder_rotate_90"),
        ("gripper_close_ratio", "ur5_gripper_close_ratio"),
    ):
        if hasattr(cfg, a):
            ren_ov[k] = getattr(cfg, a)
    kin = resolve_ur5_kinematics_cfg(kin_ov)
    ren = resolve_ur5_render_cfg(ren_ov)

    urdf_path = str(kin["urdf_path"]).strip()
    if not urdf_path:
        print("[warn] skip pybullet render: urdf_path is empty")
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
    ee_raw = int(kin["ee_link_index"])
    ee_idx = ee_raw if ee_raw >= 0 else pick_default_ee_link_index(rid, arm[-1], cid)
    try:
        ee_info = p.getJointInfo(rid, ee_idx, physicsClientId=cid)
        ee_name = ee_info[12].decode("utf-8", errors="ignore")
        print(f"[pybullet] ee link index={ee_idx}, name={ee_name}")
    except Exception:
        pass
    tool_axis_name = str(kin["tool_axis"]).strip().lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(tool_axis_name, 2)
    grasp_offset = float(ren["grasp_offset"])
    grasp_axis_shift = float(ren["grasp_axis_shift"])
    debug_axes = bool(ren["debug_axes"])
    cyl_rotate_90 = bool(ren["cylinder_rotate_90"])

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
            r = float(np.clip(float(ren["gripper_close_ratio"]), 0.0, 1.0))
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
                    r = float(np.clip(float(ren["gripper_close_ratio"]), 0.0, 1.0))
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
