#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import torch

from demo_on_eikonal_only import (
    ACTIVE_PROFILE,
    DEFAULT_DATASETS,
    DEFAULT_OUTDIR,
    MLPConstraint,
    _choose_device,
    _enable_interactive_backend_if_possible,
    _plot_planar_arm_planning,
    _resolve_dataset,
    _set_ur5_runtime_from_cfg,
    build_cfg,
)

# ----------------------------------------------------------------------
# Planner-only overrides (applied on top of demo cfg/checkpoint cfg)
# Edit here to tune planning without touching training script.
# ----------------------------------------------------------------------
PLANNER_OVERRIDES = {
    # core planner optimization
    "plan_opt_steps": 1500,
    "plan_opt_lr": 0.005,
    "plan_opt_lam_smooth": 0.1,
    "plan_trust_scale": 1,
    "plan_lam_manifold": 1.0,
    "plan_lam_len_joint": 2.0,
    # initialization and pair sampling
    "plan_init_mode": "joint_spline",  # "joint_spline" | "workspace_ik"
    "plan_joint_mid_noise": 0.0,
    "plan_pair_min_ratio": 0.15,
    "plan_pair_max_ratio": 0.35,
    "plan_pair_tries": 1200,
    # projection and visualization controls used during planning plot
    "proj_alpha": 0.3,
    "proj_steps": 120,
    "zero_eps_quantile": 90.0,
    "plan_save_gif": True,
    "plan_anim_fps": 6,
    "plan_anim_stride": 1,
    "plan_pybullet_render": True,
    "plan_pybullet_real_time_dt": 0.06,
}

# Optional dataset-specific planner override examples:
DATASET_PLANNER_OVERRIDES = {
    # "spatial_arm_up_n6": {"plan_opt_lr": 0.008, "plan_lam_len_joint": 0.6},
}


def apply_planner_overrides(cfg, dataset_name: str) -> None:
    for k, v in PLANNER_OVERRIDES.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    ds_ov = DATASET_PLANNER_OVERRIDES.get(dataset_name, {})
    for k, v in ds_ov.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def run_planning_only(name: str) -> None:
    cfg = build_cfg(name, profile=ACTIVE_PROFILE)
    apply_planner_overrides(cfg, name)
    cfg.device = _choose_device(str(cfg.device))
    if bool(cfg.show_3d_plot):
        ok = _enable_interactive_backend_if_possible()
        if not ok:
            print("[warn] interactive matplotlib backend unavailable; disable show_3d_plot")
            cfg.show_3d_plot = False
    _set_ur5_runtime_from_cfg(cfg)
    print(
        "[plan_cfg] "
        f"steps={cfg.plan_opt_steps}, lr={cfg.plan_opt_lr}, smooth={cfg.plan_opt_lam_smooth}, "
        f"lam_man={cfg.plan_lam_manifold}, lam_len={cfg.plan_lam_len_joint}, trust_scale={cfg.plan_trust_scale}, "
        f"init={cfg.plan_init_mode}"
    )

    ckpt_path = os.path.join(DEFAULT_OUTDIR, f"{name}_on_eikonal_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    in_dim = int(ckpt["in_dim"])
    constraint_dim = int(ckpt["constraint_dim"])
    hidden = int(ckpt["hidden"])
    depth = int(ckpt["depth"])
    model = MLPConstraint(in_dim=in_dim, hidden=hidden, depth=depth, out_dim=constraint_dim).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if "x_train" in ckpt:
        x_train = ckpt["x_train"]
    else:
        ds = _resolve_dataset(name, cfg)
        x_train = ds["x_train"]

    out_plan = os.path.join(DEFAULT_OUTDIR, f"{name}_on_eikonal_planning_demo_replan.png")
    _plot_planar_arm_planning(
        model,
        name,
        x_train,
        out_plan,
        cfg,
        render_pybullet=bool(cfg.plan_pybullet_render),
    )
    print(f"saved: {out_plan}")


def main() -> None:
    for name in DEFAULT_DATASETS:
        print(f"[plan] dataset={name}, profile={ACTIVE_PROFILE}")
        run_planning_only(str(name))


if __name__ == "__main__":
    main()
