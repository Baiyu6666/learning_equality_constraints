#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import torch

from methods.vector_eikonal.vector_eikonal import (
    DEFAULT_DATASETS,
    DEFAULT_OUTDIR,
    MLP,
    _choose_device,
    _enable_interactive_backend_if_possible,
    _resolve_dataset,
    build_cfg,
)
from core.planner import _plot_planar_arm_planning

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ----------------------------------------------------------------------
# Planner-only overrides (applied on top of demo cfg/checkpoint cfg)
# Edit here to tune planning without touching training script.
# ----------------------------------------------------------------------
PLANNER_OVERRIDES = {
    # planner optimization and sampling (planner sub-config)
    "planner": {
        "opt_steps": 1500,
        "opt_lr": 0.005,
        "opt_lam_smooth": 0.1,
        "trust_scale": 1.0,
        "lam_manifold": 1.0,
        "lam_len_joint": 2.0,
        "init_mode": "joint_spline",  # "joint_spline" | "workspace_ik"
        "joint_mid_noise": 0.0,
        "pair_min_ratio": 0.15,
        "pair_max_ratio": 0.35,
        "pair_tries": 1200,
        "save_gif": True,
        "anim_fps": 6,
        "anim_stride": 1,
        "pybullet_render": True,
        "pybullet_real_time_dt": 0.06
    },
    # projection controls used during planning plot
    "projector": {"alpha": 0.3, "steps": 120, "min_steps": 30},
    # "zero_eps_quantile": 95.0,
}

# Optional dataset-specific planner override examples:
DATASET_PLANNER_OVERRIDES = {
    # "6d_spatial_arm_up_n6": {"plan_opt_lr": 0.008, "plan_lam_len_joint": 0.6},
}


def apply_planner_overrides(cfg, dataset_name: str) -> None:
    for k, v in PLANNER_OVERRIDES.items():
        if k == "planner" and hasattr(cfg, "planner") and isinstance(getattr(cfg, "planner"), dict):
            cur = dict(getattr(cfg, "planner"))
            cur.update(dict(v))
            setattr(cfg, "planner", cur)
            continue
        if k == "projector" and hasattr(cfg, "projector") and isinstance(getattr(cfg, "projector"), dict):
            cur = dict(getattr(cfg, "projector"))
            cur.update(dict(v))
            setattr(cfg, "projector", cur)
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    ds_ov = DATASET_PLANNER_OVERRIDES.get(dataset_name, {})
    for k, v in ds_ov.items():
        if k == "planner" and hasattr(cfg, "planner") and isinstance(getattr(cfg, "planner"), dict):
            cur = dict(getattr(cfg, "planner"))
            cur.update(dict(v))
            setattr(cfg, "planner", cur)
            continue
        if k == "projector" and hasattr(cfg, "projector") and isinstance(getattr(cfg, "projector"), dict):
            cur = dict(getattr(cfg, "projector"))
            cur.update(dict(v))
            setattr(cfg, "projector", cur)
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def run_planning_only(name: str) -> None:
    cfg = build_cfg(name)
    apply_planner_overrides(cfg, name)
    cfg.device = _choose_device(str(cfg.device))
    if bool(cfg.show_3d_plot):
        ok = _enable_interactive_backend_if_possible()
        if not ok:
            print("[warn] interactive matplotlib backend unavailable; disable show_3d_plot")
            cfg.show_3d_plot = False
    pln = dict(getattr(cfg, "planner", {})) if isinstance(getattr(cfg, "planner", {}), dict) else {}
    print(
        "[plan_cfg] "
        f"steps={pln.get('opt_steps')}, lr={pln.get('opt_lr')}, smooth={pln.get('opt_lam_smooth')}, "
        f"lam_man={pln.get('lam_manifold')}, lam_len={pln.get('lam_len_joint')}, trust_scale={pln.get('trust_scale')}, "
        f"init={pln.get('init_mode')}"
    )

    outdir = DEFAULT_OUTDIR if os.path.isabs(DEFAULT_OUTDIR) else os.path.join(_PROJECT_ROOT, DEFAULT_OUTDIR)
    ckpt_path = os.path.join(outdir, f"{name}_on_eikonal_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    in_dim = int(ckpt["in_dim"])
    constraint_dim = int(ckpt["constraint_dim"])
    hidden = int(ckpt["hidden"])
    depth = int(ckpt["depth"])
    model = MLP(in_dim=in_dim, hidden=hidden, depth=depth, out_dim=constraint_dim).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if "x_train" in ckpt:
        x_train = ckpt["x_train"]
    else:
        ds = _resolve_dataset(name, cfg)
        x_train = ds["x_train"]

    out_plan = os.path.join(outdir, f"{name}_on_eikonal_planning_demo_replan.png")
    _plot_planar_arm_planning(
        model,
        name,
        x_train,
        out_plan,
        cfg,
        render_pybullet=bool(pln.get("pybullet_render", False)),
    )
    print(f"saved: {out_plan}")


def main() -> None:
    for name in DEFAULT_DATASETS:
        print(f"[plan] dataset={name}")
        run_planning_only(str(name))


if __name__ == "__main__":
    main()
