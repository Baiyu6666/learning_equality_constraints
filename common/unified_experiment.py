from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from datasets.constraint_datasets import set_seed
from core.dataset_resolve import resolve_dataset
from core.eval_runner import run_eval_metrics
from core.projection import project_trajectory_numpy
from core.planner import plan_path
from core.kinematics import (
    is_arm_dataset as _is_arm_dataset,
    is_workspace_pose_dataset as _is_workspace_pose_dataset,
    workspace_embed_for_eval as _workspace_embed_for_eval,
    wrap_np_pi as _wrap_np_pi,
    wrap_workspace_pose_rpy_np as _wrap_workspace_pose_rpy_np,
)
from methods.baseline_udf import baseline_udf as udf
from methods.baseline_vae import models as vae_models
from methods.baseline_vae import train as vae_train
from methods.baseline_vae import viz as vae_viz
from methods.vector_eikonal import vector_eikonal as ve
from methods.baseline_udf.plots import (
    plot_knn_normals,
    plot_loss_curves,
    plot_planned_paths,
)
from common.plot_common import plot_contour_traj_2d
from methods.vector_eikonal.plots import (
    _plot_constraint_2d,
    _plot_zero_surfaces_3d,
    _plot_ur5_eval_projection_workspace_orientation_3d,
    _plot_ur5_projection_error_distribution_from_pairs,
    _plot_workspace_pose_orientation_3d,
    _plot_workspace_pose_projection_error_distributions,
)
from evaluator.evaluator import eval_bounds_from_train, resolve_gt_grid

from common.config_loader import apply_overrides, load_layered_config

VALID_METHODS = {"eikonal", "margin", "delta", "vae"}


@dataclass
class VAEConfig:
    seed: int = 36
    device: str = "auto"
    n_train: int = 512
    n_grid: int = 4096
    latent_dim: int = -1
    hidden_dims: tuple[int, ...] = (64, 32)
    epochs: int = 500
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta_final: float = 0.05
    warmup_epochs: int = 440
    ur5_backend: str = "analytic"
    viz_enable: bool = True
    viz_sample_latent_n: int = 150
    viz_max_eval_points: int = 2000
    projector: dict[str, Any] = field(
        default_factory=lambda: {"alpha": 0.3, "steps": 80, "min_steps": 20}
    )


class _VAEProjectorField(nn.Module):
    """Expose VAE as a residual field f(x)=x-D(E_mu(x)) for shared evaluator APIs."""

    def __init__(self, vae_model: vae_models.VAE):
        super().__init__()
        self.vae_model = vae_model

    def project_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.vae_model.encode(x)
        return self.vae_model.decode(mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.project_tensor(x)
        return x - x_hat


def _print_eval_lines(dataset: str, metrics: dict[str, Any]) -> None:
    print(
        f"[eval] {dataset} | proj_dist={float(metrics.get('proj_manifold_dist', float('nan'))):.6f} "
        f"| pred_recall={float(metrics.get('pred_recall', float('nan'))):.6f} "
        f"| pred_FPrate={float(metrics.get('pred_FPrate', float('nan'))):.6f} "
        f"| chamfer={float(metrics.get('bidirectional_chamfer', float('nan'))):.6f} "
        f"| gt->learned={float(metrics.get('gt_to_learned_mean', float('nan'))):.6f} "
        f"| learned->gt={float(metrics.get('learned_to_gt_mean', float('nan'))):.6f} "
        f"| space={metrics.get('dist_space', 'unknown')}"
    )


def _nn_dist_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    print(
        f"[eval] {dataset} | proj_steps={float(metrics.get('proj_steps', float('nan'))):.2f} "
        f"| proj_true_dist={float(metrics.get('proj_true_dist', float('nan'))):.6f} "
        f"| proj_v_residual={float(metrics.get('proj_v_residual', float('nan'))):.6f} "
        f"| eval_eps={float(metrics.get('eval_eps_used', float('nan'))):.6f} "
        f"| pred_precision={float(metrics.get('pred_precision', float('nan'))):.6f}"
    )


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _axis_labels_for_dataset(dataset: str, dim: int) -> tuple[str, ...]:
    if _is_arm_dataset(dataset):
        return tuple([f"q{i+1}" for i in range(dim)])
    return tuple([f"x{i+1}" for i in range(dim)])


def _make_vis_cfg_for_method(dataset: str, run_cfg: Any, eval_cfg: Any) -> Any:
    # Reuse eikonal plotting defaults, but respect current run/eval projection and bounds knobs.
    vis_vals = asdict(ve.build_cfg(dataset))
    if hasattr(run_cfg, "__dict__"):
        for k, v in vars(run_cfg).items():
            vis_vals[k] = v
    if hasattr(eval_cfg, "__dict__"):
        vis_vals.update(vars(eval_cfg))
    return SimpleNamespace(**vis_vals)


def _save_common_method_plots(
    *,
    dataset: str,
    method_tag: str,
    model: nn.Module,
    x_train: np.ndarray,
    outdir: str,
    vis_cfg: Any,
    eval_artifacts: dict[str, Any],
) -> None:
    dim = int(x_train.shape[1])
    labels = _axis_labels_for_dataset(dataset, dim)
    mins, maxs = eval_bounds_from_train(x_train, vis_cfg)
    rng = np.random.default_rng(int(getattr(vis_cfg, "seed", 0)) + 3001)
    n_traj = max(8, int(getattr(vis_cfg, "n_traj", 64)))
    x0 = rng.uniform(mins, maxs, size=(n_traj, dim)).astype(np.float32)

    with torch.no_grad():
        f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(str(getattr(vis_cfg, "device", "cpu"))))
        if f_on.dim() == 1:
            f_on = f_on.unsqueeze(1)
        h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy().reshape(-1)
    q = float(getattr(vis_cfg, "zero_eps_quantile", 90.0))
    eps_stop = float(np.percentile(np.abs(h_on), q))
    traj = project_trajectory_numpy(
        model,
        x0,
        device=str(getattr(vis_cfg, "device", "cpu")),
        proj_steps=int(getattr(vis_cfg, "proj_steps", 80)),
        proj_alpha=float(getattr(vis_cfg, "proj_alpha", 0.3)),
        proj_min_steps=int(getattr(vis_cfg, "proj_min_steps", 0)),
        f_abs_stop=eps_stop,
    )

    if dim == 2:
        out_path = os.path.join(outdir, f"{dataset}_{method_tag}_contour_traj.png")
        _plot_constraint_2d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{dataset}: {method_tag}",
            axis_labels=(labels[0], labels[1]),
            cfg=vis_cfg,
        )
        return
    if dim == 3:
        out_path = os.path.join(outdir, f"{dataset}_{method_tag}_zero_surfaces_3d.png")
        _plot_zero_surfaces_3d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{dataset}: {method_tag}",
            axis_labels=(labels[0], labels[1], labels[2]),
            cfg=vis_cfg,
            intersection_points=eval_artifacts.get("proj", None),
        )


def _save_udf_legacy_plots(
    *,
    method: str,
    dataset: str,
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    cfg: Any,
    vis_cfg: Any,
    eval_artifacts: dict[str, Any],
    train_history: dict[str, Any],
    outdir: str,
    ds_info: dict[str, Any],
) -> None:
    x_eval = eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)).astype(np.float32)
    if len(x_eval) == 0:
        return

    n_train = len(x_train)
    idx_list = [
        min(n_train - 1, int(round((n_train - 1) * r)))
        for r in np.linspace(0.05, 0.95, 15)
    ]
    k = udf.effective_knn_norm_estimation_points(cfg, n_train)
    plot_knn_normals(
        x_train,
        idx_list=idx_list,
        k=k,
        out_path=os.path.join(outdir, f"{dataset}_{method}_knn_normal.png"),
        title=f"{dataset}: KNN + Normal ({method})",
        cfg=vis_cfg,
        grid=grid,
        sigma_per_point=None,
    )

    n_plot = 16 if x_train.shape[1] == 3 else 128
    n_plot = min(n_plot, len(x_eval))
    x0 = x_eval[:n_plot].astype(np.float32)
    eps_used = float(eval_artifacts.get("eval_eps_used", 0.0))
    if not np.isfinite(eps_used) or eps_used <= 0.0:
        eps_used = 1e-6
    traj = project_trajectory_numpy(
        model,
        x0,
        device=str(getattr(cfg, "device", "cpu")),
        proj_steps=int(getattr(cfg, "proj_steps", 100)),
        proj_alpha=float(getattr(cfg, "proj_alpha", 0.3)),
        proj_min_steps=int(getattr(cfg, "proj_min_steps", 0)),
        f_abs_stop=eps_used,
    )
    if int(x_train.shape[1]) == 2:
        axis_labels = _axis_labels_for_dataset(dataset, int(x_train.shape[1]))
        plot_contour_traj_2d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=os.path.join(outdir, f"{dataset}_{method}_contour_traj.png"),
            title=f"{dataset}: {method.capitalize()} Baseline",
            axis_labels=(axis_labels[0], axis_labels[1]),
            cfg=vis_cfg,
            line_color="green",
        )
    plot_loss_curves(
        train_history,
        out_path=os.path.join(outdir, f"{dataset}_{method}_loss_curves.png"),
        title=f"{dataset}: {method.capitalize()} Baseline Losses",
        cfg=vis_cfg,
    )

    if method != "delta" or x_train.shape[1] != 2:
        return
    plan_rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 77)
    n_pairs = 4
    replace = len(x_eval) < 2 * n_pairs
    picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
    plan_pairs = [
        (x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]])
        for i in range(n_pairs)
    ]
    plans_proj: list[np.ndarray] = []
    plans_constr: list[np.ndarray] = []
    use_linear = str(getattr(cfg, "plan_method", "trajectory_opt")).lower() in (
        "linear_project",
        "linear_proj",
        "projection",
    )
    for x_start, x_goal in plan_pairs:
        x_start = udf.true_projection(x_start[None, :], grid)[0][0]
        x_goal = udf.true_projection(x_goal[None, :], grid)[0][0]
        planned = plan_path(
            model=model,
            x_start=x_start,
            x_goal=x_goal,
            cfg=cfg,
            planner_name=str(getattr(cfg, "plan_method", "trajectory_opt")),
            n_waypoints=int(getattr(cfg, "plan_steps", 64) + 1),
            dataset_name=dataset,
            periodic_joint=bool(ds_info.get("periodic_joint", False)),
            f_abs_stop=eps_used,
        )
        if use_linear:
            plans_proj.append(planned)
        else:
            plans_constr.append(planned)
    plot_planned_paths(
        model,
        x_train,
        grid,
        plans_proj,
        plans_constr,
        vis_cfg,
        out_path=os.path.join(outdir, f"{dataset}_{method}_planner_paths.png"),
        title=f"{dataset}: Delta Planned Paths",
        zero_level_eps=eps_used,
    )


def _pick_cfg_field_names(cfg_obj: Any) -> set[str]:
    if hasattr(cfg_obj, "__dataclass_fields__"):
        return set(getattr(cfg_obj, "__dataclass_fields__").keys())
    return set(vars(cfg_obj).keys())


def _apply_dict_to_cfg(cfg_obj: Any, mapping: dict[str, Any]) -> None:
    valid = _pick_cfg_field_names(cfg_obj)
    unknown = [k for k in mapping.keys() if k not in valid]
    if unknown:
        print(f"[warn] ignore unsupported config keys for {type(cfg_obj).__name__}: {unknown}")
    for k, v in mapping.items():
        if k not in valid:
            continue
        setattr(cfg_obj, k, v)


def _apply_projector_subcfg(cfg_obj: Any) -> None:
    proj = getattr(cfg_obj, "projector", None)
    if not isinstance(proj, dict):
        return
    if "alpha" in proj and hasattr(cfg_obj, "proj_alpha"):
        setattr(cfg_obj, "proj_alpha", float(proj["alpha"]))
    if "steps" in proj and hasattr(cfg_obj, "proj_steps"):
        setattr(cfg_obj, "proj_steps", int(proj["steps"]))
    if "min_steps" in proj and hasattr(cfg_obj, "proj_min_steps"):
        setattr(cfg_obj, "proj_min_steps", int(proj["min_steps"]))


def _resolve_run_config(
    method: str,
    dataset: str,
    *,
    config_root: str,
    cli_overrides: list[str],
) -> tuple[dict[str, Any], list[str]]:
    cfg_dict, loaded_paths = load_layered_config(config_root, method, dataset)
    cfg_dict = apply_overrides(cfg_dict, cli_overrides)
    return cfg_dict, loaded_paths


def run_eikonal_one(
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    cfg_mapping: dict[str, Any],
) -> dict[str, Any]:
    cfg = ve.build_cfg(dataset)
    _apply_dict_to_cfg(cfg, cfg_mapping)
    _apply_projector_subcfg(cfg)

    if seed_override is not None:
        cfg.seed = int(seed_override)

    cfg.device = ve._choose_device(str(cfg.device))
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_seed(int(cfg.seed))
    outdir = os.path.join(out_root, "eikonal")
    os.makedirs(outdir, exist_ok=True)
    ve.run_dataset(dataset, cfg, outdir)

    eval_path = os.path.join(outdir, f"{dataset}_on_eikonal_eval.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "method": "eikonal",
        "dataset": dataset,
        "metrics": _normalize_metrics(metrics),
        "eval_path": eval_path,
        "config": asdict(cfg),
    }


def run_udf_one(
    method: str,
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    cfg_mapping: dict[str, Any],
) -> dict[str, Any]:
    cfg = udf.Config()
    _apply_dict_to_cfg(cfg, cfg_mapping)
    _apply_projector_subcfg(cfg)

    if seed_override is not None:
        cfg.seed = int(seed_override)

    if str(cfg.device) == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_seed(int(cfg.seed))

    ds = resolve_dataset(
        dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=str(getattr(cfg, "ur5_backend", "analytic")),
    )
    x_train = ds["x_train"]

    knn_k = udf.effective_knn_norm_estimation_points(cfg, len(x_train))
    n_hat = udf.knn_normals(x_train, knn_k, cfg)
    model, train_stats, train_history = udf.train_baseline(cfg, mode=method, x=x_train, n_hat=n_hat)

    if _is_arm_dataset(dataset):
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(dataset):
        post_fn = _wrap_workspace_pose_rpy_np
    else:
        post_fn = None
    embed_fn = (
        lambda q, _name=dataset: _workspace_embed_for_eval(
            _name,
            q,
            ur5_use_pybullet_n6=(str(getattr(cfg, "ur5_backend", "analytic")).lower() == "pybullet"),
        )
    ) if (_is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset)) else None

    project_fn = udf._make_project_fn(cfg)
    metrics, eval_cfg, eval_artifacts = run_eval_metrics(
        cfg=cfg,
        method_key=method,
        dataset_name=dataset,
        model=model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )
    _print_eval_lines(dataset, metrics)

    outdir = os.path.join(out_root, method)
    os.makedirs(outdir, exist_ok=True)
    vis_cfg = _make_vis_cfg_for_method(dataset, cfg, eval_cfg)
    grid_vis = ds.get("grid", None)
    if grid_vis is None:
        grid_vis = x_train
    _save_udf_legacy_plots(
        method=str(method),
        dataset=str(dataset),
        model=model,
        x_train=x_train,
        grid=np.asarray(grid_vis, dtype=np.float32),
        cfg=cfg,
        vis_cfg=vis_cfg,
        eval_artifacts=eval_artifacts,
        train_history=train_history,
        outdir=outdir,
        ds_info=ds,
    )
    if int(x_train.shape[1]) != 2:
        _save_common_method_plots(
            dataset=dataset,
            method_tag=str(method),
            model=model,
            x_train=x_train,
            outdir=outdir,
            vis_cfg=vis_cfg,
            eval_artifacts=eval_artifacts,
        )
    if dataset in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
        out_dist = os.path.join(outdir, f"{dataset}_{method}_proj_value_distribution.png")
        _plot_ur5_projection_error_distribution_from_pairs(
            q_eval=eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_dist,
            title=f"{dataset} ({method}): orientation-angle error before/after projection",
            use_pybullet_n6=(dataset == "6d_spatial_arm_up_n6"),
        )
        out_ws = os.path.join(outdir, f"{dataset}_{method}_eval_proj_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_ws,
            title=f"{dataset} ({method}): eval projected points in workspace + tool orientation",
            use_pybullet_n6=(dataset == "6d_spatial_arm_up_n6"),
        )
    if dataset == "6d_workspace_sine_surface_pose":
        out_pose = os.path.join(outdir, f"{dataset}_{method}_workspace_pose_orientation.png")
        _plot_workspace_pose_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} ({method}): projected eval poses + orientation z-axis",
        )
        out_err = os.path.join(outdir, f"{dataset}_{method}_workspace_pose_proj_error_distributions.png")
        _plot_workspace_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 6), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} ({method}): projection errors before/after",
        )

    eval_path = os.path.join(outdir, f"{dataset}_{method}_eval.json")
    ckpt_path = os.path.join(outdir, f"{dataset}_{method}_model.pt")

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "dataset": str(dataset),
            "method": str(method),
            "model_state": model.state_dict(),
            "in_dim": int(x_train.shape[1]),
            "hidden": int(cfg.hidden),
            "depth": int(cfg.depth),
            "x_train": x_train,
            "train_stats": train_stats,
            "cfg": asdict(cfg),
        },
        ckpt_path,
    )

    return {
        "method": method,
        "dataset": dataset,
        "metrics": _normalize_metrics(metrics),
        "eval_path": eval_path,
        "ckpt_path": ckpt_path,
        "config": asdict(cfg),
    }


def run_one(
    method: str,
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    config_root: str,
    cli_overrides: list[str],
) -> tuple[dict[str, Any], list[str]]:
    if method not in VALID_METHODS:
        raise ValueError(f"unsupported method '{method}', valid={sorted(VALID_METHODS)}")

    cfg_mapping, loaded_paths = _resolve_run_config(
        method,
        dataset,
        config_root=config_root,
        cli_overrides=cli_overrides,
    )

    if method == "eikonal":
        result = run_eikonal_one(
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
        )
    elif method == "vae":
        result = run_vae_one(
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
        )
    else:
        result = run_udf_one(
            method,
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
        )

    result["loaded_config_paths"] = loaded_paths
    return result, loaded_paths


def _resolve_vae_latent_dim(cfg: VAEConfig, data_dim: int, true_codim: int) -> int:
    if int(cfg.latent_dim) > 0:
        return int(cfg.latent_dim)
    # Default to intrinsic dimension estimate d-codim, clamp to [1, d-1].
    est = int(data_dim) - int(true_codim)
    return max(1, min(int(data_dim) - 1, est))


def run_vae_one(
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    cfg_mapping: dict[str, Any],
) -> dict[str, Any]:
    cfg = VAEConfig()
    _apply_dict_to_cfg(cfg, cfg_mapping)
    _apply_projector_subcfg(cfg)

    if seed_override is not None:
        cfg.seed = int(seed_override)

    if str(cfg.device) == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_seed(int(cfg.seed))

    ds = resolve_dataset(
        dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=str(getattr(cfg, "ur5_backend", "analytic")),
    )
    x_train = ds["x_train"]
    data_dim = int(x_train.shape[1])
    true_codim = int(ds.get("true_codim", 1))
    latent_dim = _resolve_vae_latent_dim(cfg, data_dim, true_codim)
    hidden = tuple(int(v) for v in cfg.hidden_dims)

    vae_model = vae_models.VAE(in_dim=data_dim, latent_dim=latent_dim, hidden=hidden).to(cfg.device)
    train_cfg = vae_train.TrainConfig(
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        beta_final=float(cfg.beta_final),
        warmup_epochs=int(cfg.warmup_epochs),
    )
    x_train_t = torch.from_numpy(x_train.astype(np.float32)).to(cfg.device)
    vae_train.train_vae(vae_model, x_train_t, train_cfg, torch.device(cfg.device))

    field_model = _VAEProjectorField(vae_model).to(cfg.device)
    field_model.eval()

    def project_fn(_model: nn.Module, x0: np.ndarray, _eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            xt = torch.from_numpy(x0.astype(np.float32)).to(cfg.device)
            x_proj = field_model.project_tensor(xt).detach().cpu().numpy().astype(np.float32)
        steps = np.ones((len(x_proj),), dtype=np.float32)
        return x_proj, steps

    if _is_arm_dataset(dataset):
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(dataset):
        post_fn = _wrap_workspace_pose_rpy_np
    else:
        post_fn = None
    embed_fn = (
        lambda q, _name=dataset: _workspace_embed_for_eval(
            _name,
            q,
            ur5_use_pybullet_n6=(str(getattr(cfg, "ur5_backend", "analytic")).lower() == "pybullet"),
        )
    ) if (_is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset)) else None

    metrics, eval_cfg, eval_artifacts = run_eval_metrics(
        cfg=cfg,
        method_key="vae",
        dataset_name=dataset,
        model=field_model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )
    _print_eval_lines(dataset, metrics)
    outdir = os.path.join(out_root, "vae")
    os.makedirs(outdir, exist_ok=True)

    if bool(cfg.viz_enable):
        x_eval = eval_artifacts["x_eval"].astype(np.float32)
        x_proj = eval_artifacts["proj"].astype(np.float32)
        if int(cfg.viz_max_eval_points) > 0 and len(x_eval) > int(cfg.viz_max_eval_points):
            idx = np.random.default_rng(int(cfg.seed) + 99).choice(
                len(x_eval), size=int(cfg.viz_max_eval_points), replace=False
            )
            x_eval = x_eval[idx].astype(np.float32)
            x_proj = x_proj[idx].astype(np.float32)

        gt_grid = resolve_gt_grid(dataset, eval_cfg, x_train=x_train).astype(np.float32)
        if post_fn is not None:
            gt_grid = post_fn(gt_grid).astype(np.float32)

        if embed_fn is not None:
            x_eval_metric = embed_fn(x_eval).astype(np.float32)
            gt_metric = embed_fn(gt_grid).astype(np.float32)
        else:
            x_eval_metric = x_eval
            gt_metric = gt_grid
        tau = float(eval_cfg.eval_tau_ratio) * float(
            np.mean(gt_metric.max(axis=0) - gt_metric.min(axis=0))
        )
        d_true = _nn_dist_numpy(x_eval_metric, gt_metric)
        y_true = (d_true < tau).astype(np.int64)

        with torch.no_grad():
            xt = torch.from_numpy(x_eval.astype(np.float32)).to(cfg.device)
            mu, _ = vae_model.encode(xt)
            z_eval = mu.detach().cpu().numpy().astype(np.float32)

            n_samp = max(1, int(cfg.viz_sample_latent_n))
            z_samp = np.random.default_rng(int(cfg.seed) + 7).normal(
                size=(n_samp, int(latent_dim))
            ).astype(np.float32)
            x_dec = vae_model.decode(torch.from_numpy(z_samp).to(cfg.device)).detach().cpu().numpy().astype(np.float32)

        cache = {
            "x": x_eval,
            "y_true": y_true,
            "err": np.linalg.norm(x_eval - x_proj, axis=1).astype(np.float32),
            "x_proj": x_proj,
        }
        vae_pack = (cache, z_eval, x_dec, z_samp)
        ds_vis = SimpleNamespace(name=str(dataset), dim=int(data_dim))
        viz_path = os.path.join(outdir, f"{dataset}_vae_visualize_all.png")
        vae_viz.visualize_all(
            ds=ds_vis,
            x_train=x_train,
            ae_pack=None,
            vae_pack=vae_pack,
            latent_dim=int(latent_dim),
            save_path=viz_path,
            show=False,
        )
        print(f"saved: {viz_path}")

    vis_cfg = _make_vis_cfg_for_method(dataset, cfg, eval_cfg)
    _save_common_method_plots(
        dataset=dataset,
        method_tag="vae",
        model=field_model,
        x_train=x_train,
        outdir=outdir,
        vis_cfg=vis_cfg,
        eval_artifacts=eval_artifacts,
    )

    if dataset in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
        out_dist = os.path.join(outdir, f"{dataset}_vae_proj_value_distribution.png")
        _plot_ur5_projection_error_distribution_from_pairs(
            q_eval=eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_dist,
            title=f"{dataset} (vae): orientation-angle error before/after projection",
            use_pybullet_n6=(dataset == "6d_spatial_arm_up_n6"),
        )
        out_ws = os.path.join(outdir, f"{dataset}_vae_eval_proj_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_ws,
            title=f"{dataset} (vae): eval projected points in workspace + tool orientation",
            use_pybullet_n6=(dataset == "6d_spatial_arm_up_n6"),
        )
    if dataset == "6d_workspace_sine_surface_pose":
        out_pose = os.path.join(outdir, f"{dataset}_vae_workspace_pose_orientation.png")
        _plot_workspace_pose_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} (vae): projected eval poses + orientation z-axis",
        )
        out_err = os.path.join(outdir, f"{dataset}_vae_workspace_pose_proj_error_distributions.png")
        _plot_workspace_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 6), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} (vae): projection errors before/after",
        )

    eval_path = os.path.join(outdir, f"{dataset}_vae_eval.json")
    ckpt_path = os.path.join(outdir, f"{dataset}_vae_model.pt")

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "dataset": str(dataset),
            "method": "vae",
            "model_state": vae_model.state_dict(),
            "in_dim": data_dim,
            "latent_dim": latent_dim,
            "hidden_dims": hidden,
            "x_train": x_train,
            "cfg": asdict(cfg),
        },
        ckpt_path,
    )

    return {
        "method": "vae",
        "dataset": dataset,
        "metrics": _normalize_metrics(metrics),
        "eval_path": eval_path,
        "ckpt_path": ckpt_path,
        "config": asdict(cfg),
    }
