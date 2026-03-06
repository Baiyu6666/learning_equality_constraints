from __future__ import annotations

from typing import Any, Callable
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from datasets.constraint_datasets import generate_dataset, lift_xy_to_3d_var, lift_xy_to_3d_zero

DEFAULT_EVAL_CFG: dict[str, Any] = {
    "device": "cpu",
    "eval_pad_ratio": 0.6,
    "eval_min_axis_span_ratio": 0.08,
    "zero_eps_quantile": 90.0,
    "eval_chamfer_n_gt": 1024,
    "eval_chamfer_n_seed": 4096,
    "eval_chamfer_near_ratio": 0.75,
    "eval_chamfer_near_noise_std_ratio": 0.1,
    "eval_tau_ratio": 0.015,
}
EVALUATOR_FIXED_SEED = 2026

# Centralized evaluator override hooks.
# Priority: DEFAULT_EVAL_CFG < method override < dataset override
EVAL_METHOD_OVERRIDES: dict[str, dict[str, Any]] = {
    "vector_eikonal": {},
    "margin": {},
    "delta": {},
    "vae": {},
    "ecomann": {},
}
EVAL_DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    # Keep explicit keys as templates; defaults are currently identical to DEFAULT_EVAL_CFG.
    "2d_figure_eight": {},
    "2d_ellipse": {},
    "2d_noisy_sine": {},
    "2d_sine": {},
    "2d_sparse_sine": {},
    "2d_discontinuous": {},
    "2d_looped_spiro": {},
    "2d_sharp_star": {},
    "2d_hetero_noise": {},
    "2d_planar_arm_line_n2": {},
    "3d_saddle_surface": {},
    "3d_sphere_surface": {},
    "3d_torus_surface": {},
    "3d_planar_arm_line_n3": {},
    "3d_spatial_arm_plane_n3": {},
    "3d_spatial_arm_ellip_n3": {},
    "3d_spatial_arm_circle_n3": {},
    "6d_spatial_arm_up_n6": {},
    "6d_spatial_arm_up_n6_py": {},
    "6d_workspace_sine_surface_pose": {},
    "6d_workspace_sine_surface_pose_traj": {},
}


# Deterministic evaluator RNG:
# every evaluation call uses the exact same sampling stream for strict reproducibility.
def _fixed_rng() -> np.random.Generator:
    return np.random.default_rng(EVALUATOR_FIXED_SEED)


_GT_GRID_CACHE: dict[tuple[str, int], np.ndarray] = {}


def resolve_gt_grid(
    dataset_name: str,
    cfg: Any,
    x_train: np.ndarray | None = None,
) -> np.ndarray:
    cfg_ds = SimpleNamespace(**vars(cfg)) if hasattr(cfg, "__dict__") else SimpleNamespace()
    if not hasattr(cfg_ds, "n_grid"):
        cfg_ds.n_grid = int(getattr(cfg, "eval_chamfer_n_seed", 4096))
    if not hasattr(cfg_ds, "n_train"):
        cfg_ds.n_train = int(max(64, getattr(cfg_ds, "n_grid", 4096)))
    if not hasattr(cfg_ds, "seed"):
        cfg_ds.seed = 0

    cache_key = (
        str(dataset_name),
        int(getattr(cfg_ds, "n_grid", 4096)),
        int(getattr(cfg_ds, "seed", 0)),
        float(getattr(cfg_ds, "z_amp1", 0.35)),
        float(getattr(cfg_ds, "z_amp2", 0.20)),
        float(getattr(cfg_ds, "z_freq1", 1.5)),
        float(getattr(cfg_ds, "z_freq2", 1.2)),
    )
    if cache_key in _GT_GRID_CACHE:
        return _GT_GRID_CACHE[cache_key]

    name = str(dataset_name)
    if name.startswith("3d_0z_"):
        base = name[len("3d_0z_"):]
        x2, grid2 = generate_dataset(base, cfg_ds)
        src = grid2 if grid2 is not None else x2
        out = lift_xy_to_3d_zero(src).astype(np.float32)
        _GT_GRID_CACHE[cache_key] = out
        return out
    if name.startswith("3d_vz_"):
        base = name[len("3d_vz_"):]
        x2, grid2 = generate_dataset(base, cfg_ds)
        src = grid2 if grid2 is not None else x2
        out = lift_xy_to_3d_var(src, cfg_ds).astype(np.float32)
        _GT_GRID_CACHE[cache_key] = out
        return out
    try:
        x, grid = generate_dataset(name, cfg_ds)
        src = grid if grid is not None else x
        out = src.astype(np.float32)
        _GT_GRID_CACHE[cache_key] = out
        return out
    except Exception:
        if x_train is None:
            raise
        return x_train.astype(np.float32)


def resolve_eval_cfg(
    base_cfg: Any,
    method_key: str | None = None,
    dataset_name: str | None = None,
) -> Any:
    vals = dict(DEFAULT_EVAL_CFG)
    # Method configs should not carry evaluator hyperparameters.
    # Keep runtime/device and required passthrough fields.
    if hasattr(base_cfg, "device"):
        vals["device"] = getattr(base_cfg, "device")
    if method_key:
        vals.update(EVAL_METHOD_OVERRIDES.get(method_key, {}))
    if dataset_name:
        vals.update(EVAL_DATASET_OVERRIDES.get(dataset_name, {}))
    return SimpleNamespace(**vals)


def eval_bounds_from_train(x_train: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    mins = x_train.min(axis=0)
    maxs = x_train.max(axis=0)
    span_raw = maxs - mins
    min_axis_ratio = float(getattr(cfg, "eval_min_axis_span_ratio", DEFAULT_EVAL_CFG["eval_min_axis_span_ratio"]))
    ref_span = max(float(np.max(span_raw)), 1e-6)
    min_axis_span = max(0.0, min_axis_ratio) * ref_span
    span = np.maximum(span_raw, min_axis_span)
    pad_ratio = float(getattr(cfg, "eval_pad_ratio", DEFAULT_EVAL_CFG["eval_pad_ratio"]))
    scale = max(1.0 + pad_ratio, 1e-6)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span * scale
    return center - half, center + half


def _clip_points_to_eval_bounds(x: np.ndarray, x_train: np.ndarray, cfg: Any) -> np.ndarray:
    """Evaluation-only safety clip to the padded train-domain bounds."""
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    return np.clip(
        x.astype(np.float32),
        mins.reshape(1, -1).astype(np.float32),
        maxs.reshape(1, -1).astype(np.float32),
    ).astype(np.float32)


def sample_eval_seed_points(
    x_train: np.ndarray,
    cfg: Any,
) -> np.ndarray:
    rng = _fixed_rng()
    n_seed = max(64, int(cfg.eval_chamfer_n_seed))
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    span = np.maximum(maxs - mins, 1e-6).astype(np.float32)
    near_ratio = float(np.clip(cfg.eval_chamfer_near_ratio, 0.0, 1.0))
    n_near = int(round(n_seed * near_ratio))
    n_box = max(0, n_seed - n_near)
    out = []
    if n_near > 0:
        idx = rng.integers(0, len(x_train), size=n_near)
        x0_near = x_train[idx].astype(np.float32).copy()
        noise_std = float(max(cfg.eval_chamfer_near_noise_std_ratio, 1e-6))
        x0_near = x0_near + rng.normal(size=x0_near.shape).astype(np.float32) * (
            noise_std * span.reshape(1, -1)
        )
        x0_near = np.clip(x0_near, mins.reshape(1, -1), maxs.reshape(1, -1))
        out.append(x0_near.astype(np.float32))
    if n_box > 0:
        out.append(rng.uniform(mins, maxs, size=(n_box, len(mins))).astype(np.float32))
    if not out:
        return rng.uniform(mins, maxs, size=(n_seed, len(mins))).astype(np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def compute_eps_stop(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    *,
    return_h_on: bool = False,
) -> float | tuple[float, np.ndarray]:
    with torch.no_grad():
        f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
        if f_on.dim() == 1:
            f_on = f_on.unsqueeze(1)
        h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy().reshape(-1)
    eps = float(np.percentile(np.abs(h_on), float(cfg.zero_eps_quantile)))
    if bool(return_h_on):
        return eps, h_on.astype(np.float32)
    return eps


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


def _true_projection(x: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(grid.astype(np.float64))
        d, idx = tree.query(x.astype(np.float64), k=1)
        return grid[idx], d.astype(np.float32)
    except Exception:
        xx = x.astype(np.float32)
        gg = grid.astype(np.float32)
        d2 = np.sum((xx[:, None, :] - gg[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        d = np.sqrt(np.maximum(d2[np.arange(len(x)), idx], 0.0)).astype(np.float32)
        return gg[idx], d


def _workspace_surface_z_and_normal_from_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Keep consistent with datasets.constraint_datasets._workspace_sine_surface_pose_n6
    a1, a2 = 0.55, 0.35
    fx, fy = 1.2, 1.0
    z = (a1 * np.sin(fx * x) + a2 * np.cos(fy * y)).astype(np.float32)
    dzdx = (a1 * fx * np.cos(fx * x)).astype(np.float32)
    dzdy = (-a2 * fy * np.sin(fy * y)).astype(np.float32)
    n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=1).astype(np.float32)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return z, n


def _workspace_pose_analytic_target_embed(embed_xyz_zaxis: np.ndarray) -> np.ndarray:
    x = embed_xyz_zaxis[:, 0].astype(np.float32)
    y = embed_xyz_zaxis[:, 1].astype(np.float32)
    z_true, n_true = _workspace_surface_z_and_normal_from_xy(x, y)
    tgt = np.concatenate(
        [x[:, None], y[:, None], z_true[:, None], n_true.astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    return tgt


def _workspace_pose_analytic_dist_embed(embed_xyz_zaxis: np.ndarray) -> np.ndarray:
    tgt = _workspace_pose_analytic_target_embed(embed_xyz_zaxis)
    d = np.linalg.norm(
        embed_xyz_zaxis.astype(np.float32) - tgt.astype(np.float32),
        axis=1,
    ).astype(np.float32)
    return d


def evaluate_bidirectional_chamfer(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    project_fn: Callable[[nn.Module, np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    dataset_name: str | None = None,
    embed_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    postprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    eps_stop_override: float | None = None,
    x0_override: np.ndarray | None = None,
    learned_samples_override: np.ndarray | None = None,
) -> dict[str, float]:
    rng = _fixed_rng()
    if dataset_name is not None:
        gt_grid = resolve_gt_grid(str(dataset_name), cfg, x_train=x_train)
        n_gt = min(len(gt_grid), max(64, int(cfg.eval_chamfer_n_gt)))
        idx = rng.choice(len(gt_grid), size=n_gt, replace=False)
        gt_samples = gt_grid[idx].astype(np.float32)
    else:
        n_gt = min(len(x_train), max(64, int(cfg.eval_chamfer_n_gt)))
        idx = rng.choice(len(x_train), size=n_gt, replace=False)
        gt_samples = x_train[idx].astype(np.float32)
    if postprocess_fn is not None:
        gt_samples = postprocess_fn(gt_samples).astype(np.float32)

    eps_stop = float(eps_stop_override) if eps_stop_override is not None else float(compute_eps_stop(model, x_train, cfg))
    if learned_samples_override is not None:
        learned_samples = _clip_points_to_eval_bounds(
            learned_samples_override.astype(np.float32),
            x_train,
            cfg,
        )
    else:
        x0 = x0_override.astype(np.float32) if x0_override is not None else sample_eval_seed_points(x_train, cfg)
        if postprocess_fn is not None:
            x0 = postprocess_fn(x0)
        learned_samples, _ = project_fn(model, x0, eps_stop)
        learned_samples = _clip_points_to_eval_bounds(
            learned_samples.astype(np.float32),
            x_train,
            cfg,
        )
        learned_samples = learned_samples.astype(np.float32)
        if postprocess_fn is not None:
            learned_samples = postprocess_fn(learned_samples)

    if embed_fn is not None:
        a = embed_fn(gt_samples)
        b = embed_fn(learned_samples)
        dist_space = "workspace"
    else:
        a = gt_samples
        b = learned_samples
        dist_space = "data_space"

    d_gt_to_learned = _nn_dist(a, b)
    d_learned_to_gt = _nn_dist(b, a)
    return {
        "bidirectional_chamfer": float(np.mean(d_gt_to_learned) + np.mean(d_learned_to_gt)),
        "gt_to_learned_mean": float(np.mean(d_gt_to_learned)),
        "learned_to_gt_mean": float(np.mean(d_learned_to_gt)),
        "gt_to_learned_p95": float(np.percentile(d_gt_to_learned, 95)),
        "learned_to_gt_p95": float(np.percentile(d_learned_to_gt, 95)),
        "dist_space": dist_space,
        "n_gt": int(len(gt_samples)),
        "n_learned": int(len(learned_samples)),
        "eps_stop": float(eps_stop),
    }


def evaluate_projection_metrics(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    project_fn: Callable[[nn.Module, np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    dataset_name: str | None = None,
    embed_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    postprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    return_artifacts: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
    if dataset_name is not None:
        grid = resolve_gt_grid(str(dataset_name), cfg, x_train=x_train)
    else:
        grid = x_train.astype(np.float32)
    if postprocess_fn is not None:
        grid = postprocess_fn(grid).astype(np.float32)

    eval_eps_used, h_on = compute_eps_stop(model, x_train, cfg, return_h_on=True)
    x_eval = sample_eval_seed_points(x_train, cfg)
    if postprocess_fn is not None:
        x_eval = postprocess_fn(x_eval).astype(np.float32)
    with torch.no_grad():
        f_eval = model(torch.from_numpy(x_eval.astype(np.float32)).to(cfg.device))
        if f_eval.dim() == 1:
            f_eval = f_eval.unsqueeze(1)
        h_eval = torch.linalg.norm(f_eval, dim=1).detach().cpu().numpy().reshape(-1)
    on_mean_v = float(np.mean(0.5 * (h_on ** 2)))

    use_workspace_analytic = (
        str(dataset_name) in ("6d_workspace_sine_surface_pose", "6d_workspace_sine_surface_pose_traj")
        and embed_fn is not None
    )

    if embed_fn is not None:
        grid_metric = embed_fn(grid)
        x_eval_metric = embed_fn(x_eval)
        if (
            use_workspace_analytic
            and x_eval_metric.ndim == 2
            and x_eval_metric.shape[1] >= 6
        ):
            d_true_eval = _workspace_pose_analytic_dist_embed(
                x_eval_metric[:, :6].astype(np.float32)
            )
            dist_space = "workspace_analytic"
        else:
            _, d_true_eval = _true_projection(x_eval_metric, grid_metric)
            dist_space = "workspace"
    else:
        grid_metric = grid
        x_eval_metric = x_eval
        _, d_true_eval = _true_projection(x_eval_metric, grid_metric)
        dist_space = "data_space"

    proj, steps = project_fn(model, x_eval, eval_eps_used)
    proj = _clip_points_to_eval_bounds(
        proj.astype(np.float32),
        x_train,
        cfg,
    )
    if postprocess_fn is not None:
        proj = postprocess_fn(proj).astype(np.float32)

    if embed_fn is not None:
        proj_metric_all = embed_fn(proj)
        if (
            use_workspace_analytic
            and proj_metric_all.ndim == 2
            and proj_metric_all.shape[1] >= 6
            and x_eval_metric.ndim == 2
            and x_eval_metric.shape[1] >= 6
        ):
            proj_true = _workspace_pose_analytic_target_embed(
                x_eval_metric[:, :6].astype(np.float32)
            )
        else:
            proj_true, _ = _true_projection(x_eval_metric, grid_metric)
    else:
        proj_metric_all = proj
        proj_true, _ = _true_projection(x_eval_metric, grid_metric)

    proj_mask = np.isfinite(proj).all(axis=1)
    if np.any(proj_mask):
        proj_to_trueproj = np.linalg.norm(proj_metric_all[proj_mask] - proj_true[proj_mask], axis=1)
        if (
            use_workspace_analytic
            and proj_metric_all.ndim == 2
            and proj_metric_all.shape[1] >= 6
        ):
            proj_final_true_dist = _workspace_pose_analytic_dist_embed(
                proj_metric_all[proj_mask, :6].astype(np.float32)
            )
        else:
            proj_final_true_dist = _nn_dist(proj_metric_all[proj_mask], grid_metric)
        with torch.no_grad():
            f_proj = model(torch.from_numpy(proj[proj_mask].astype(np.float32)).to(cfg.device))
            if f_proj.dim() == 1:
                f_proj = f_proj.unsqueeze(1)
            h_proj = torch.linalg.norm(f_proj, dim=1).detach().cpu().numpy().reshape(-1)
        proj_residual = np.abs(h_proj)
        proj_steps_mean = float(np.mean(steps[proj_mask]))
    else:
        proj_to_trueproj = np.array([float("nan")], dtype=np.float32)
        proj_final_true_dist = np.array([float("nan")], dtype=np.float32)
        proj_residual = np.array([float("nan")], dtype=np.float32)
        proj_steps_mean = float("nan")

    tau = float(cfg.eval_tau_ratio) * float(np.mean(grid_metric.max(axis=0) - grid_metric.min(axis=0)))

    # Recall: measure how many GT manifold points are classified as on-manifold by learned constraint.
    with torch.no_grad():
        f_gt = model(torch.from_numpy(grid.astype(np.float32)).to(cfg.device))
        if f_gt.dim() == 1:
            f_gt = f_gt.unsqueeze(1)
        h_gt = torch.linalg.norm(f_gt, dim=1).detach().cpu().numpy().reshape(-1)
    pred_on_gt = np.abs(h_gt) < eval_eps_used
    coverage = float(np.mean(pred_on_gt)) if pred_on_gt.size > 0 else float("nan")

    # False-positive rate: still measured on sampled eval points that are far from GT manifold.
    pred_zero_eval = np.abs(h_eval) < eval_eps_used
    far_eval = d_true_eval >= tau
    false_pos = float(np.mean(pred_zero_eval[far_eval])) if np.any(far_eval) else float("nan")

    # Precision: use projected points as predicted on-manifold set and check GT distance against tau.
    if np.any(proj_mask):
        pred_on_true_ratio = float(np.mean(proj_final_true_dist < tau))
    else:
        pred_on_true_ratio = float("nan")

    metrics = {
        "pred_on_mean_v": on_mean_v,
        "proj_manifold_dist": float(np.mean(proj_final_true_dist)),
        "proj_v_residual": float(np.mean(proj_residual)),
        "proj_true_dist": float(np.mean(proj_to_trueproj)),
        "proj_steps": proj_steps_mean,
        "pred_recall": coverage,
        "pred_FPrate": false_pos,
        "pred_precision": pred_on_true_ratio,
        "eval_eps_used": float(eval_eps_used),
        "dist_space": dist_space,
    }
    if not bool(return_artifacts):
        return metrics
    artifacts = {
        "x_eval": x_eval.astype(np.float32),
        "proj": proj.astype(np.float32),
        "steps": steps.astype(np.float32),
        "eval_eps_used": float(eval_eps_used),
    }
    return metrics, artifacts
