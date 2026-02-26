from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

from methods.baseline_udf.baseline_udf import (
    Config,
    generate_dataset,
    knn_normals,
    plot_contour_and_trajectory,
    project_trajectory,
    set_seed,
    train_baseline,
)
from evaluator.evaluator import (
    compute_eps_stop,
    evaluate_projection_metrics,
    resolve_eval_cfg,
    sample_eval_seed_points,
)


DEFAULT_DATASETS = [
    "2d_figure_eight",
    "2d_ellipse",
    "2d_noisy_sine",
    "2d_sparse_sine",
    "2d_hetero_noise",
    "2d_looped_spiro",
]

NORM_PROJ_BASELINE = {
    "2d_figure_eight": {"mean": 0.120565, "std": 0.107640},
    "2d_ellipse": {"mean": 0.081101, "std": 0.125372},
    "2d_noisy_sine": {"mean": 0.459975, "std": 0.225468},
    "2d_sparse_sine": {"mean": 0.264254, "std": 0.239843},
    "2d_looped_spiro": {"mean": 0.190081, "std": 0.200199},
}


def _expand_values(spec: Any) -> List[Any]:
    if isinstance(spec, dict):
        if "values" in spec:
            return list(spec["values"])
        if "value" in spec:
            return [spec["value"]]
        if "range" in spec:
            r = spec["range"]
            n = int(spec.get("n", 3))
            scale = spec.get("scale", "linear")
            if scale == "log":
                return list(np.logspace(np.log10(r[0]), np.log10(r[1]), n))
            return list(np.linspace(r[0], r[1], n))
    if isinstance(spec, list):
        return list(spec)
    return [spec]


def _param_grid(params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    keys = list(params.keys())
    values = [_expand_values(params[k]) for k in keys]
    for combo in itertools.product(*values):
        yield {k: v for k, v in zip(keys, combo)}


def _apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> None:
    alias = {
        "k_ratio": "knn_norm_estimation_ratio",
        "k_min": "knn_norm_estimation_min_points",
        "k_accept_ratio": "knn_off_data_filter_ratio",
        "k_accept_min": "knn_off_data_filter_min_points",
    }
    for k, v in overrides.items():
        k = alias.get(k, k)
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def _make_project_fn(cfg: Config):
    def _project_points_for_eval(
        model: torch.nn.Module, x0: np.ndarray, eps_stop: float
    ) -> tuple[np.ndarray, np.ndarray]:
        x0_t = torch.from_numpy(x0.astype(np.float32)).to(cfg.device)
        traj = project_trajectory(model, x0_t, cfg, eps_stop)
        final = traj[-1].astype(np.float32)
        steps = np.full((len(final),), max(0, traj.shape[0] - 1), dtype=np.float32)
        return final, steps

    return _project_points_for_eval


def run_one(
    cfg_base: Config,
    method: str,
    params: Dict[str, Any],
    datasets: List[str],
) -> Dict[str, Any]:
    cfg = Config(**cfg_base.__dict__)
    _apply_overrides(cfg, params)
    if cfg.device == "auto":
        import torch

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    results = []
    for idx, name in enumerate(datasets):
        set_seed(cfg.seed)
        x_train, grid = generate_dataset(name, cfg)
        eval_cfg = resolve_eval_cfg(cfg, method_key=method, dataset_name=name)
        if method in ("margin", "delta"):
            knn_norm_estimation_points = max(
                cfg.knn_norm_estimation_min_points,
                int(round(cfg.knn_norm_estimation_ratio * len(x_train))),
            )
            n_hat = knn_normals(x_train, knn_norm_estimation_points, cfg)
            model, _, _ = train_baseline(cfg, mode=method, x=x_train, n_hat=n_hat)
        else:
            raise ValueError(f"unknown method: {method}")
        project_fn = _make_project_fn(cfg)

        metrics = evaluate_projection_metrics(
            model=model,
            x_train=x_train,
            cfg=eval_cfg,
            project_fn=project_fn,
            dataset_name=name,
        )
        results.append({"dataset": name, "metrics": metrics})

    avg = {}
    keys = results[0]["metrics"].keys()
    for k in keys:
        avg[k] = float(np.mean([r["metrics"][k] for r in results]))

    return {"method": method, "params": params, "results": results, "avg": avg}


def _format_value(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


def _build_run_name(
    method: str,
    params: Dict[str, Any],
    shared: Dict[str, Any],
    tuning_keys: Optional[Set[str]] = None,
    method_tag: Optional[str] = None,
) -> str:
    default_cfg = Config()
    parts = [method_tag or method]
    for k, v in params.items():
        if tuning_keys is None or k in tuning_keys:
            parts.append(f"{k}={_format_value(v)}")
    for k, v in shared.items():
        if hasattr(default_cfg, k) and getattr(default_cfg, k) != v:
            parts.append(f"{k}={_format_value(v)}")
    return "-".join(parts)


def _build_wandb_sweep(
    method: str, params: Dict[str, Any], shared: Dict[str, Any]
) -> Tuple[Dict[str, Any], Set[str]]:
    sweep_params: Dict[str, Any] = {"method": {"value": method}}
    tuning_keys: Set[str] = set()
    for k, spec in params.items():
        values = _expand_values(spec)
        if len(values) > 1:
            tuning_keys.add(k)
        sweep_params[k] = {"values": values}
    for k, v in shared.items():
        sweep_params[k] = {"value": v}
    sweep_config = {
        "method": "grid",
        "parameters": sweep_params,
    }
    return sweep_config, tuning_keys


_SWEEP_CONTEXT: Dict[str, Any] = {}


def _wandb_train() -> None:
    if wandb is None:
        raise RuntimeError("wandb not available")
    with wandb.init() as run:
        cfg = Config()
        shared = _SWEEP_CONTEXT["shared"]
        method = wandb.config.get("method", _SWEEP_CONTEXT["method"])
        method_tag = wandb.config.get("method_tag", None)
        _apply_overrides(cfg, shared)
        overrides = {}
        for k, v in dict(wandb.config).items():
            if k == "method":
                continue
            if hasattr(cfg, k):
                overrides[k] = v
        _apply_overrides(cfg, overrides)
        if cfg.device == "auto":
            import torch

            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg.device == "cuda":
            import torch

            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        run.name = _build_run_name(
            method, overrides, shared, _SWEEP_CONTEXT["tuning_keys"], method_tag
        )
        if method_tag:
            run.tags = list(run.tags) + [str(method_tag)]

        datasets = _SWEEP_CONTEXT["datasets"]
        rng = np.random.default_rng(7)
        results = []
        metrics_table = wandb.Table(
            columns=["dataset", "metric", "value", "method", "seed"]
        )
        norm_vals = []
        for idx, name in enumerate(datasets):
            set_seed(cfg.seed)
            x_train, grid = generate_dataset(name, cfg)
            eval_cfg = resolve_eval_cfg(cfg, method_key=method, dataset_name=name)
            x_eval = sample_eval_seed_points(x_train, eval_cfg)

            if method in ("margin", "delta"):
                knn_norm_estimation_points = max(
                    cfg.knn_norm_estimation_min_points,
                    int(round(cfg.knn_norm_estimation_ratio * len(x_train))),
                )
                n_hat = knn_normals(x_train, knn_norm_estimation_points, cfg)
                model, _, _ = train_baseline(cfg, mode=method, x=x_train, n_hat=n_hat)
            else:
                raise ValueError(f"unknown method: {method}")
            project_fn = _make_project_fn(cfg)

            metrics = evaluate_projection_metrics(
                model=model,
                x_train=x_train,
                cfg=eval_cfg,
                project_fn=project_fn,
                dataset_name=name,
            )
            results.append({"dataset": name, "metrics": metrics})
            wandb.log({f"{name}/{k}": v for k, v in metrics.items()})
            for k, v in metrics.items():
                metrics_table.add_data(name, k, v, method, cfg.seed)
            base = NORM_PROJ_BASELINE.get(name)
            if base is not None and base["std"] > 0:
                norm_val = (metrics["proj_manifold_dist"] - base["mean"]) / base["std"]
                norm_vals.append(norm_val)
                wandb.log({f"{name}/norm_proj_manifold_dist": norm_val})
                metrics_table.add_data(
                    name, "norm_proj_manifold_dist", norm_val, method, cfg.seed
                )
            if x_train.shape[1] == 3:
                n_plot = 16
            else:
                n_plot = 32
            n_plot = min(n_plot, len(x_eval))
            x0 = x_eval[:n_plot]
            x0_t = None
            if cfg.device == "cuda":
                import torch

                x0_t = torch.from_numpy(x0).to(cfg.device)
            else:
                import torch

                x0_t = torch.from_numpy(x0)
            proj_eps_used = compute_eps_stop(model, x_train, eval_cfg)
            traj_t, _ = project_trajectory(model, x0_t, cfg, f_abs_stop=proj_eps_used)
            traj = traj_t.cpu().numpy()
            out_dir = run.dir
            plot_contour_and_trajectory(
                model,
                x_train,
                x0,
                traj,
                cfg,
                out_path=os.path.join(out_dir, f"{name}_data_contour.png"),
                title=f"{name}: {method}",
                grid=grid,
                zero_level_eps=proj_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
            )
            if wandb is not None:
                wandb.log(
                    {
                        f"{name}/data_contour": wandb.Image(
                            os.path.join(out_dir, f"{name}_data_contour.png")
                        )
                    }
                )

        avg = {}
        keys = results[0]["metrics"].keys()
        for k in keys:
            avg[k] = float(np.mean([r["metrics"][k] for r in results]))
        wandb.log({f"avg/{k}": v for k, v in avg.items()})
        if norm_vals:
            avg["norm_proj_manifold_dist"] = float(np.mean(norm_vals))
            wandb.log({"avg/norm_proj_manifold_dist": avg["norm_proj_manifold_dist"]})
            metrics_table.add_data(
                "avg",
                "norm_proj_manifold_dist",
                avg["norm_proj_manifold_dist"],
                method,
                cfg.seed,
            )
        for k, v in avg.items():
            metrics_table.add_data("avg", k, v, method, cfg.seed)
        wandb.log({"metrics_table": metrics_table})

        if _SWEEP_CONTEXT.get("results_path"):
            record = {
                "method": method,
                "params": overrides,
                "results": results,
                "avg": avg,
            }
            with open(_SWEEP_CONTEXT["results_path"], "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="sweep_config.json",
        help="Path to sweep JSON config.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    datasets = cfg_data.get("datasets", DEFAULT_DATASETS)
    shared = cfg_data.get("shared", {})
    methods_cfg = cfg_data.get("methods", {})
    wandb_cfg = cfg_data.get("wandb", {})
    if not methods_cfg:
        raise ValueError("methods config is empty")

    cfg_base = Config()
    _apply_overrides(cfg_base, shared)
    cfg_base.wandb_enable = False

    out_root = cfg_data.get("output_root", "outputs_levelset_datasets")
    sweep_dir = os.path.join(out_root, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_jsonl = os.path.join(sweep_dir, f"results_{stamp}.jsonl")

    wb_enable = bool(wandb_cfg.get("enable", False))
    if wb_enable and wandb is None:
        print("wandb not available; disable wandb logging.")
        wb_enable = False

    all_runs = []
    if wb_enable:
        for method, spec in methods_cfg.items():
            params = spec.get("params", {})
            sweep_config, tuning_keys = _build_wandb_sweep(method, params, shared)
            sweep_id = wandb.sweep(
                sweep_config,
                project=wandb_cfg.get("project", cfg_base.wandb_project),
                entity=wandb_cfg.get("entity", cfg_base.wandb_entity),
            )
            _SWEEP_CONTEXT.clear()
            _SWEEP_CONTEXT.update(
                {
                    "datasets": datasets,
                    "shared": shared,
                    "method": method,
                    "tuning_keys": tuning_keys,
                    "results_path": out_jsonl,
                }
            )
            count = wandb_cfg.get("count", None)
            if count is not None:
                wandb.agent(sweep_id, function=_wandb_train, count=int(count))
            else:
                wandb.agent(sweep_id, function=_wandb_train)
    else:
        for method, spec in methods_cfg.items():
            params = spec.get("params", {})
            grid = list(_param_grid(params)) if params else [{}]
            for p in grid:
                run = run_one(cfg_base, method, p, datasets)
                all_runs.append(run)
                with open(out_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(run) + "\n")

    if all_runs:
        summary_path = os.path.join(sweep_dir, f"summary_{stamp}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2)
        print(f"sweep done: {len(all_runs)} runs")
        print(f"results: {out_jsonl}")
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
