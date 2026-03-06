#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from collections import defaultdict
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Keep local W&B files under repo-root/wandb regardless of launch cwd.
os.environ["WANDB_DIR"] = os.path.join(_PROJECT_ROOT, "wandb")

from common.unified_experiment import VALID_METHODS, run_one

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

def _resolve_config_root(config_root: str) -> str:
    p = str(config_root).strip()
    if not p:
        p = "configs"
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    fallback = os.path.join(_PROJECT_ROOT, p)
    return fallback


def _resolve_outdir(outdir: str) -> str:
    p = str(outdir).strip()
    if not p:
        p = "outputs_unified"
    if os.path.isabs(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def _log_dataset_plots_to_wandb(*, outdir: str, method: str, dataset: str, step: int | None = None) -> None:
    method_dir = os.path.join(outdir, method)
    if not os.path.isdir(method_dir):
        return
    patterns = [
        os.path.join(method_dir, f"{dataset}*.png"),
        os.path.join(method_dir, f"{dataset}*.jpg"),
        os.path.join(method_dir, f"{dataset}*.jpeg"),
    ]
    plot_paths: list[str] = []
    for pat in patterns:
        plot_paths.extend(glob.glob(pat))
    if not plot_paths:
        return
    for p in sorted(set(plot_paths)):
        key = os.path.splitext(os.path.basename(p))[0]
        try:
            payload = {f"{dataset}/{method}/plot/{key}": wandb.Image(p)}
            if step is None:
                wandb.log(payload)
            else:
                wandb.log(payload, step=int(step))
        except Exception as e:
            print(f"[warn] wandb image log failed: {p} ({e})")


def _sanitize_wandb_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _sanitize_wandb_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_wandb_value(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _flatten_dict_for_wandb(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict_for_wandb(v, key))
        else:
            out[key] = _sanitize_wandb_value(v)
    return out


def _sync_resolved_config_to_wandb(
    *,
    wb_run: Any,
    method: str,
    dataset: str,
    seed: int | None,
    resolved_config: dict[str, Any],
    loaded_paths: list[str],
) -> None:
    prefix = f"resolved__{method}__{dataset}__seed{seed if seed is not None else 'none'}"
    payload = _flatten_dict_for_wandb(dict(resolved_config), f"{prefix}__cfg")
    payload[f"{prefix}__seed"] = seed
    payload[f"{prefix}__loaded_config_paths"] = [str(p) for p in loaded_paths]
    payload[f"{prefix}__loaded_layers_count"] = int(len(loaded_paths))
    try:
        wb_run.config.update(payload, allow_val_change=True)
    except Exception as e:
        print(f"[warn] wandb config sync failed for {method}/{dataset}/seed={seed}: {e}")


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _override_keys(overrides: list[str]) -> set[str]:
    keys: set[str] = set()
    for expr in overrides:
        if "=" not in str(expr):
            continue
        k = str(expr).split("=", 1)[0].strip()
        if k:
            keys.add(k)
    return keys


def _with_default_non_gif_overrides(overrides: list[str]) -> list[str]:
    out = list(overrides)
    keys = _override_keys(out)
    if "plan_save_gif" not in keys:
        out.append("plan_save_gif=false")
    if "plan_pybullet_render" not in keys:
        out.append("plan_pybullet_render=false")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run benchmark suite over methods and datasets")
    p.add_argument("--methods", default="eikonal", help="comma-separated methods")
    p.add_argument("--datasets", default="6d_workspace_sine_surface_pose", help="comma-separated datasets")
    p.add_argument("--seeds", default="10", help="comma-separated seeds; empty means no seed override")
    p.add_argument("--outdir", default="outputs_unified")
    p.add_argument("--config-root", default="configs")
    p.add_argument("--override", action="append", default=[], help="dotted key=value override")

    p.add_argument("--wandb-enable", action="store_true", default=True)
    p.add_argument("--wandb-project", default="equality_manifold_unified")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-run-name", default="")
    return p


def _metric_keys(results: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for r in results:
        keys.update(r.get("metrics", {}).keys())
    return sorted(keys)


def _write_per_case_csv(path: str, results: list[dict[str, Any]]) -> None:
    metric_cols = _metric_keys(results)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "seed", *metric_cols])
        for r in results:
            row = [r.get("method"), r.get("dataset"), r.get("seed")]
            m = r.get("metrics", {})
            row.extend([m.get(k, "") for k in metric_cols])
            w.writerow(row)


def _write_leaderboard_csv(path: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        grouped[str(r["method"])].append(r)

    metric_cols = _metric_keys(results)
    rows: list[dict[str, Any]] = []
    for method, items in grouped.items():
        agg: dict[str, Any] = {"method": method, "n_cases": len(items)}
        for k in metric_cols:
            vals = [it.get("metrics", {}).get(k) for it in items]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            agg[k] = (sum(vals) / len(vals)) if vals else ""
        rows.append(agg)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_cases", *metric_cols])
        for row in sorted(rows, key=lambda x: str(x["method"])):
            w.writerow([row["method"], row["n_cases"], *[row.get(k, "") for k in metric_cols]])
    return rows


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    args = _build_parser().parse_args()
    config_root = _resolve_config_root(str(args.config_root))
    outdir = _resolve_outdir(str(args.outdir))
    methods = _split_csv(args.methods)
    datasets = _split_csv(args.datasets)
    seeds = _split_csv(args.seeds)

    if not methods:
        raise ValueError("no methods provided")
    bad = [m for m in methods if m not in VALID_METHODS]
    if bad:
        raise ValueError(f"unsupported methods: {bad}; valid={sorted(VALID_METHODS)}")
    if not datasets:
        raise ValueError("no datasets provided")

    seed_values: list[int | None]
    if seeds:
        seed_values = [int(s) for s in seeds]
    else:
        seed_values = [None]

    os.makedirs(outdir, exist_ok=True)
    effective_overrides = _with_default_non_gif_overrides(args.override)
    partial_jsonl = os.path.join(outdir, "per_run_metrics.jsonl")
    partial_summary = os.path.join(outdir, "summary_metrics.partial.json")

    wb_run = None
    if args.wandb_enable:
        if wandb is None:
            print("[warn] wandb requested but not available; continuing without wandb")
        else:
            wb_run = wandb.init(
                project=str(args.wandb_project),
                entity=(str(args.wandb_entity) or None),
                name=(str(args.wandb_run_name) or None),
                config={
                    "methods": methods,
                    "datasets": datasets,
                    "seeds": seed_values,
                    "config_root": config_root,
                    "override": args.override,
                },
            )

    all_results: list[dict[str, Any]] = []
    step = 0
    total_per_method = {m: len(datasets) * len(seed_values) for m in methods}
    done_per_method = {m: 0 for m in methods}
    for method in methods:
        for dataset in datasets:
            for seed in seed_values:
                print(f"[run] method={method} dataset={dataset} seed={seed}")
                result, loaded = run_one(
                    method=method,
                    dataset=dataset,
                    out_root=outdir,
                    seed_override=seed,
                    config_root=config_root,
                    cli_overrides=effective_overrides,
                )
                result["seed"] = seed
                result["loaded_config_paths"] = loaded
                all_results.append(result)

                # Persist each finished run immediately so interrupted benchmarks
                # can still be recovered/aggregated.
                row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "metrics": result.get("metrics", {}),
                    "config": result.get("config", {}),
                    "loaded_config_paths": loaded,
                }
                _append_jsonl(partial_jsonl, row)
                with open(partial_summary, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

                m = result["metrics"]
                print(
                    f"[done] method={method} dataset={dataset} seed={seed} "
                    f"proj_dist={m.get('proj_manifold_dist', float('nan')):.6f} "
                    f"recall={m.get('pred_recall', float('nan')):.6f} "
                    f"FPrate={m.get('pred_FPrate', float('nan')):.6f}"
                )
                done_per_method[method] += 1
                print(
                    f"[progress] method={method} "
                    f"{done_per_method[method]}/{total_per_method[method]} completed"
                )
                if wb_run is not None:
                    step += 1
                    _sync_resolved_config_to_wandb(
                        wb_run=wb_run,
                        method=str(method),
                        dataset=str(dataset),
                        seed=(int(seed) if seed is not None else None),
                        resolved_config=dict(result.get("config", {})),
                        loaded_paths=list(loaded),
                    )
                    wandb.log({f"{dataset}/{method}/{k}": v for k, v in m.items()}, step=step)
                    _log_dataset_plots_to_wandb(
                        outdir=outdir,
                        method=str(method),
                        dataset=str(dataset),
                        step=step,
                    )

    summary_path = os.path.join(outdir, "summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    per_case_csv = os.path.join(outdir, "per_case_metrics.csv")
    leaderboard_csv = os.path.join(outdir, "leaderboard_mean_by_method.csv")
    _write_per_case_csv(per_case_csv, all_results)
    leaderboard = _write_leaderboard_csv(leaderboard_csv, all_results)

    print(f"[saved] {summary_path}")
    print(f"[saved] {partial_jsonl}")
    print(f"[saved] {partial_summary}")
    print(f"[saved] {per_case_csv}")
    print(f"[saved] {leaderboard_csv}")
    print("[leaderboard]", leaderboard)

    if wb_run is not None:
        for row in leaderboard:
            method = str(row["method"])
            payload = {f"avg/{method}/{k}": v for k, v in row.items() if k not in ("method", "n_cases") and isinstance(v, (int, float))}
            if payload:
                step += 1
                wandb.log(payload, step=step)
        wandb.finish()


if __name__ == "__main__":
    main()
