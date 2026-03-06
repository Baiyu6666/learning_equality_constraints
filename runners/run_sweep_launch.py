#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep local W&B files under repo-root/wandb regardless of launch cwd.
os.environ["WANDB_DIR"] = str(_PROJECT_ROOT / "wandb")

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create W&B sweep from yaml and immediately start agent.")
    p.add_argument("--yaml", default="configs/sweeps/ecomann_6d_workspace", help="path to sweep yaml")
    p.add_argument("--count", type=int, default=None, help="max trial count")
    p.add_argument("--project", default=None, help="override project")
    p.add_argument("--entity", default=None, help="override entity")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if wandb is None:
        raise RuntimeError("wandb is required but not installed")
    if yaml is None:
        raise RuntimeError("PyYAML is required but not installed")

    raw_yaml = str(args.yaml)
    cand = Path(raw_yaml)
    if cand.is_absolute():
        yaml_path = cand
    else:
        # Compatibility: allow running from arbitrary cwd.
        # Try cwd-relative first, then repo-root-relative (repo root = parent of runners/).
        cwd_path = (Path.cwd() / cand).resolve()
        repo_root = Path(__file__).resolve().parents[1]
        repo_path = (repo_root / cand).resolve()
        yaml_path = cwd_path if cwd_path.exists() else repo_path
    if not yaml_path.exists():
        raise FileNotFoundError(str(yaml_path))

    with open(str(yaml_path), "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)
    if not isinstance(sweep_cfg, dict):
        raise ValueError("sweep yaml root must be a dict")

    project = args.project or sweep_cfg.get("project")
    entity = args.entity or sweep_cfg.get("entity")

    sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=entity)
    print(f"[sweep] created sweep_id={sweep_id}")
    wandb.agent(sweep_id, project=project, entity=entity, count=args.count)


if __name__ == "__main__":
    main()
