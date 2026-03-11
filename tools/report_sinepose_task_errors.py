#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.abspath(os.path.join(repo, path))


def _load_summary(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(summary: dict[str, Any]) -> str:
    task = str(summary.get("task", "unknown"))
    pm = float(summary.get("mean_position_error", float("nan")))
    ps = float(summary.get("std_position_error", float("nan")))
    om = float(summary.get("mean_orientation_error_deg", float("nan")))
    osd = float(summary.get("std_orientation_error_deg", float("nan")))
    return (
        f"{task}: "
        f"position={pm:.6f} ± {ps:.6f}, "
        f"orientation(deg)={om:.4f} ± {osd:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Report mean/std position and orientation errors for sinepose tasks.")
    p.add_argument(
        "--obstacle-summary",
        default="outputs_unified/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/eikonal/sinepose_obstacle_paper/sinepose_planning_summary.json",
    )
    p.add_argument(
        "--waypoints-summary",
        default="outputs_unified/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/eikonal/sinepose_waypoints_paper/sinepose_waypoints_summary.json",
    )
    args = p.parse_args()

    obs = _load_summary(_resolve(args.obstacle_summary))
    wps = _load_summary(_resolve(args.waypoints_summary))
    print(_fmt(obs))
    print(_fmt(wps))


if __name__ == "__main__":
    main()
