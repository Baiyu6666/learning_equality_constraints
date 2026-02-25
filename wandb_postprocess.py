import numpy as np
import wandb

entity = "pby"
project = "equality constraint learning"
sweep_id = "houfiwds"

env_keys = [
    "figure_eight/proj_manifold_dist",
    "ellipse/proj_manifold_dist",
    "noise_only/proj_manifold_dist",
    "sparse_only/proj_manifold_dist",
    "hetero_noise/proj_manifold_dist",
    "looped_spiro/proj_manifold_dist",
]

api = wandb.Api()

# 优先用过滤器；如果失败就用手动过滤
try:
    runs = api.runs(f"{entity}/{project}", filters={"sweep": sweep_id})
except Exception:
    runs = [r for r in api.runs(f"{entity}/{project}") if r.sweep and r.sweep.id == sweep_id]

rows = []
for k in env_keys:
    vals = []
    for r in runs:
        v = r.summary.get(k, None)
        if v is not None:
            vals.append(v)
    if vals:
        rows.append((k.split("/")[0], float(np.mean(vals)), float(np.std(vals))))

# 上传汇总表
wandb.init(project=project, entity=entity, name=f"sweep_{sweep_id}_env_summary")
table = wandb.Table(columns=["env", "mean", "std"])
for r in rows:
    table.add_data(*r)
wandb.log({f"sweep_{sweep_id}_env_summary": table})
wandb.finish()
