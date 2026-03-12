# EcoMaNN 原项目参数整理（按源码）

本文件按你导入的原仓库源码整理：  
`smp_manifold_learning-master`

主要来源：
- `smp_manifold_learning/scripts/train_ecmnn.py`
- `smp_manifold_learning/scripts/run_accuracy_experiment_ecmnn.bash`
- `smp_manifold_learning/scripts/run_ablation_test_ecmnn.bash`
- `smp_manifold_learning/scripts/run_aug_variation_test_ecmnn.bash`
- `smp_manifold_learning/scripts/proj_ecmnn.py`
- `smp_manifold_learning/scripts/compute_proj_ecmnn_stats.py`
- `README.md`

## 1. `train_ecmnn.py` 的 CLI 默认参数

| 参数 | 默认值 |
|---|---|
| `-d/--dataset_option` | `1` |
| `-u/--is_performing_data_augmentation` | `1` |
| `-s/--is_optimizing_signed_siamese_pairs` | `1` |
| `-a/--is_aligning_lpca_normal_space_eigvecs` | `1` |
| `-c/--is_augmenting_w_rand_comb_of_normaleigvecs` | `1` |
| `-r/--rand_seed` | `38` |
| `-p/--plot_save_dir` | `../plot/ecmnn/` |
| `-v/--aug_dataloader_save_dir` | `../plot/ecmnn/` |
| `-l/--is_using_logged_aug_dataloader` | `0` |
| `-n/--is_dataset_noisy` | `0` |
| `-m/--siam_mode` | `all` |
| `-t/--N_normal_space_traversal_sphere` | `9` |

`siam_mode` 可选：`all / no_siam_reflection / no_siam_same_levelvec / no_siam_frac_aug`

## 2. `train_ecmnn(...)` 函数默认超参数（未被 dataset 分支覆盖前）

| 参数 | 默认值 |
|---|---|
| `initial_learning_rate` | `0.001` |
| `weight_decay` | `0.0` |
| `num_epochs` | `15`（但主脚本里几乎都覆盖） |
| `batch_size` | `128` |
| `device` | `cpu`（若传 `gpu` 且可用则用 `cuda:0`） |
| `hidden_sizes` | `[14,11,9,7,5,3]`（主脚本会覆盖） |
| `N_normal_space_traversal` | `9` |
| `clean_aug_data` | `True` |
| `is_optimizing_signed_siamese_pairs` | `True` |
| `is_aligning_lpca_normal_space_eigvecs` | `True` |
| `is_augmenting_w_rand_comb_of_normaleigvecs` | `True` |
| `N_local_neighborhood_mult` | `1` |

训练优化器固定为：`RMSprop(lr=initial_learning_rate, weight_decay=weight_decay)`。

## 3. `ECMNNDatasetLoader` 里的关键默认

| 参数 | 默认值 |
|---|---|
| `N_normal_space_traversal` | `9` |
| `N_siam_same_levelvec` | `5` |
| `clean_aug_data` | `True` |
| `aug_clean_thresh` | `1e-1` |
| `N_normal_space_eigvecs_alignment_repetition` | `1` |
| `N_local_neighborhood_mult` | `1` |

关键计算规则：
- 局部邻域大小：`N_local_neighborhood = N_local_neighborhood_mult * 2 * (2^dim_ambient)`
- 切/法空间维度：由 Local PCA 特征值“最大跌落”模式自动估计
- 增强步长：`epsilon = N_local_neighborhood_mult * sqrt(mean_tangent_eigval)`

## 4. 按 dataset 的参数（`dataset_option` 分支）

> 下表是 `train_ecmnn.py` 主逻辑里实际设置。

| dataset_option | 数据集路径 | `model_name` | `num_epochs` | `hidden_sizes` | `N_local_neighborhood_mult` | `N_normal_space_traversal` | 其他 |
|---|---|---|---:|---|---:|---:|---|
| `1` | `../data/synthetic/unit_sphere_random` | `model_3d_sphere` | `25` | `[36,24,18,10]` | 正常 `1`；noisy 时 `2` | 正常 `N_normal_space_traversal_sphere`(默认9)；noisy 时 `8` | `dims_cross_section=[1,2]`；`is_plotting=True`；会动态生成数据：`N_data=N_local_neighborhood_mult*5000` |
| `2` | `../data/synthetic/unit_circle_loop_random` | `model_3d_circle_loop` | 正常 `25`；noisy `50` | `[36,24,18,10]` | 正常 `1`；noisy `3` | 正常 `9`；noisy `5` | `dims_cross_section=[1,2]`；`is_plotting=True`；会动态生成数据：`N_data=N_local_neighborhood_mult*1000` |
| `3` | `../data/trajectories/3dof_v2_traj` | `model_3dof_traj` | `25` | `[36,24,18,10]` | `1` | `5` | `dims_cross_section=[1,2]`；`is_plotting=True` |
| `4` | `../data/trajectories/6dof_traj` | `model_6dof_traj` | `25` | `[36,24,18,10]` | `1` | `2` | 默认不画 cross-section |
| `5` | `../data/trajectories/nav_dataset_on` | `model_nav_dataset_on` | `3` | 沿用前值（脚本未重设） | `1` | `2` | 注释标记为 inequality，N/A |
| `6` | `../data/trajectories/rotation_ineq_traj` | `model_tilt` | `3` | 沿用前值（脚本未重设） | `1` | `2` | 注释标记为 inequality，且提到 Jacobian SVD 可能失败 |
| `7` | `../data/trajectories/jaco_handover_traj` | `model_handover` | `20` | 沿用前值（脚本未重设） | `1` | `2` |  |
| `8` | `../data/trajectories/pybullet_pouring_ik_solutions_from_left_clean` | `pouring` | `100` | 沿用前值（脚本未重设） | `1` | `2` |  |

说明：
- 噪声只在 `dataset_option 1/2` 分支里显式处理（`is_dataset_noisy` 控制）。
- `hidden_sizes` 在主脚本里先统一改成 `[36,24,18,10]`。

## 5. 原项目脚本里的“按实验模式覆盖”

### 5.1 `run_accuracy_experiment_ecmnn.bash`
- 数据集：`1,2,3,4`
- seed：`1..3`
- 调用：`python3 train_ecmnn.py -d <manifold_id> -r <seed> -l 1 ...`
- 即：使用 `train_ecmnn.py` 的“正常模式”默认（augmentation/siamese/alignment 都开）。

### 5.2 `run_ablation_test_ecmnn.bash`
- 数据集：`1,2,3,4`
- seed：`0..2`
- expmode 运行：`0,1,3,4,5,6,7,8`（脚本里跳过了 2）

expmode 与开关映射：
- `0 normal`: `u=1,c=1,s=1,a=1,n=0,m=all`
- `1 wo_augmentation`: `u=0,c=1,s=1,a=0,n=0,m=all`
- `3 wo_siamese_losses`: `u=1,c=1,s=0,a=1,n=0,m=all`
- `4 wo_nspace_alignment`: `u=1,c=1,s=1,a=0,n=0,m=all`
- `5 noisy_normal`: `u=1,c=1,s=1,a=1,n=1,m=all`
- `6 no_siam_reflection`: `u=1,c=1,s=1,a=1,n=0,m=no_siam_reflection`
- `7 no_siam_frac_aug`: `u=1,c=1,s=1,a=1,n=0,m=no_siam_frac_aug`
- `8 no_siam_same_levelvec`: `u=1,c=1,s=1,a=1,n=0,m=no_siam_same_levelvec`

### 5.3 `run_aug_variation_test_ecmnn.bash`
- 只跑 `manifold_id=1`（3Dsphere）
- expmode=0（normal）
- 关键 sweep：`-t bt`, `bt=1..7`，即 sweep `N_normal_space_traversal_sphere`
- seed：`0..2`

## 6. Projection 评估脚本参数

### 6.1 `proj_ecmnn.py`（函数 `eval_projection`）
常用参数：
- `n_data_samples`
- `tolerance`（on-manifold 判定阈值）
- `step_size`（Projection 迭代步长）
- `extrapolation_factor`
- `hidden_sizes`
- `log_dir`（模型路径）

脚本 main 里的示例默认（非论文批评估）：
- `n_data_samples=2000`
- `tolerance=1e-1`
- `step_size=0.25`
- `extrapolation_factor=1.0`
- `hidden_sizes=[36,24,18,10]`

### 6.2 `compute_proj_ecmnn_stats.py`
默认设置：
- `n_data_samples=100`
- `tolerance=1e-1`
- `step_size=0.25`
- `extrapolation_factor=1.0`
- `epoch=25`
- 对 `dataset_option=4`（6dof_traj）评估时，`hidden_sizes=[128,64,32,16]`；其余 `[36,24,18,10]`
- expmode 统计：`[0,1,3,4,5,6,7,8]`

## 7. README 里与实验流程相关的补充

README 明确提到：
- ablation 跑完后需要**手动挑 best checkpoint**（通过 log 选择 epoch，然后复制到 `best/model_xxx.pth`）
- 然后再执行 `compute_proj_ecmnn_stats.py` 做投影评估统计

---

如果你需要，我可以再补一个“对照你当前统一框架配置名”的映射表（比如把 `-u/-s/-a/-c/-m/-t` 映射到你 `configs/methods/ecomann.json` 的字段），方便你直接做 dataset-specific config。  
