# 目录结构说明 (Directory Structure)

本文档描述 `/root/autodl-tmp` 重构清理后的结构（2026-07-12）。

## 重组目标

1. **保留 openpi 官方仓库**不动（`openpi/`）
2. **保留所有 low-NFE 实现**：`smfVLA/`, `snapflow/`, `freeflow/`, `dmf/`
3. **共享数据集与 checkpoint**：`datasets/` 和 `checkpoints/` 在根目录集中管理
4. **统一评估框架**：`eval/scripts/run_eval.py` 为唯一入口
5. **删除冗余**：历史日志、一次性脚本、旧式 eval 路径、DMF PyTorch 参考实现等

## 目录结构

```
autodl-tmp/
├── openpi/                     # pi0.5 官方框架（共享，所有方法依赖）
│
├── smfVLA/                     # SMF (SplitMeanFlow) 方法
│   ├── configs/train/
│   ├── scripts/                # run_train.py, train.sh
│   ├── src/smf_vla/
│   ├── third_party/openpi      # -> /root/autodl-tmp/openpi (symlink)
│   ├── data                    # -> ../datasets/libero (symlink)
│   ├── CLAUDE.md, README.md
│   └── pyproject.toml
│
├── snapflow/                   # SnapFlow 方法
│   ├── configs/train/
│   ├── scripts/                # run_train.py, train.sh
│   ├── src/snapflow/
│   ├── third_party/openpi      # -> /root/autodl-tmp/openpi (symlink)
│   └── pyproject.toml
│
├── freeflow/                   # FreeFlow 方法
│   ├── configs/train/
│   ├── scripts/                # train.sh, eval_libero_plus_quick_all.sh
│   ├── src/freeflow/
│   ├── third_party/openpi      # -> ../../openpi (symlink)
│   ├── CLAUDE.md, README.md
│   └── pyproject.toml
│
├── dmf/                        # DMF (Decoupled MeanFlow) 方法
│   ├── configs/train/
│   ├── scripts/                # run_train.py, train.sh, convert_calvin_data_to_lerobot.py
│   ├── src/dmf_vla/
│   └── README.md
│
├── eval/                       # 统一评估框架
│   └── scripts/
│       ├── run_eval.py         # 统一入口: --dataset {libero,libero-plus,calvin}
│       ├── eval_utils.py       # 核心: load_policy() + detect_checkpoint_type()
│       ├── eval_direct.py      # LIBERO 标准
│       ├── eval_libero_plus.py # LIBERO-Plus 鲁棒性
│       ├── eval_calvin.py      # CALVIN (debug/partial)
│       ├── eval_calvin_benchmark.py + calvin_official_protocol.py  # CALVIN 官方协议
│       ├── calvin_utils.py     # CALVIN 共享工具
│       └── activate_calvin_env.sh
│   └── results/                # 评估结果 JSON
│
├── datasets/                   # 共享数据集
│   ├── libero/
│   ├── libero-plus/
│   ├── libero-plus-training/
│   └── calvin*/, calvin_lerobot*/
│
├── checkpoints/                # 共享 checkpoint
│   ├── smf_base/pi05_libero/   # 所有方法的 finetune 基座
│   ├── snapflow_finetuned/
│   └── dmf_finetuned_calvin/
│
├── scripts/                    # 通用转换工具
│   ├── convert_pytorch_to_jax.py
│   └── convert_calvin_to_jax.py
│
├── results/                    # 历史评估结果 JSON（实验记录）
├── docs/                        # 文档
│   ├── directory_structure.md
│   ├── evaluation.md
│   └── 20260602_154947_smf_base_training_plan.md
│
├── AGENTS.md                   # OpenCode agent 指南
└── CLAUDE.md                   # 完整架构参考
```

## 注意事项

1. **单一 conda 环境**：所有训练和评估使用 `openpi_server`
   (`/root/miniconda3/envs/openpi_server`)
2. **`openpi/` 是共享的**：不要在方法的 `third_party/openpi` symlink 下编辑，直接改
   `openpi/`
3. **评估无需手动设置 PYTHONPATH**：`eval_utils.setup_paths()` 自动注入所有方法
   `src/` 路径
4. **代码风格**：方法目录和 `eval/` 用 line-length 100；`openpi/` 用 120
