# π0.5 Few-NFE VLA Distillation Research

基于 [openpi](https://github.com/Physical-Intelligence/openpi) 框架的少步数（Few-NFE）视觉-语言-动作（VLA）模型研究仓库。所有方法均以 π0.5 为教师模型，通过蒸馏 / 一致性训练将多步去噪（NFE=10）压缩至 1 步推理（1-NFE），并在 LIBERO-Plus 和 CALVIN 基准上评测鲁棒性。

## 仓库总览

```
/root/autodl-tmp/
├── openpi/                 # 共享 π0.5 框架（所有方法 + 评测依赖）
├── smfVLA/                 # 方法 1：SplitMeanFlow（SMF）
├── snapflow/               # 方法 2：SnapFlow（自蒸馏，继承 SMF）
├── freeflow/               # 方法 3：FreeFlow（无数据蒸馏）
├── dmf/                    # 方法 4：Decoupled MeanFlow（DMF）
├── piflow/                 # 方法 5：π-Flow（GMM 策略蒸馏）
├── eval/                   # 统一评测框架（LIBERO-Plus + CALVIN）
├── checkpoints/            # 模型权重（gitignored，仅本地存在）
├── datasets/               # 数据集（gitignored，仅本地存在）
├── docs/                   # 实验记录与报告模板
├── nvidia_libs/            # NVIDIA CUDA 库（gitignored）
├── AGENTS.md               # AI agent 简要指引
└── CLAUDE.md               # （旧版 agent 指引，已被 AGENTS.md 取代）
```

## 共享框架：`openpi/`

Physical Intelligence 开源 π0/π0.5 VLA 框架，所有方法均继承自其中的 `Pi0` 基类。**直接编辑 `openpi/` 目录，不要通过各方法的 `third_party/openpi` 符号链接编辑。**

```
openpi/
├── src/openpi/
│   ├── models/             # JAX/NNX 模型
│   │   ├── pi0.py           # ← Pi0 / Pi05 基类（所有方法的父类）
│   │   ├── pi0_config.py    # 模型配置 dataclass
│   │   ├── gemma.py         # Gemma LLM（nn.scan + 逐层 adaRMS）
│   │   ├── siglip.py         # SigLIP 视觉编码器
│   │   ├── vit.py            # ViT
│   │   └── model.py          # restore_params、Observation 等
│   ├── models_pytorch/      # PyTorch 变体（CALVIN 评测使用）
│   ├── policies/            # 策略封装
│   │   ├── policy.py        # ← Policy 类（.infer() 推理入口）
│   │   ├── policy_config.py # create_trained_policy
│   │   └── libero_policy.py
│   ├── transforms.py        # 观测/动作变换（Normalize、Resize 等）
│   ├── training/
│   │   ├── config.py        # 训练配置（get_config("pi05_libero")）
│   │   ├── checkpoints.py   # checkpoint 保存/加载、norm_stats
│   │   ├── optimizer.py     # AdamW + cosine schedule
│   │   └── sharding.py      # JAX 分片
│   └── shared/              # image_tools、nnx_utils、normalize
├── packages/openpi-client/  # 客户端包（加入 PYTHONPATH）
├── examples/                # 含 libero 示例
├── third_party/libero/      # LIBERO 环境（评测用）
└── pyproject.toml           # line-length 120（与方法目录的 100 不同）
```

## 五个方法目录

每个方法共享相同的 `src/<pkg>/training/` 四文件核心结构：

| 文件 | 作用 |
|------|------|
| `jax_trainer.py` | JIT 训练循环（AdamW + cosine + warmup；DMF/Pi-Flow 额外用 EMA） |
| `<method>_loss.py` | 方法特定的损失函数 |
| `freeze_utils.py` | glob 模式冻结/训练掩码（均冻结 VLM backbone，训练 `*_1` action-expert 层） |
| `data_loader.py` | LeRobot v2.0 Parquet 数据加载器 |

### `smfVLA/` — SplitMeanFlow

- **包名：** `smf_vla`
- **模型：** `Pi05SMF`（继承 `openpi.models.pi0.Pi0`）
- **核心思想：** 训练模型预测 *平均速度* `u(z_t, r, t)`，1-NFE 推理：`z_0 = z_1 - u(z_1, 0, 1)`
- **损失：** SMF（自一致性，r<t）+ FM（Bernoulli p=0.3，r=t）
- **数据集：** `datasets/libero`
- **文档：** `smfVLA/CLAUDE.md`、`smfVLA/README.md`

```
smfVLA/
├── src/smf_vla/
│   ├── models/pi05_smf.py        # Pi05SMF 模型
│   └── training/                 # jax_trainer、smf_loss、freeze_utils、data_loader
├── configs/train/smf_base_libero.yaml
├── scripts/train.sh + run_train.py
└── third_party/openpi → /root/autodl-tmp/openpi  (符号链接)
```

### `snapflow/` — SnapFlow

- **包名：** `snapflow`
- **模型：** `Pi05SnapFlow`（继承 **`Pi05SMF`**，依赖 smfVLA）
- **核心思想：** 自蒸馏 2-step Euler 快捷一致性损失
- **损失：** `L = α·L_FM + (1-α)·λ·L_shortcut`
- **数据集：** `datasets/libero`
- **文档：** `snapflow/CLAUDE.md`、`snapflow/README.md`

```
snapflow/
├── src/snapflow/
│   ├── models/
│   │   ├── pi05_snapflow.py     # Pi05SnapFlow(Pi05SMF)
│   │   └── target_time_mlp.py   # 零初始化 2 层 MLP
│   └── training/                # snapflow_loss（3 次前向）
├── configs/train/snapflow_libero.yaml
├── scripts/train.sh + run_train.py
└── third_party/openpi → /root/autodl-tmp/openpi  (符号链接)
```

### `freeflow/` — FreeFlow

- **包名：** `freeflow`
- **模型：** `Pi05FreeFlow`（继承 `openpi.models.pi0.Pi0`）
- **核心思想：** 无数据蒸馏——学生从先验采样，学习教师的多步积分路径
- **损失：** `L = L_path + λ·L_correction`
- **数据集：** `datasets/libero`
- **文档：** `freeflow/CLAUDE.md`、`freeflow/README.md`

```
freeflow/
├── src/freeflow/
│   ├── models/
│   │   ├── pi05_freeflow.py     # Pi05FreeFlow
│   │   └── teacher_wrapper.py  # 冻结 π0.5 教师（NFE=10 Euler）
│   └── training/                # freeflow_loss、run_train.py（入口在 training/ 内）
├── configs/train/               # 5 个配置（base、plus、full、no_correction、backup）
├── scripts/train.sh + eval 脚本
└── third_party/openpi → ../openpi  (相对符号链接)
```

### `dmf/` — Decoupled MeanFlow

- **包名：** `dmf_vla`
- **模型：** `Pi05DMF`（继承 `openpi.models.pi0.Pi0`）
- **核心思想：** 将 Transformer 解耦为编码器（前 2/3 层，条件 t）和解码器（后 1/3，条件 r），`dmf_depth_ratio=0.67`
- **损失：** `L = 0.5·(L_FM + L_MF)`，L_MF 使用 JVP 计算 `du/dt`
- **特殊：** `logvar_proj` 学习方差；EMA decay=0.9999；无 `third_party/openpi`（靠 PYTHONPATH）
- **数据集：** `datasets/libero-plus-training`
- **文档：** `dmf/README.md`

```
dmf/
├── src/dmf_vla/
│   ├── models/pi05_dmf.py       # Pi05DMF + 3D adaRMS_cond
│   ├── inference/dmf_sampler.py # 独立采样器（测试用，eval 用模型内置 sample_actions）
│   └── training/                # dmf_loss（JVP）、jax_trainer（EMA）
├── configs/train/               # dmf_libero_plus.yaml（默认）、dmf_libero.yaml、dmf_calvin.yaml
├── scripts/train.sh + run_train.py
└── （无 third_party/openpi 符号链接）
```

### `piflow/` — π-Flow

- **包名：** `piflow_vla`
- **模型：** `Pi05PiFlow`（继承 `openpi.models.pi0.Pi0`）
- **核心思想：** 学生预测 GMM 策略参数，解析式 GMFlow rollout 在推理时**零**教师调用
- **三个零初始化头：** `gmm_mean_proj`、`gmm_logstd_proj`、`gmm_logweight_proj`（K=8 分量）
- **损失：** L2 速度模仿蒸馏（pi-ID），无 JVP/GAN/辅助网络
- **特殊：** EMA decay=0.9999；无 `third_party/openpi`（靠 PYTHONPATH）
- **数据集：** `datasets/libero-plus-training`
- **文档：** `piflow/README.md`

```
piflow/
├── src/piflow_vla/
│   ├── models/
│   │   ├── pi05_piflow.py      # Pi05PiFlow + GMM 头
│   │   └── gmflow.py           # 解析式 GMFlow 速度 + Euler rollout
│   ├── inference/piflow_sampler.py  # 独立采样器
│   └── training/                # piflow_loss、jax_trainer（双模型 JIT）
├── tests/test_gmflow.py        # 18 个单元测试
├── configs/train/piflow_libero_plus.yaml
└── scripts/train.sh + run_train.py
```

## 方法对照表

| 方法 | 包名 | 模型类 | 继承自 | `third_party/openpi` | 训练数据 | Checkpoint 输出 |
|------|------|--------|--------|----------------------|----------|----------------|
| SMF | `smf_vla` | `Pi05SMF` | `Pi0` | 绝对符号链接 | `datasets/libero` | 使用 base checkpoint |
| SnapFlow | `snapflow` | `Pi05SnapFlow` | `Pi05SMF` | 绝对符号链接 | `datasets/libero` | `checkpoints/snapflow_finetuned/step_N` |
| FreeFlow | `freeflow` | `Pi05FreeFlow` | `Pi0` | 相对符号链接 | `datasets/libero` | `freeflow/checkpoints/finetuned/freeflow/step_N` |
| DMF | `dmf_vla` | `Pi05DMF` | `Pi0` | 无（PYTHONPATH） | `datasets/libero-plus-training` | `checkpoints/dmf_finetuned/step_N` |
| Pi-Flow | `piflow_vla` | `Pi05PiFlow` | `Pi0` | 无（PYTHONPATH） | `datasets/libero-plus-training` | `checkpoints/piflow_finetuned/step_N` |

所有方法的共享微调基座：`checkpoints/pi05_libero/`。

## 评测框架：`eval/`

```
eval/
├── common/                     # 跨基准共享
│   ├── policy_loader.py        # ← load_policy()：pi05/dmf/piflow 统一加载
│   ├── constants.py            # 路径常量、MAX_STEPS_MAP
│   └── utils.py                # setup_paths()、结果 JSON 工具
├── libero_plus/                # LIBERO-Plus 鲁棒性评测
│   ├── main.py                 # 入口：python -m eval.libero_plus.main
│   ├── presets.py              # quick/normal/fullset 模式定义
│   └── runner.py               # 扰动采样 + episode 循环
├── calvin/                     # CALVIN ABCD→D 基准（仅 pi0.5 PyTorch）
│   ├── main.py                 # 入口：python -m eval.calvin.main
│   ├── protocol.py
│   └── runner.py
├── configs/                    # LIBERO 各 suite 的任务配置 YAML
├── scripts/
│   └── run_libero_parallel.sh  # 并行评测启动脚本
└── results/                    # 时间戳命名的 JSON 结果
```

### 模型类型与自动检测

`--model-type` 支持 `pi05`、`dmf`、`piflow`。NFE 支持 1/2/4/10。

`detect_checkpoint_type()` 通过扫描 checkpoint 参数键名自动识别类型：
- `gmm_mean_proj` → piflow
- `logvar_proj` → dmf
- 其他 → pi05

### 评测模式

| 基准 | 模式 | 内容 |
|------|------|------|
| LIBERO-Plus | `quick` | 1 suite（spatial），10 tasks，5 ep |
| LIBERO-Plus | `normal` | 4 suites，扰动采样，5 ep，<10h |
| LIBERO-Plus | `fullset` | 5 suites，全部 tasks，50 ep |
| CALVIN | `quick` | debug 数据集，5 seqs |
| CALVIN | `normal` | ABCD，100 seqs |
| CALVIN | `fullset` | ABCD，1000 seqs |

## 环境配置

- **统一环境：** `openpi_server`，解释器位于 `/root/miniconda3/envs/openpi_server/bin/python`，含 `jax==0.5.3` + `libero`
- **WandB：** 训练前须 `export WANDB_API_KEY=...`，否则静默跳过（仅警告）
- **CALVIN 评测：** 使用独立 `calvin_eval` 环境，通过 `eval/scripts/activate_calvin_env.sh` 激活

## 训练

```bash
cd /root/autodl-tmp/<method> && bash scripts/train.sh                              # 默认配置
bash scripts/train.sh configs/train/<config>.yaml --resume <ckpt>                 # 恢复训练
```

- 每个 `train.sh` 自动设置 `PYTHONPATH`（方法 `src/` + openpi `src/` + client）并激活 `openpi_server`
- **所有方法冻结 VLM backbone**；仅训练 action-expert 层（后缀 `*_1`）、投影层和各方法新增的时间嵌入头

## 评测

```bash
# LIBERO-Plus
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type pi05 --nfe 1 --mode quick
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type dmf --nfe 10 --mode normal
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode quick

# CALVIN（仅 pi0.5 PyTorch）
cd /root/autodl-tmp && python -m eval.calvin.main --model-type pi05 --nfe 1 --mode quick
cd /root/autodl-tmp && python -m eval.calvin.main --model-type pi05 --nfe 10 --mode normal
```

结果保存至 `eval/results/<model_type>/` 下时间戳命名的 JSON 文件。

## 数据集与 Checkpoint

均被 gitignore，仅本地存在：

| 数据集 | 路径 | 用途 |
|--------|------|------|
| LIBERO（标准） | `datasets/libero/` | SMF、SnapFlow、FreeFlow 训练 |
| LIBERO-Plus（扰动） | `datasets/libero-plus/` | LIBERO-Plus 评测 |
| LIBERO-Plus（训练版） | `datasets/libero-plus-training/` | DMF、Pi-Flow 训练 |
| CALVIN | `datasets/calvin/`、`datasets/calvin_lerobot/` | CALVIN 评测 |
| CALVIN D→D | `datasets/calvin_D-D/` | CALVIN D→D 子集 |

| Checkpoint | 路径 | 说明 |
|-----------|------|------|
| π0.5 LIBERO base | `checkpoints/pi05_libero/` | 所有方法的共享微调基座 |
| π0.5 CALVIN (PyTorch) | `checkpoints/pi05_calvin_pt/` | CALVIN PyTorch 评测 |
| DMF finetuned | `checkpoints/dmf_finetuned/step_N` | DMF 微调结果 |
| Pi-Flow finetuned | `checkpoints/piflow_finetuned/step_N` | Pi-Flow 微调结果 |

## 代码风格

方法目录（`smfVLA`、`snapflow`、`freeflow`、`dmf`、`piflow`）和 `eval/` 使用：

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

> **注意：** `openpi/` 使用 line-length 120，不要将其格式化为 100。

## 深入文档

- `smfVLA/CLAUDE.md` — SMF 算法细节与 7 种训练变体
- `snapflow/CLAUDE.md` — SnapFlow 自蒸馏损失公式
- `freeflow/CLAUDE.md` — FreeFlow 无数据蒸馏与 LIBERO-Plus 扰动维度
- `dmf/README.md` — DMF 编解码器解耦与 JVP 损失
- `piflow/README.md` — π-Flow GMM 策略蒸馏（最详细，252 行）
- `docs/experiment_workflow.md` — 实验记录工作流
- `docs/experiment_template.md` — 实验报告模板
- `docs/experiments/` — 已保存的实验报告
