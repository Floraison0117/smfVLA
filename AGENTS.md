# AGENTS.md

Compact guide for OpenCode agents. For the full architecture deep-dive
(methods, loss tables, per-head differences) read `CLAUDE.md` first — it is
accurate and authoritative. This file captures only what is easy to miss.

## Environment

- **One env for everything:** `openpi_server`. Interpreter at
  `/root/miniconda3/envs/openpi_server/bin/python`. Has `jax==0.5.3` + `libero`.
- `WANDB_API_KEY` must be exported before training or WandB is silently skipped
  (train scripts only warn). CALVIN eval uses a separate `calvin_eval` env via
  `eval/scripts/activate_calvin_env.sh`; do not assume it for normal work.

## Repo layout gotchas

- `openpi/` (top-level) is the **shared** pi0.5 framework used by all four
  methods and by eval. It is NOT a per-method subdirectory.
- Each method's `<method>/third_party/openpi` is a **symlink** to
  `/root/autodl-tmp/openpi` (DMF has none and uses `../openpi` instead). Never
  edit code through these symlinks — edit `openpi/` directly.
- Method package names differ: `smf_vla`, `snapflow`, `freeflow`, `dmf_vla`
  (under `<method>/src/`). SnapFlow's model subclasses SMF's `Pi05SMF`.
- Shared finetune base for all methods: `checkpoints/smf_base/pi05_libero/`.

## Training

```bash
cd /root/autodl-tmp/<method> && bash scripts/train.sh                              # default config
bash scripts/train.sh configs/train/<config>.yaml --resume <ckpt>                 # resume
```

- Each `train.sh` sets `PYTHONPATH` (method `src/` + openpi `src/` + client) and
  activates `openpi_server` itself — run it, don't invoke `run_train.py` bare.
- DMF's default config is `configs/train/dmf_libero_plus.yaml`; DMF trains on
  `datasets/libero-plus-training`, the other three on `datasets/libero`.
- All methods **freeze the VLM backbone**; only action-expert layers (suffix
  `*_1`), projection layers, and each method's new time-embedding heads train.
- Finetuned checkpoints land in: SnapFlow `checkpoints/snapflow_finetuned/step_N`,
  FreeFlow `freeflow/checkpoints/finetuned/freeflow/step_N`, DMF
  `checkpoints/dmf_finetuned/step_N`. SMF eval uses the base checkpoint.

## Evaluation

```bash
cd /root/autodl-tmp/eval/scripts
python run_eval.py --dataset libero      --mode preset --nfe 1 --model-type <m> [--checkpoint <c>]
python run_eval.py --dataset libero-plus --mode quick  --nfe 1 --model-type <m>
python run_eval.py --dataset calvin      --calvin-dataset debug --nfe 1 --model-type <m>
```

- `eval/scripts/eval_utils.py` is the core. Its `setup_paths()` injects all four
  method `src/` dirs + openpi into `sys.path`, so no manual PYTHONPATH is needed
  for eval.
- `detect_checkpoint_type()` **auto-sniffs the method from checkpoint param keys**
  (`t_time_mlp`+`r_time_mlp`→DMF, `target_time_mlp`→SnapFlow, `time_proj`→SMF,
  `time_mlp_in`→FreeFlow). `--model-type` is usually optional.
- NFE = sampling `num_steps`; all methods support 1/2/4/10.
- Two checkpoint formats: standard (`ckpt/params/`) and FreeFlow (flat, with
  `_METADATA` at root) — both handled by `eval_utils`.
- **`preset` mode = 5 episodes/task, NOT 50.** Only `fullset`/`full` use 50.
- LIBERO-Plus loads the perturbed `libero-plus` package, not vanilla `libero`.
- **CALVIN eval (`eval_calvin.py`) is debug/partial** — full sequences and
  task_oracle are not implemented. Treat CALVIN results as untrusted.
- Results → `eval/results/<model_type>/...` as timestamped JSON; each records its
  checkpoint in `metadata.checkpoint`.

## Code style

Method dirs (`smfVLA`, `snapflow`, `freeflow`, `dmf`) and `eval/` use:

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

Note: `openpi/` itself uses line-length 120 — do not reformat openpi files to 100.

## Deeper docs

- `CLAUDE.md` (root) — full method/loss/eval reference.
- `smfVLA/CLAUDE.md`, `snapflow/CLAUDE.md`, `freeflow/CLAUDE.md`, `dmf/README.md`.
