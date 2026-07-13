# CLAUDE.md

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
- Shared finetune base for all methods: `checkpoints/pi05_libero/`.
- **Nested git repos:** `openpi/`, `smfVLA/`, and `dmf/` each have their own
  `.git` — they are embedded repos (not submodules). Commits inside them are
  independent of the root repo. `snapflow/` and `freeflow/` are tracked by the
  root repo directly.
- `datasets/` and `checkpoints/` are gitignored — present on disk but never
  committed. Same for `logs/`, `wandb/`, `*.log`.
- Root `scripts/` holds only two conversion utilities (`convert_pytorch_to_jax.py`,
  `convert_calvin_to_jax.py`); it is NOT where method training scripts live.

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
# LIBERO-Plus
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type pi05 --nfe 1 --mode quick
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type dmf --nfe 10 --mode normal

# CALVIN (pi0.5 only, PyTorch)
cd /root/autodl-tmp && python -m eval.calvin.main --model-type pi05 --nfe 1 --mode quick
cd /root/autodl-tmp && python -m eval.calvin.main --model-type pi05 --nfe 10 --mode normal
```

- `eval/common/` — shared policy loader (`pi05` + `dmf` JAX), result utils, constants.
- `eval/libero_plus/` — LIBERO-Plus evaluation. `main.py` is the entry point. Modes:
  `quick` (1 suite, spatial, 10 tasks, 5 ep), `normal` (4 suites, perturbation sampling,
  5 ep, <10h), `fullset` (5 suites, all tasks, 50 ep).
- `eval/calvin/` — CALVIN ABCD→D official benchmark (pi0.5 PyTorch only, no DMF).
  Modes: `quick` (debug dataset, 5 seqs), `normal` (ABCD, 100 seqs), `fullset` (ABCD,
  1000 seqs).
- Model types: `pi05` (original) and `dmf`. NFE = `num_steps`; supports 1/2/4/10.
- `detect_checkpoint_type()` auto-sniffs: `logvar_proj`→dmf, else pi05.
- LIBERO-Plus loads the perturbed `libero-plus` package, not vanilla `libero`.
- Results → `eval/results/` as timestamped JSON; shell scripts in `eval/scripts/`
  for batch runs (`run_calvin_benchmark_all_nfe.sh`, `eval_dmf_all.sh`).

## Code style

Method dirs (`smfVLA`, `snapflow`, `freeflow`, `dmf`) and `eval/` use:

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

Note: `openpi/` itself uses line-length 120 — do not reformat openpi files to 100.

## Deeper docs

- `smfVLA/CLAUDE.md`, `snapflow/CLAUDE.md`, `freeflow/CLAUDE.md`, `dmf/README.md` — per-method details.
- `docs/experiment_workflow.md` — experiment recording workflow; `docs/experiment_template.md` — report template; `docs/experiments/` — saved reports.
