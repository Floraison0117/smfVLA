# AGENTS.md

## Environment

- **One env for everything:** `openpi_server`. Interpreter at
  `/root/miniconda3/envs/openpi_server/bin/python`. Has `jax==0.5.3` + `libero`.
- `WANDB_API_KEY` must be exported before training or WandB is silently skipped
  (train scripts only warn).

## Repo layout gotchas

- `openpi/` (top-level) is the **shared** pi0.5 framework used by all five
  methods and by eval. It is NOT a per-method subdirectory.
- Each method's `<method>/third_party/openpi` is a **symlink** to
  `/root/autodl-tmp/openpi` (DMF and Pi-Flow have none and use `../openpi`
  instead). Never edit code through these symlinks — edit `openpi/` directly.
- Method package names differ: `smf_vla`, `snapflow`, `freeflow`, `dmf_vla`,
  `piflow_vla` (under `<method>/src/`). SnapFlow's model subclasses SMF's
  `Pi05SMF`; Pi-Flow's `Pi05PiFlow` subclasses `openpi.models.pi0.Pi0` directly.
- Shared finetune base for all methods: `checkpoints/pi05_libero/`.
- **Single root repo:** all of `openpi/`, `smfVLA/`, `snapflow/`, `freeflow/`,
  `dmf/`, and `piflow/` are tracked directly by the root repo — no nested
  `.git`, no submodules. (Previously `openpi/`, `smfVLA/`, and `dmf/` were
  embedded git repos; they were flattened into the root repo in commit
  `4bb9e22`. The remote is `Floraison0117/smfVLA.git`.)
- `datasets/` and `checkpoints/` are gitignored — present on disk but never
  committed. Same for `logs/`, `wandb/`, `*.log`.
- Root `scripts/` holds checkpoint utilities (NOT method training scripts):
  `merge_lora_checkpoint.py` — merges LoRA adapters (`lora_a`/`lora_b`) back
  into base weights, producing a non-LoRA checkpoint loadable by DMF/Pi-Flow
  (which use `gemma_2b` + `gemma_300m`, not the LoRA variants). Used to convert
  the InternRobotics pi0.5 LoRA fine-tune (71 keys) into
  `checkpoints/pi05_calvin` (non-LoRA, 51 keys, same structure as
  `pi05_libero`). The original LoRA checkpoint was removed after merging, so
  `pi05_calvin` now holds the merged non-LoRA weights.
  (The old `convert_calvin_pt_to_jax.py` and its output
  `checkpoints/pi05_calvin_jax` were both removed; CALVIN eval now uses
  `checkpoints/pi05_calvin` for the `pi05` baseline too.)

## Training

```bash
cd /root/autodl-tmp/<method> && bash scripts/train.sh                              # default config
bash scripts/train.sh configs/train/<config>.yaml --resume <ckpt>                 # resume
```

- Each `train.sh` sets `PYTHONPATH` (method `src/` + openpi `src/` + client) and
  activates `openpi_server` itself — run it, don't invoke `run_train.py` bare.
- DMF's default config is `configs/train/dmf_libero_plus.yaml`; DMF trains on
  `datasets/libero-plus-training`. Pi-Flow's default config is
  `configs/train/piflow_libero_plus.yaml` and also trains on
  `datasets/libero-plus-training`. smfVLA / SnapFlow / FreeFlow now default to
  their `*_libero_plus.yaml` configs (`smf_base_libero_plus.yaml`,
  `snapflow_libero_plus.yaml`, `freeflow_base_libero_plus.yaml`) and likewise
  train on `datasets/libero-plus-training`.
- All methods **freeze the VLM backbone**; only action-expert layers (suffix
  `*_1`), projection layers, and each method's new time-embedding heads train.
- All `train.sh` / `run_train.py` set the JAX env (`JAX_PLATFORMS=cuda`,
  `XLA_FLAGS=--xla_gpu_autotune_level=0`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`,
  `JAX_COMPILATION_CACHE_MAX_SIZE`) **before** `import jax` (see
  `docs/training-debug.md` §4; OOM root cause if missing).
- Finetuned checkpoints land in: SnapFlow `checkpoints/snapflow_finetuned/step_N`,
  FreeFlow `freeflow/checkpoints/finetuned/freeflow/step_N`, DMF
  `checkpoints/dmf_finetuned/step_N`, Pi-Flow
  `checkpoints/piflow_finetuned/step_N`, SMF `checkpoints/smf_finetuned/step_N`.
  (SMF previously fell back to the base checkpoint; it now has a finetuned
  location too.) CALVIN finetunes land in `checkpoints/dmf_finetuned_calvin/`
  and `checkpoints/piflow_finetuned_calvin/` respectively (see CALVIN section).
- **Training time (30k steps):** DMF and Pi-Flow ~11h (~1340ms/step on RTX PRO
  6000, ~432M trainable + 3B frozen). JIT compile adds ~5min on first run.
- **Data-loader note:** DMF, Pi-Flow, smfVLA, SnapFlow and FreeFlow now share an
  identical `data_loader.py` (smfVLA/SnapFlow/FreeFlow symlink to
  `smfVLA/src/smf_vla/training/data_loader.py`; synced 2026-07-16). The shared
  version filters episodes with missing `.mp4` files before building the frame
  index, and does single-pass video decoding with LRU-cached parquet reads.
  (Pi-Flow's old copy was missing the video-file filter for v2.1 datasets —
  episodes with missing `.mp4` files caused `KeyError` mid-training; the old
  SnapFlow/FreeFlow copy silently fell back to black images instead of
  crashing, training on wrong data. See `docs/training-debug.md` §8.)

### CALVIN training

DMF and Pi-Flow support CALVIN training (smfVLA/SnapFlow/FreeFlow do not yet
have CALVIN configs). The base checkpoint is **not** `pi05_libero` (LIBERO
domain) but `checkpoints/pi05_calvin` — a non-LoRA checkpoint produced by
merging the LoRA adapters from the InternRobotics pi0.5 LoRA fine-tune
(`InternRobotics/InternData-Calvin_ABC`, 30k steps) back into the base weights
via `scripts/merge_lora_checkpoint.py`. The merge is necessary because
DMF/Pi-Flow use the non-LoRA model variant (`gemma_2b` + `gemma_300m`, 51 keys);
loading the raw LoRA checkpoint (71 keys) directly would silently drop all 20
LoRA adapter keys, discarding the CALVIN LLM adaptation. The original LoRA
checkpoint was removed after merging, so `pi05_calvin` now holds the merged
non-LoRA weights.

```bash
# DMF CALVIN (uses datasets/calvin_lerobot, LeRobot v2.0 parquet)
cd /root/autodl-tmp/dmf && bash scripts/train.sh configs/train/dmf_calvin.yaml

# Pi-Flow CALVIN (same dataset)
cd /root/autodl-tmp/piflow && bash scripts/train.sh configs/train/piflow_calvin.yaml
```

- **Data pipeline:** CALVIN episode `.npz` files → LeRobot v2.0 parquet via
  `dmf/scripts/convert_calvin_data_to_lerobot.py` → `datasets/calvin_lerobot/`
  (200 episodes, pre-converted). `norm_stats.json` lives in the dataset dir
  (same values as the eval checkpoint's assets; differs slightly from the
  InternRobotics checkpoint's own norm_stats because the training datasets
  differ). The shared `data_loader.py` reads norm_stats from the dataset path.
- **Base checkpoint:** `checkpoints/pi05_calvin` (51 keys, non-LoRA, same param
  structure as `pi05_libero`). Produced once by `scripts/merge_lora_checkpoint.py`
  from the (now-deleted) InternRobotics LoRA checkpoint; the merged result is the
  canonical CALVIN pi0.5 used by both training and eval, so re-running the merge
  is not needed.
- **Freeze/trainable patterns** are identical to the LIBERO configs (freeze
  VLM backbone, train action-expert `*_1` + projections + method heads). CALVIN
  configs use `batch_size: 16`, `save_every: 2000`.
- **Finetuned checkpoints:** DMF → `checkpoints/dmf_finetuned_calvin/step_N`,
  Pi-Flow → `checkpoints/piflow_finetuned_calvin/step_N`.

## Evaluation

```bash
# LIBERO-Plus
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type pi05 --nfe 1 --mode quick
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type dmf --nfe 10 --mode normal
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type snapflow --nfe 1 --mode quick

# CALVIN (JAX, supports pi05/dmf/piflow/smf/snapflow/freeflow)
cd /root/autodl-tmp && python -m eval.calvin.main --model-type pi05 --nfe 1 --mode quick
cd /root/autodl-tmp && python -m eval.calvin.main --model-type dmf --nfe 10 --mode normal
cd /root/autodl-tmp && python -m eval.calvin.main --model-type piflow --nfe 1 --mode fullset
```

- `eval/common/` — shared policy loader (`pi05` + `dmf` + `piflow` + `smf` +
  `snapflow` + `freeflow` JAX), result utils, constants.
- `eval/libero_plus/` — LIBERO-Plus evaluation. `main.py` is the entry point. Modes:
  `quick` (1 suite, spatial, 10 tasks, 5 ep), `normal` (4 suites, perturbation sampling,
  5 ep, <10h), `fullset` (5 suites, all tasks, 50 ep).
- `eval/calvin/` — CALVIN ABCD→D official benchmark (JAX, supports
  pi05/dmf/piflow/smf/snapflow/freeflow). Modes: `quick` (debug dataset, 5 seqs),
  `normal` (ABCD, 100 seqs), `fullset` (ABCD, 1000 seqs). Default pi05 checkpoint:
  `checkpoints/pi05_calvin` (the merged non-LoRA CALVIN pi0.5).
- Model types: `pi05` (original), `dmf`, `piflow`, `smf`, `snapflow`, `freeflow`.
  NFE = `num_steps`; supports 1/2/4/10. All six are supported by both LIBERO-Plus
  and CALVIN. `smf`/`snapflow`/`freeflow` are 1-NFE methods (nfe is typically 1).
- `detect_checkpoint_type()` auto-sniffs: `target_time_mlp`→snapflow,
  `time_proj`→smf, `gmm_mean_proj`→piflow, `logvar_proj`→dmf, else pi05.
  (FreeFlow has no new head → indistinguishable from pi05 by params; pass
  `--model-type freeflow` explicitly to load it as `Pi05FreeFlow`.)
- LIBERO-Plus loads the perturbed `libero-plus` package, not vanilla `libero`.
- Results → `eval/results/` as timestamped JSON; shell scripts in `eval/scripts/`
  for batch runs (`run_calvin_benchmark_all_nfe.sh`, `run_libero_parallel.sh`).

## Code style

Method dirs (`smfVLA`, `snapflow`, `freeflow`, `dmf`, `piflow`) and `eval/` use:

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

Note: `openpi/` itself uses line-length 120 — do not reformat openpi files to 100.

## Deeper docs

- `smfVLA/CLAUDE.md`, `snapflow/CLAUDE.md`, `freeflow/CLAUDE.md`, `dmf/README.md`,
  `piflow/README.md` — per-method details.
- `docs/experiment_workflow.md` — experiment recording workflow; `docs/experiment_template.md` — report template; `docs/experiments/` — saved reports.
- `docs/training-debug.md` — layered debugging checklist + case studies (OOM,
  NaN, slow training, KV-split optimization, data-loader video filter fix).
