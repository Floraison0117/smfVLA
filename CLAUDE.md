# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo trains and evaluates **1-NFE (one-step) action generation** for VLA models built on the **pi0.5** backbone. Four methods distill/compress pi0.5's 10-step flow into a single inference step:

- **SMF (SplitMeanFlow)** — model predicts *average* velocity + self-consistency
- **SnapFlow** — flow matching + 2-step shortcut self-distillation (extends SMF)
- **FreeFlow** — data-free distillation: 1-step student vs. frozen 10-step teacher
- **DMF (Decoupled MeanFlow)** — encoder(t)/decoder(r) transformer split + JVP MeanFlow loss

**Conda env for everything (train + eval):** `openpi_server` (has both `jax` 0.5.3 and `libero`). Other `libero_*`/`calvin_eval` envs are stale/incomplete.

---

## Architecture: two shared skeletons

### Training — identical 5-file layout per method

Every method lives in `<method>/` (`smfVLA`, `snapflow`, `freeflow`, `dmf`) with the **same** structure:

```
<method>/configs/train/*.yaml                          # hyperparams + freeze + base ckpt
<method>/scripts/run_train.py  (or src/.../run_train.py)   # entry point (bash scripts/train.sh wraps it)
<method>/src/<pkg>/training/jax_trainer.py             # JIT train loop: AdamW + warmup + cosine decay, grad clip, WandB
<method>/src/<pkg>/training/*_loss.py                  # THE method-specific file (see table)
<method>/src/<pkg>/models/pi05_*.py                    # model variant (differs by added time-conditioning head)
<method>/src/<pkg>/training/data_loader.py             # shared LeRobot v2.0 loader
```

(`<pkg>` = `smf_vla` / `snapflow` / `freeflow` / `dmf_vla`. SnapFlow's model **subclasses** SMF's `Pi05SMF`.)

**All four finetune from the same shared base:** `checkpoints/pi05_libero/` (`params/` + `assets/`). All **freeze the VLM backbone** and train only the action-expert layers (`*_1` suffix) + projection layers **plus each method's new time-embedding heads**.

**Per-method difference is concentrated in the loss + the added head:**

| Method | Added head | Loss (typical config weights) |
|--------|-----------|-------------------------------|
| SMF | `time_proj` (t,r) | `(1−flow_ratio)·L_SMF + flow_ratio·L_FM`  (`flow_ratio≈0.3`); L_SMF = self-consistency `u(z_t,r,t)` |
| SnapFlow | `target_time_mlp` | `α·L_FM + (1−α)·λ·L_shortcut`  (`α=0.5`, `λ=0.1`); 2-step Euler shortcut target |
| FreeFlow | dual time emb `time_mlp_in` | `L_path + λ·L_correction`  (`λ=0.1`, `teacher_nfe=10`); student 1-step vs frozen teacher 10-step Euler path. **Data-free** (no action labels) |
| DMF | `logvar_proj` | `0.5·(L_FM + L_MF)`; encoder cond. on t, decoder on r; JVP computes `du/dt` for MeanFlow; EMA (decay=0.9999), eval uses EMA model |

**Data:** LeRobot v2.0 parquet. SMF/SnapFlow/FreeFlow train on `datasets/libero`; **DMF trains on `datasets/libero-plus-training`**. Batch dict: `{observation:{image,image_mask,state}, actions, action_mean, action_std, prompt}`. Raw action dim 7 → padded to 32. Images: parquet→PIL→rotate 180°→resize 256→224 (PIL LANCZOS).

**Launch:**
```bash
cd /root/autodl-tmp/<method> && bash scripts/train.sh              # default config
bash scripts/train.sh configs/train/<config>.yaml                  # specific config
bash scripts/train.sh configs/train/<config>.yaml --resume <ckpt>  # resume
```

**Finetuned checkpoint locations (as the eval code resolves them):**

| Method | Finetuned dir |
|--------|--------------|
| SMF | (eval defaults to the base `checkpoints/pi05_libero`) |
| SnapFlow | `checkpoints/snapflow_finetuned/step_N` |
| FreeFlow | `freeflow/checkpoints/finetuned/freeflow/step_N` |
| DMF | `checkpoints/dmf_finetuned/step_N` |

### Evaluation — one dispatch, one shared core

```
eval/scripts/run_eval.py          # unified entry: --dataset {libero,libero-plus,calvin} dispatches below
  ├─ eval_direct.py               # LIBERO standard
  ├─ eval_libero_plus.py          # LIBERO-Plus robustness (multi-episode/task)
  └─ eval_calvin.py               # CALVIN  (⚠️ debug/partial — full sequences + task_oracle not implemented)
eval/scripts/eval_utils.py        # THE core: load_policy() + detect_checkpoint_type()
```

**`eval_utils.py` is the key file.** `--model-type {smf,snapflow,freeflow,dmf}` routes `load_policy()` to the matching model class (`Pi05SMF`/`Pi05SnapFlow`/`Pi05FreeFlow`/`Pi05DMF`). `detect_checkpoint_type()` **auto-sniffs the method from checkpoint param keys** so you usually don't need `--model-type`:

| Keys present | Inferred method |
|--------------|-----------------|
| `logvar_proj` | DMF |
| `target_time_mlp` | SnapFlow |
| `time_proj` | SMF |
| `time_mlp_in` + nested `{'model':...}` | FreeFlow |
| (none of the above) | original pi0.5 |

Two checkpoint formats are handled: standard (`ckpt/params/`) and FreeFlow (flat, `_METADATA` at root). `eval_utils.setup_paths()` injects each method's `src/` into `PYTHONPATH`. NFE = sampling `num_steps` (1/2/4/10) in the Policy constructor — all methods support all four.

**Launch:**
```bash
cd /root/autodl-tmp/eval/scripts
python run_eval.py --dataset libero      --mode preset --nfe 1 --model-type freeflow --checkpoint <ckpt>
python run_eval.py --dataset libero-plus --mode quick  --nfe 1 --model-type snapflow
python run_eval.py --dataset calvin      --calvin-dataset debug --nfe 1 --model-type smf
```

**Presets** (suites × tasks × episodes/task):

| Benchmark | `quick` | `preset`/`full` | `fullset`/`full90` |
|-----------|---------|-----------------|--------------------|
| LIBERO (`eval_direct.py`) | libero_spatial, 5 ep | `preset`: 4 suites, **5** ep | `fullset`: 5 suites (incl. libero_90), 50 ep |
| LIBERO-Plus (`eval_libero_plus.py`) | 4 suites, 10 tasks, 5 ep | `full`: 4 suites, all tasks | `full90`: libero_90, all tasks |

(`preset` = **5 ep/task**, not 50 — only `fullset`/`full` use 50.) LIBERO-Plus loads the perturbed `libero-plus` package, not vanilla libero. Results → `eval/results/<model_type>/...` as timestamped JSON (`{ts}_combined_<nfe>nfe.json` for LIBERO-Plus), each recording its checkpoint in `metadata.checkpoint`.

---

## Method-specific docs

- `smfVLA/CLAUDE.md`, `snapflow/CLAUDE.md`, `freeflow/CLAUDE.md` — per-method detail
- `dmf/README.md` — DMF (Decoupled MeanFlow); paper arxiv 2510.24474

---

## Code Style

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```
