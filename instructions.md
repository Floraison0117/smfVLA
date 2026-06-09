# VLA 1-NFE Training & Evaluation Instructions

Quick reference for training and evaluating π₀.₅-based 1-NFE action generation methods.

## Environment Setup

```bash
# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

# Verify GPU
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

---

## Training Commands

### Pi0.5 (Base Teacher)

```bash
# Located in openpi/, not typically trained from scratch
# Base checkpoint: checkpoints/pi05_libero/
```

### SMF (SplitMeanFlow)

```bash
cd /root/autodl-tmp/smfVLA

# Default config (smf_base)
bash scripts/train.sh

# Custom config
bash scripts/train.sh configs/train/smf_decte_curr_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/smf_base_libero.yaml
# Then modify run_train.py: --resume checkpoints/finetuned/smf_base/step_5000

# Training variants (7 configs)
bash scripts/train.sh configs/train/smf_base_libero.yaml          # Basic SMF
bash scripts/train.sh configs/train/smf_curr_libero.yaml           # Curriculum Time Sampling
bash scripts/train.sh configs/train/smf_decte_libero.yaml          # Decoupled Time Embedding
bash scripts/train.sh configs/train/smf_decte_curr_libero.yaml      # DecTE + Curriculum
bash scripts/train.sh configs/train/smf_decte_curr_anchor.yaml     # + Anchor Loss
bash scripts/train.sh configs/train/smf_decte_curr_bpl.yaml         # + Behavioral Perceptual Loss
bash scripts/train.sh configs/train/smf_full_libero.yaml           # All variants combined
```

### SnapFlow

```bash
cd /root/autodl-tmp/snapflow

# Default config
bash scripts/train.sh configs/train/snapflow_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/snapflow_libero.yaml --resume checkpoints/finetuned/snapflow/step_10000
```

### FreeFlow

```bash
cd /root/autodl-tmp/freeflow

# Default config (freeflow_base)
bash scripts/train.sh configs/train/freeflow_base_libero.yaml

# Custom config with resume
bash scripts/train.sh configs/train/freeflow_full.yaml --resume checkpoints/finetuned/freeflow/step_10000

# Training variants
bash scripts/train.sh configs/train/freeflow_base_libero_plus.yaml  # LIBERO-Plus training
bash scripts/train.sh configs/train/freeflow_no_correction.yaml     # Without error correction
bash scripts/train.sh configs/train/freeflow_full.yaml              # All variants combined
```

---

## Evaluation Commands

### Unified Entry Point (Recommended)

```bash
cd /root/autodl-tmp/eval/scripts

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIBERO Standard Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SMF - Quick test
python run_eval.py --dataset libero --mode quick --nfe 1 --model-type smf

# SMF - Preset (4 suites, 50 ep)
python run_eval.py --dataset libero --mode preset --nfe 1 --model-type smf

# SnapFlow - Full evaluation
python run_eval.py --dataset libero --mode fullset --nfe 1 --model-type snapflow

# FreeFlow - Custom
python run_eval.py --dataset libero --task-suite libero_spatial --num-episodes 10 --nfe 1 --model-type freeflow

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIBERO-Plus Robustness Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SMF - Quick (50 tasks)
python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type smf

# SnapFlow - Medium (100 tasks)
python run_eval.py --dataset libero-plus --mode medium --nfe 1 --model-type snapflow

# FreeFlow - Full (all suites)
python run_eval.py --dataset libero-plus --mode full --nfe 1 --model-type freeflow

# FreeFlow - libero_90 suite
python run_eval.py --dataset libero-plus --mode full90 --nfe 1 --model-type freeflow

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-NFE Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Test different NFE values
for nfe in 1 2 4 10; do
    python run_eval.py --dataset libero --mode quick --nfe $nfe --model-type smf
done
```

### Direct Script Usage

```bash
cd /root/autodl-tmp/eval/scripts

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIBERO Standard (eval_direct.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SMF evaluation
python eval_direct.py --preset quick --nfe 1 --model-type smf

# SnapFlow evaluation
python eval_direct.py --preset preset --nfe 1 --model-type snapflow

# FreeFlow evaluation
python eval_direct.py --preset fullset --nfe 1 --model-type freeflow

# Custom checkpoint
python eval_direct.py --preset quick --nfe 1 --model-type smf \
    --checkpoint /root/autodl-tmp/checkpoints/finetuned/smf_base/step_5000

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIBERO-Plus Robustness (eval_libero_plus.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SMF evaluation
python eval_libero_plus.py --preset quick --nfe 1 --model-type smf

# SnapFlow evaluation
python eval_libero_plus.py --suite libero_spatial --nfe 1 --model-type snapflow

# FreeFlow evaluation
python eval_libero_plus.py --preset medium --nfe 1 --model-type freeflow

# Custom checkpoint
python eval_libero_plus.py --preset quick --nfe 1 --model-type freeflow \
    --checkpoint /root/autodl-tmp/checkpoints/finetuned/freeflow/step_10000

# Multi-NFE evaluation
python eval_libero_plus.py --suite libero_spatial --nfe 1 2 4 10 --model-type smf
```

### Model-Specific Scripts

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FreeFlow Evaluation Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd /root/autodl-tmp/freeflow

# LIBERO-Plus evaluation
NFE=1 PRESET=quick CHECKPOINT=../checkpoints/finetuned/freeflow \
    bash scripts/eval_freeflow.sh

# LIBERO standard evaluation
DATASET=libero NFE=1 PRESET=preset CHECKPOINT=../checkpoints/finetuned/freeflow \
    bash scripts/eval_freeflow.sh
```

---

## Evaluation Modes

### LIBERO Standard

| Mode | Suites | Episodes/Task | Description |
|------|--------|---------------|-------------|
| `quick` | libero_spatial | 5 | Fast smoke test |
| `preset` | 4 suites | 50 | Standard evaluation |
| `fullset` | 5 suites | 50 | Complete evaluation |

### LIBERO-Plus Robustness

| Mode | Suites | Tasks | Description |
|------|--------|-------|-------------|
| `quick` | 4 suites | 10 each | Fast test (50 tasks) |
| `medium` | libero_spatial | 100 | Medium evaluation |
| `full` | 4 suites | All | Complete evaluation |
| `full90` | libero_90 | All | LIBERO-90 suite |

---

## Default Checkpoint Paths

| Model | Checkpoint Path |
|-------|----------------|
| Pi0.5 (Base) | `checkpoints/pi05_libero/` |
| SMF | `checkpoints/smf_base/pi05_libero/` |
| SnapFlow | `checkpoints/smf_base/pi05_libero/` |
| FreeFlow | `checkpoints/freeflow/pi05_libero/` |

---

## Results Location

Results are saved to:

```bash
# Unified eval results
/root/autodl-tmp/eval/results/

# Model-specific results
/root/autodl-tmp/eval/results/smf/
/root/autodl-tmp/eval/results/snapflow/
/root/autodl-tmp/eval/results/freeflow/
```

---

## NFE Options

| NFE | Description | Use Case |
|-----|-------------|----------|
| 1 | Single forward pass | SMF/SnapFlow/FreeFlow 1-NFE models |
| 2 | 2-step Euler | Baseline comparison |
| 4 | 4-step Euler | Mid-range performance |
| 10 | 10-step Euler | Original Pi0.5 performance |

---

## Common Arguments

### Evaluation Arguments

```bash
--dataset {libero,libero-plus}    # Dataset to evaluate
--mode {quick,preset,fullset}    # Evaluation preset
--nfe {1,2,4,10}                 # Number of function evaluations
--model-type {smf,snapflow,freeflow}  # Model architecture
--checkpoint PATH                # Custom checkpoint path
--seed INT                       # Random seed (default: 7)
--replan-steps INT              # Actions per inference (default: 5)
```

### Training Arguments

```bash
--config PATH                    # Training config YAML
--resume PATH                    # Resume from checkpoint
```

---

## Quick Reference Table

| Task | Command |
|------|---------|
| Train SMF | `cd smfVLA && bash scripts/train.sh` |
| Train SnapFlow | `cd snapflow && bash scripts/train.sh configs/train/snapflow_libero.yaml` |
| Train FreeFlow | `cd freeflow && bash scripts/train.sh` |
| Eval SMF (LIBERO) | `cd eval/scripts && python run_eval.py --dataset libero --mode preset --nfe 1 --model-type smf` |
| Eval SnapFlow (LIBERO-Plus) | `cd eval/scripts && python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type snapflow` |
| Eval FreeFlow (LIBERO-Plus) | `cd eval/scripts && python run_eval.py --dataset libero-plus --mode full --nfe 1 --model-type freeflow` |

---

## Troubleshooting

### GPU Memory Issues

```bash
# Check JAX GPU allocation
python -c "import jax; print(jax.devices())"

# Clear JAX cache
rm -rf /root/autodl-tmp/freeflow/.jax_cache/*
```

### Checkpoint Loading Issues

```bash
# Verify checkpoint structure
ls -la /root/autodl-tmp/checkpoints/freeflow/pi05_libero/

# Check for params/ and assets/ directories
```

### Dataset Issues

```bash
# Verify LIBERO dataset
ls -la /root/autodl-tmp/datasets/libero/

# Check for norm_stats.json
cat /root/autodl-tmp/datasets/libero/norm_stats.json | jq .
```
