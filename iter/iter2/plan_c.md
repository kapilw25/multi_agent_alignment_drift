# iter2 Plan C: MAHALS Training Commands

> **Status**: Ready to execute on GPU machine
> **Date**: Jan 27, 2026

---

## MAHALS Training Files

| File | Purpose |
|------|---------|
| `configs/train_configs/mahals/llama31_8b_sft_10pct.yaml` | SFT config (10% Tulu, 8 GPUs) |
| `configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml` | DPO config (10% preference, 8 GPUs) |
| `scripts/finetune_with_accelerate_config_mahals.sh` | SFT training script |
| `scripts/dpo_train_with_accelerate_config_mahals.sh` | DPO training script |

---

## Environment Setup (one-time)

```bash
cd literature/tulu_train
uv sync                    # Creates .venv with all dependencies
uv sync --extra compile    # Install flash-attention (required for use_flash_attn: true)
source .venv/bin/activate
```

> **Note**: `uv` is a fast Rust-based Python package manager. If not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Alternatives:**
```bash
docker build .        # Docker (includes all dependencies)
pip install -e .      # Pip fallback (if uv unavailable)
```

---

## Execution Commands (on GPU machine)

```bash
cd literature/tulu_train
source .venv/bin/activate
mkdir -p logs

# Step 1: SFT Training
sh scripts/finetune_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee logs/sft_training.log  # sanity (1 GPU)
sh scripts/finetune_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee -a logs/sft_training.log  # full (8 GPUs)

# Step 2: DPO Training (after SFT completes)
sh scripts/dpo_train_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee logs/dpo_training.log  # sanity (1 GPU)
sh scripts/dpo_train_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee -a logs/dpo_training.log  # full (8 GPUs)
```

---

## Output Models

- SFT → `output/mahals_llama31_8b_sft_10pct`
- DPO → `output/mahals_llama31_8b_dpo_10pct`

---

## Sanity Test (1 GPU) = Quick Validation, NOT Full Training

Stop after **50-200 steps** (~5-10 mins). You're checking if training *works*, not training a good model.

---

### What to Look For in Logs

#### ✅ GREEN FLAGS (Proceed to 8 GPU)

```
# Loss decreasing (most important)
Step 10: loss = 2.45
Step 50: loss = 2.12
Step 100: loss = 1.89  ← Trending DOWN = good

# No memory errors
# Training speed non-zero
samples/sec: 1.2
tokens/sec: 4900

# Gradient norm stable (not exploding)
grad_norm: 0.85
grad_norm: 1.12
grad_norm: 0.94  ← Stable range = good
```

#### ❌ RED FLAGS (Debug Before 8 GPU)

```
# OOM Error
CUDA out of memory  ← Reduce batch size or max_seq_length

# NaN/Inf loss
Step 50: loss = nan  ← LR too high or data issue

# Loss NOT decreasing
Step 10: loss = 2.45
Step 100: loss = 2.48
Step 200: loss = 2.51  ← Flat/increasing = problem

# Exploding gradients
grad_norm: 1e+8  ← Add gradient clipping
```

---

### Recommended Workflow

```bash
# 1. Start sanity test
sh scripts/finetune_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee logs/sft_training.log

# 2. Watch logs for 5-10 mins (50-200 steps)
#    - Loss going down? ✓
#    - No OOM? ✓
#    - No NaN? ✓

# 3. Kill with Ctrl+C

# 4. Spin up 8 GPU instance, run full training
sh scripts/finetune_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee -a logs/sft_training.log
```

---

### Quick Reference

| Check | Pass | Fail |
|-------|------|------|
| Loss after 100 steps | < initial loss | Same or higher |
| Memory | No OOM | CUDA OOM error |
| Loss value | Numeric | NaN or Inf |
| Grad norm | 0.1 - 10 | >1000 or 0 |
| Steps completing | Yes | Hangs/crashes |

**Rule of thumb**: If loss drops by ~10% in first 100 steps with no errors → safe to scale to 8 GPUs.

---

## Expected Training Metrics

**SFT**: `train_loss`, `learning_rate`, `total_tokens`, `per_device_tps`

**DPO**: `train_loss`, `rewards/chosen`, `rewards/rejected`, `rewards/accuracy`, `rewards/margin`

> ⚠️ **DPO Initial Loss Delay**: DPO caches reference model logprobs first. **No loss shown for several minutes** - this is normal.

---

## Checkpoint & Resume

Configs have checkpointing enabled:
- `checkpointing_steps: epoch` (SFT) / `1000` (DPO)
- `keep_last_n_checkpoints: 3`
- `clean_checkpoints_at_end: false` ← keeps checkpoints after training

**If training crashes, just re-run the same command:**

```bash
# SFT resume (auto-detects checkpoint in output/mahals_llama31_8b_sft_10pct/)
sh scripts/finetune_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee -a logs/sft_training.log

# DPO resume (auto-detects checkpoint in output/mahals_llama31_8b_dpo_10pct/)
sh scripts/dpo_train_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee -a logs/dpo_training.log
```

No manual intervention needed - open-instruct auto-detects checkpoints in `output_dir`.
