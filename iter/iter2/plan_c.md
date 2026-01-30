# iter2 Plan C: MAHALS Training Commands

> **Status**: 1 GPU Sanity Test PASSED (Jan 29, 2026)
> **Date**: Jan 27, 2026 (updated Jan 29, 2026)

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
tmux # very important incase your local machine lose connection [happens frequenty during long idle activity on local machine] to instance, though INstance is running in background

# Step 1: SFT Training

# sanity (1 GPU) - validates: data loading, loss decrease, no NaN
sh scripts/finetune_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee logs/sft_training.log

# ⚠️ BEFORE 2+ GPU: Edit YAML config to disable 8-bit optimizer (incompatible with DeepSpeed)
#    In configs/train_configs/mahals/llama31_8b_sft_10pct.yaml:
#    use_8bit_optimizer: false  # Must be false for DeepSpeed ZeRO-3

# DeepSpeed test (2 GPUs) - validates: ZeRO-3 init, NCCL, gradient sync, model sharding
sh scripts/finetune_with_accelerate_config_mahals.sh 2 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee logs/sft_2gpu_test.log

# full (8 GPUs) - only after 2 GPU test passes
sh scripts/finetune_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee -a logs/sft_training.log  

# Step 2: DPO Training (after SFT completes)

# sanity (1 GPU) - validates: data loading, loss decrease, no NaN
sh scripts/dpo_train_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee logs/dpo_training.log

# ⚠️ BEFORE 2+ GPU: Edit YAML config to disable 8-bit optimizer (incompatible with DeepSpeed)
#    In configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml:
#    use_8bit_optimizer: false  # Must be false for DeepSpeed ZeRO-3

# DeepSpeed test (2 GPUs) - validates: ZeRO-3 init, NCCL, gradient sync, model sharding
sh scripts/dpo_train_with_accelerate_config_mahals.sh 2 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee logs/dpo_2gpu_test.log

# full (8 GPUs) - only after 2 GPU test passes
sh scripts/dpo_train_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee -a logs/dpo_training.log
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

---

## Debugging Log (Jan 29, 2026)

### Sanity Test Results (1 GPU, A100 80GB)

| Metric | Value | Status |
|--------|-------|--------|
| Distributed | `DistributedType.NO` | No DeepSpeed (8-bit optimizer active) |
| Loss | 1.26 → 0.75 avg | Decreasing (healthy) |
| TPS | ~2000 tokens/sec | Good throughput |
| LR | Warming up | Correct schedule |
| Memory | No OOM | Fits in 80GB |

**Sanity test PASSED** after resolving 9 issues.

---

### Loss Analysis (Extended Run)

| Step | Loss | Notes |
|------|------|-------|
| 1 | 1.26 | Initial loss |
| 50 | ~1.05 | Early decrease |
| 1366-1368 | 0.60 - 0.98 | Fluctuating but stable |
| **Average** | **~0.75** | **~40-50% reduction** |

**Verdict**: ✅ Training is HEALTHY
- Loss decreased ~40-50% from initial
- Fluctuation in 0.6-1.0 range is normal for SFT
- No NaN, no OOM, no gradient explosion
- Ready for 8 GPU full training

---

### Bugs Fixed in open-instruct Codebase

| # | Error | Fix | File |
|---|-------|-----|------|
| 1 | YAML config not parsed | `parse()` instead of `parse_args_into_dataclasses()` | `finetune.py:964` |
| 2 | `dpo_tune.py` not found | Script path → `dpo.py` | `dpo_train_...mahals.sh` |
| 3 | Two dataset selection mechanisms | `dataset_mixer_list` default → `None` | `finetune.py:128` |
| 4 | wandb API key error | Added `wandb_enabled` guard | `finetune.py:426-428` |
| 5 | TensorBoard type error | Whitelist sanitization for config | `finetune.py:433-435` |
| 6 | Float not iterable | Handle numeric types from YAML | `dataset_transformation.py:1956` |
| 7 | Chat template not set | Added `chat_template_name: tulu` | YAML configs |
| 8 | Dataset cache stale | Clear `local_dataset_cache/` when config changes | Manual |
| 9 | OOM with DeepSpeed | 1 GPU: No DeepSpeed + 8-bit optimizer | `finetune_...mahals.sh` |

---

### Key Configuration for 1 GPU vs 2 GPU vs 8 GPU

| Config | 1 GPU (Sanity) | 2 GPU (DeepSpeed Test) | 8 GPU (Full) |
|--------|----------------|------------------------|--------------|
| DeepSpeed | ❌ No | ✅ ZeRO-3 | ✅ ZeRO-3 |
| 8-bit Optimizer | ✅ true | ❌ false | ❌ false |
| Validates | Data, loss, NaN | NCCL, sharding, sync | Full training |
| Cost | ~$2 | ~$5 | ~$100+ |

> **Toggle Checklist (before 2+ GPU):**
> ```yaml
> # In llama31_8b_sft_10pct.yaml and llama31_8b_dpo_10pct.yaml:
> use_8bit_optimizer: false  # Change from true → false
> ```

### 2 GPU Test: What to Look For

✅ **GREEN FLAGS:**
```
DeepSpeed ZeRO Stage 3 initialized
NCCL initialized successfully
Loss decreasing across steps
```

❌ **RED FLAGS:**
```
NCCL timeout/connection refused
DeepSpeed initialization failed
Hang at distributed barrier
```

---

### Cache Clearing (if config changes)

```bash
# Clear dataset cache when changing max_seq_length or tokenizer settings
rm -rf /workspace/multi_agent_alignment_drift/literature/tulu_train/local_dataset_cache/*
```
