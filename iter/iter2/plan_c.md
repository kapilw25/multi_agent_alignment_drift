# iter2 Plan C: MAHALS Training Commands

> **Status**: ✅ SFT COMPLETE | ✅ DPO COMPLETE | ✅ BOTH PUSHED TO HF | 🔜 Phase 3 Ready
> **Date**: Jan 31, 2026
>

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
tmux # very important incase your local machine lose connection [happens frequenty during long idle activity on local machine] to instance, though INstance is running in background
cd literature/tulu_train
source .venv/bin/activate
huggingface-cli login --token <HF_TOKEN>
mkdir -p logs
rm -rf /workspace/multi_agent_alignment_drift/literature/tulu_train/local_dataset_cache/*  
```

```bash
# Step 1: SFT Training

# sanity (1 GPU) - validates: data loading, loss decrease, no NaN
sh scripts/finetune_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee logs/sft_1gpu_test.log

# ⚠️ BEFORE 2+ GPU: Edit YAML config to disable 8-bit optimizer (incompatible with DeepSpeed)
#    In configs/train_configs/mahals/llama31_8b_sft_10pct.yaml:
#    use_8bit_optimizer: false  # Must be false for DeepSpeed ZeRO-3

# DeepSpeed test (2 GPUs) - validates: ZeRO-3 init, NCCL, gradient sync, model sharding
sh scripts/finetune_with_accelerate_config_mahals.sh 2 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee logs/sft_2gpu_test.log

# full (8 GPUs) - only after 2 GPU test passes
rm -rf /workspace/multi_agent_alignment_drift/literature/tulu_train/local_dataset_cache/*  
huggingface-cli login --token <HF_TOKEN>
sh scripts/finetune_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_sft_10pct.yaml 2>&1 | tee -a logs/sft_training.log  
```

```bash
# Step 2: DPO Training (after SFT completes)
rm -rf /workspace/multi_agent_alignment_drift/literature/tulu_train/local_dataset_cache/* 
# sanity (1 GPU) - validates: data loading, loss decrease, no NaN
sh scripts/dpo_train_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee logs/dpo_1gpu_test.log

# ⚠️ BEFORE 2+ GPU: Edit YAML config to disable 8-bit optimizer (incompatible with DeepSpeed)
#    In configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml:
#    use_8bit_optimizer: false  # Must be false for DeepSpeed ZeRO-3

# DeepSpeed test (2 GPUs) - validates: ZeRO-3 init, NCCL, gradient sync, model sharding
rm -rf /workspace/multi_agent_alignment_drift/literature/tulu_train/local_dataset_cache/* 
sh scripts/dpo_train_with_accelerate_config_mahals.sh 2 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee logs/dpo_2gpu_test.log

# full (8 GPUs) - only after 2 GPU test passes
rm -rf /workspace/multi_agent_alignment_drift/literature/tulu_train/local_dataset_cache/*  
huggingface-cli login --token hf_***
sh scripts/dpo_train_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee -a logs/dpo_8gpu_test.log
```

---

## Output Models

| Stage | Local Path | HuggingFace |
|-------|------------|-------------|
| SFT | `output/mahals_llama31_8b_sft_10pct` | [anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS](https://huggingface.co/anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS) ✅ |
| DPO | `output/mahals_llama31_8b_dpo_10pct` | [anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS](https://huggingface.co/anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS) ✅ |

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

### DPO-Specific: What to Watch For (First 50 Steps)

| Check | Expected | Red Flag |
|-------|----------|----------|
| Reference model caching | "Computing reference logprobs" message | OOM during caching |
| Loss appears | After 2-5 min (reference caching done) | Never shows loss |
| Loss value | ~0.5-1.5 initial | NaN or >10 |
| rewards/accuracy | Should appear | Missing metric |
| Memory | ~45-55 GB per GPU | OOM |

**DPO Green Flags in Logs:**
```
Computing reference model logprobs...  ← Normal, takes 2-5 min
Step 1: loss = 0.693                   ← DPO loss should start here
rewards/chosen: 0.1                    ← Preference learning
rewards/rejected: -0.1                 ← Should be lower than chosen
rewards/accuracy: 0.5                  ← Should increase over time
```

**DPO Red Flags:**
```
CUDA out of memory                     ← Reduce batch to 1, grad_accum to 16
loss = nan                             ← Check beta value or data
rewards/accuracy stuck at 0.5          ← Model not learning preferences
```

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

#### SFT Bugs (finetune.py)

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

#### DPO Bugs (dpo.py, dpo_utils.py)

| # | Error | Fix | File |
|---|-------|-----|------|
| 10 | Two dataset selection mechanisms | `mixer_list` default → `None` | `dpo_utils.py:167` |
| 11 | wandb API key error (callback) | Added `wandb_enabled` guard | `dpo.py:229-232` |
| 12 | HF repo_id namespace duplication | `hf_repo_id` → just repo name | YAML configs |
| 13 | `clean_checkpoints_at_end` not recognized | Commented out (SFT-only param) | `llama31_8b_dpo_10pct.yaml` |
| 14 | `dataset_mixer` dict→list missing | Added conversion like `finetune.py:477` | `dpo.py:335-337` |
| 15 | wandb login unconditional (utils) | Guard `maybe_use_ai2_wandb_entity()` | `utils.py:1367-1372` |
| 16 | `dpo.py` uses OLMo-only loading | Switch to `dpo_tune_cache.py` | `dpo_train_...mahals.sh` |
| 17 | `dpo_tune_cache.py` needs YAML | `parse()` instead of `parse_args_into_dataclasses()` | `dpo_tune_cache.py:760` |
| 18 | wandb login unconditional (dpo_tune) | Guard inside `with_tracking` block | `dpo_tune_cache.py:197-199` |
| 19 | `init_trackers()` wandb + logger kwarg | (a) Guard wandb init_kwargs, (b) `accelerator.print()` vs `logger.info(main_process_only)` | `dpo_tune_cache.py:203-218,238` |
| 20 | `log_with` hardcoded to "wandb" | Use `args.report_to` instead | `dpo_tune_cache.py:122` |
| 21 | TensorBoard rejects list/dict/None | Sanitize `experiment_config` to str | `dpo_tune_cache.py:204-206` |
| 22 | `dataset_mixer` is str not dict | `ast.literal_eval()` to parse | `dpo_tune_cache.py:264-265` |
| 23 | `mixer_list_splits/transform_fn/target_columns` stringified | `ast.literal_eval()` for all list params | `dpo_tune_cache.py:268-273` |
| 24-28 | ALL `'None'` strings from YAML | Generalized: `for attr in dir(args): if val == 'None': setattr(None)` | `dpo_tune_cache.py:274-277` |
| 25 | `additional_model_arguments='{}'` string | `ast.literal_eval()` to parse dict | `dpo_tune_cache.py:278-279` |
| 29 | `build_reference_logprobs_cache()` wrong params | Fix call signature: `device`, `cache_path`, `is_main_process`, `model_dims` | `dpo_tune_cache.py:555-568` |
| 30 | OOM at step 2/844 (2GPU DPO) | DPO needs 2 models (policy+ref) → reduce `per_device_train_batch_size: 2→1`, `gradient_accumulation_steps: 8→16` | `llama31_8b_dpo_10pct.yaml` |

---

### Key Configuration for 1 GPU vs 2 GPU vs 8 GPU

| Config | 1 GPU (Sanity) | 2 GPU (DeepSpeed Test) | 8 GPU (Full) |
|--------|----------------|------------------------|--------------|
| DeepSpeed | ❌ No | ✅ ZeRO-3 | ✅ ZeRO-3 |
| 8-bit Optimizer | ✅ true | ❌ false | ❌ false |
| Batch Size | 2 | 2 | 2 |
| Grad Accum | 64 (for eff=128) | 32 (for eff=128) | 8 (for eff=128) |
| Validates | Data, loss, NaN | NCCL, sharding, sync | Full training |
| TPS (tokens/sec) | ~2000 | ~4000 (proj) | ~9000-10000 (proj) |
| Scaling Efficiency | - | baseline | ~75-85% expected |

> **Toggle Checklist (before 2+ GPU):** ✅ DONE (Jan 30, 2026)
> ```yaml
> # In llama31_8b_sft_10pct.yaml:
> use_8bit_optimizer: false  # Changed from true → false
> ```

### 2 GPU DeepSpeed Test Results (Jan 30, 2026)

| Metric | Value | Status |
|--------|-------|--------|
| Distributed | `DistributedType.DEEPSPEED Backend: nccl` | ZeRO-3 active |
| Steps Completed | 64 | Stopped manually (Ctrl+C) |
| Loss | 1.178 → 0.757 | ~37% reduction ✅ |
| Avg TPS | 2090 tokens/sec | Good throughput |
| Time/Step | ~10 sec | ~16h for full run |
| Total Steps | 5844 | 2 epochs × 10% data |
| Memory | No OOM | Fits in 2×80GB |
| NCCL | No timeout | Communication healthy |

**2 GPU DeepSpeed test PASSED** - Ready for 8 GPU full training.

---

### 8 GPU SFT Training Results (Jan 30, 2026) ✅ COMPLETE

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| GPUs | 8× A100 80GB |
| Distributed | DeepSpeed ZeRO-3, NCCL backend |
| Precision | BF16 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 16 |
| **Effective Batch Size** | 8 × 1 × 16 = **128** |
| max_seq_length | 4096 |
| learning_rate | 5e-6 |
| num_train_epochs | 2 |
| Total Steps | 1462 |

**Dataset:**

| Metric | Value |
|--------|-------|
| Source | allenai/tulu-3-sft-mixture |
| Full Size | 939,343 samples |
| MAHALS (10%) | 93,934 samples |

**Loss Progression:**

| Step | Loss | TPS (tokens/sec) | Notes |
|------|------|------------------|-------|
| 1 | 1.19 | 5,809 | Initial (warmup) |
| 10 | 0.99 | 6,490 | -17% |
| 50 | 0.86 | 6,588 | -28% |
| 100 | 0.82 | 6,612 | -31% |
| 500 | 0.71 | 6,649 | -40% |
| 1000 | 0.56 | 6,649 | -53% |
| **1462** | **0.32** | **6,650** | **-73% (final)** |

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Training Time | 5h 11m 22s |
| Avg Time/Step | 12.8 sec |
| Avg TPS | ~6,650 tokens/sec |
| Total Tokens Processed | ~386M tokens |
| GPU Memory (per GPU) | ~39-45 GB / 80 GB (~50%) |

**Timeline:**

| Event | Timestamp |
|-------|-----------|
| Training Start | 2026-01-30 03:30:33 |
| Epoch 0 Checkpoint | 2026-01-30 06:06 |
| Training End | 2026-01-30 08:41:55 |
| HF Upload Complete | 2026-01-30 08:47 |

---

### HuggingFace Upload (Jan 30, 2026) ✅ COMPLETE

**Model URL:** https://huggingface.co/anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS

| File | Size |
|------|------|
| pytorch_model-00001-of-00004.bin | 4.98 GB |
| pytorch_model-00002-of-00004.bin | 5.00 GB |
| pytorch_model-00003-of-00004.bin | 4.92 GB |
| pytorch_model-00004-of-00004.bin | 1.17 GB |
| tokenizer.json | 17.2 MB |
| **Total** | **16.1 GB** |

**Upload Stats:**

| Metric | Value |
|--------|-------|
| Upload Speed | 85.6 MB/s |
| Format | PyTorch .bin (BF16) |
| Shards | 4 |

**Comparison with AllenAI's Official Model:**

| Model | Format | Size |
|-------|--------|------|
| allenai/Llama-3.1-Tulu-3-8B-SFT | safetensors | 16.07 GB |
| **Our SFT model** | pytorch .bin | 16.06 GB |

> **Note**: Same size because both use BF16 precision. 8B params × 2 bytes = 16 GB.

---

### Batch Size Optimization (Jan 30, 2026)

**Problem**: GPU memory at ~45% utilization, burning $8/hr for 8× A100 80GB.

**Solution**: Increase batch size from 1→2, reduce gradient accumulation 16→8 (same effective batch).

| Config | Before (v1) | After (v2) | Change |
|--------|-------------|------------|--------|
| `per_device_train_batch_size` | 1 | 2 | +100% |
| `gradient_accumulation_steps` | 16 | 8 | -50% |
| **Effective Batch Size** | 8×1×16=128 | 8×2×8=128 | Same |

**Memory Impact**:

| GPU | Before | After (projected) | Headroom |
|-----|--------|-------------------|----------|
| Highest (GPU1) | 39 GB | ~58-60 GB | ~20 GB ✅ |
| Lowest (GPU0) | 33 GB | ~52-54 GB | ~26 GB ✅ |

**Speed & Cost Impact**:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| SFT Time | ~5h | ~3.3h | 1.5x faster |
| DPO Time | ~3h | ~2h | 1.5x faster |
| Total Cost @ $8/hr | ~$64 | ~$42 | **~$22 saved** |

**Files Modified**:
- `configs/train_configs/mahals/llama31_8b_sft_10pct.yaml`
- `configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml`

> **Note**: SFT run completed with batch=1 config (5h 11m). DPO will use batch=2 config (~2h expected).

---

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

---

### 2 GPU DPO Training Results (Jan 31, 2026) ✅ COMPLETE

**Decision**: Ran full DPO on 2 GPUs instead of 8 GPUs.
- **Rationale**: ~3h total, $6 cost vs $8, sufficient for MAHALS validation
- **Trade-off**: Effective batch size = 32 (vs 128 on 8 GPUs) - acceptable for 10% subset

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| GPUs | 2× A100 80GB |
| Distributed | DeepSpeed ZeRO-3, NCCL |
| per_device_train_batch_size | 1 (Bug #30 fix) |
| gradient_accumulation_steps | 16 |
| **Effective Batch Size** | 2 × 1 × 16 = **32** |
| max_seq_length | 2048 |
| learning_rate | 5e-7 |
| warmup_ratio | 0.1 (~84 steps) |
| Total Steps | 844 |

**Loss Progression (FINAL):**

| Step | Loss | Δ from Initial | LR Phase |
|------|------|----------------|----------|
| 1 | 0.693 | baseline | Warmup start |
| 52 | 0.690 | -0.4% | Warmup |
| 103 | 0.634 | -8.6% | Peak LR (~5e-7) |
| 197 | 0.521 | -24.9% | Decay |
| 500 | ~0.45 | -35% | Decay |
| 800 | ~0.40 | -42% | Near end |
| **844** | **0.245** | **-64.6%** | **Final** |

**Timeline:**

| Event | Timestamp |
|-------|-----------|
| Training Start | 2026-01-30 23:54:39 |
| Training End | 2026-01-31 02:59:42 |
| **Total Time** | **3h 5m** |
| HF Upload | 2026-01-31 03:15 (manual) |

**Final Health Check:**

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| No OOM | Training completes | 844/844 (100%) | ✅ |
| Loss initial | ~0.69 (log(2)) | 0.6933 | ✅ |
| Loss final | < 0.5 | **0.245** | ✅ -64.6% |
| No NaN | Numeric | All numeric | ✅ |
| Model saved | Local + HF | Both complete | ✅ |

**Verdict**: DPO training COMPLETE. Loss reduced 64.6% (0.693 → 0.245). Model uploaded to HuggingFace.

---

### HuggingFace Upload - DPO (Jan 31, 2026) ✅ COMPLETE

**Model URL:** https://huggingface.co/anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS

| File | Size |
|------|------|
| pytorch_model-00001-of-00004.bin | 4.98 GB |
| pytorch_model-00002-of-00004.bin | 5.00 GB |
| pytorch_model-00003-of-00004.bin | 4.92 GB |
| pytorch_model-00004-of-00004.bin | 1.17 GB |
| tokenizer.json | 17.2 MB |
| README.md | ✅ |
| **Total** | **16.1 GB** |

---

## Phase 3: Same-Architecture Validation (NEXT)

### Model Registry Entry

```json
"Llama31_MAHALS": {
  "base": "anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS",
  "instruct": "anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS",
  "display_name": "Llama-3.1-MAHALS-8B",
  "hidden_dim": 4096,
  "num_layers": 32,
  "note": "Custom SFT→DPO pair (10% Tulu, Full FT)"
}
```

**Verification:** ✅ Both models accessible on HuggingFace (config + tokenizer loadable)

### Phase 3 Commands

```bash
# On 1x A100 80GB instance (inference only, no distributed needed)
cd /workspace/multi_agent_alignment_drift
source venv_alignment/bin/activate  # or appropriate venv

# Sanity test (~1-2 hrs)
tmux
python -u src/p03_same_arch_validation.py --mode sanity --models Llama31_MAHALS 2>&1 | tee logs/phase3_mahals_sanity.log

# Full test (~5-8 hrs)
tmux
python -u src/p03_same_arch_validation.py --mode full --models Llama31_MAHALS 2>&1 | tee logs/phase3_mahals_full.log
```

### Expected Phase 3 Output

- Steering vector: `outputs/phase2_steering_vectors/Llama31_MAHALS/steering_vector.pth`
- AQI curves: `outputs/phase3_same_arch_validation/Llama31_MAHALS/aqi_vs_lambda.png`
- Per-axiom analysis: `outputs/phase3_same_arch_validation/Llama31_MAHALS/per_axiom_curves.png`

### Success Criteria

| Metric | Target |
|--------|--------|
| AQI(λ=0) → AQI(λ=1) | Monotonic increase |
| AQI improvement | > 5 points |
| No errors | Complete run |
