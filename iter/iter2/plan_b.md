# iter2 Plan B: Custom SFT→DPO Training for D-STEER Validation

> **Status**: Phase 0 COMPLETE (FAILED) → Proceed to Phase 1
> **Date**: Jan 25, 2026
> **Goal**: Train custom SFT→DPO pair on Llama 3.1 8B, compare with Tulu reference

---

## Validation Ladder

| Phase | Model | Cost | Status |
|-------|-------|------|--------|
| **0** | Llama31_PKU_Custom | FREE (already trained) | ❌ **FAILED** (Δ=-2.98, not monotonic) |
| 1 | Tulu 10% subset | ~$15 / 14 hrs | ⏳ **Next Step** |
| 2 | Tulu Full | ~$160 / 6 days | If Phase 1 succeeds |

**Logic**: Start with cheapest validation, escalate only if needed.

---

## Phase 0: PKU Custom Pair (Immediate Test - FREE)

### Existing Trained Models

Already trained SFT→DPO pair from `finetuning_evaluation` project:

| Model | HuggingFace Endpoint |
|-------|---------------------|
| **SFT** | `anonymousML123/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct` |
| **DPO** | `anonymousML123/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct` |

### Training Config Used

| Parameter | SFT | DPO |
|-----------|-----|-----|
| Dataset | PKU-SafeRLHF (12K) | PKU-SafeRLHF (12K) |
| Learning Rate | 2e-4 | 1e-5 |
| Epochs | 1 | 1 |
| Beta | - | 0.1 |
| Method | LoRA | LoRA |
| Time | 40 mins | 2.2 hours |

### Registry Entry

Added to `src/utils/model_registry.json`:

```json
"Llama31_PKU_Custom": {
  "base": "anonymousML123/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct",
  "instruct": "anonymousML123/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct",
  "display_name": "Llama-3.1-PKU-Custom",
  "hidden_dim": 4096,
  "num_layers": 32,
  "note": "Custom SFT→DPO pair trained on PKU-SafeRLHF (12K samples, LoRA)"
}
```

### Run Command

```bash
python -u src/p03_same_arch_validation.py --mode sanity --models Llama31_PKU_Custom 2>&1 | tee logs/phase3_pku_custom.log
```

### Decision Matrix

| Result | Interpretation | Next Step |
|--------|----------------|-----------|
| ✅ Monotonic AQI increase | PKU training valid for D-STEER | Expand to other models |
| ❌ Flat/erratic AQI | Dataset too small OR hyperparams wrong | Proceed to Phase 1 (Tulu 10%) |

### Phase 0 Results (Jan 25, 2026)

**Command executed:**
```bash
python -u src/p03_same_arch_validation.py --mode sanity --models Llama31_PKU_Custom 2>&1 | tee logs/phase3_pku_custom.log
```

**Llama31_PKU_Custom AQI Trajectory:**

| λ | AQI |
|---|-----|
| 0.0 | 68.12 |
| 0.25 | 55.00 ↓ |
| 0.50 | 55.00 → |
| 0.75 | 58.61 ↑ |
| 1.0 | 65.14 ↑ |

**Pattern**: Erratic U-shaped curve (drops at λ=0.25, recovers but never exceeds baseline)

**Result**: ❌ **FAILED** - AQI decreased by 2.98 points, NOT monotonic

**All 5 Models Comparison:**

| Model | AQI(λ=0) | AQI(λ=1) | Δ | Monotonic |
|-------|----------|----------|---|-----------|
| **Llama31_Tulu** | 58.01 | 79.84 | **+21.83** | ✅ YES |
| Zephyr_7B | 65.86 | 67.78 | +1.92 | ❌ NO |
| Llama31_PKU_Custom | 68.12 | 65.14 | -2.98 | ❌ NO |
| Mistral_7B | 69.30 | 57.94 | -11.36 | ❌ NO |
| Falcon_7B | 29.50 | 15.29 | -14.21 | ❌ NO |

**Key Finding**: Only **Llama31_Tulu** (official AllenAI SFT→DPO pair) shows monotonic AQI improvement. All other models show erratic or declining AQI.

**Likely Causes for PKU Failure:**
1. **Dataset size**: PKU-SafeRLHF (12K) vs Tulu (939K) = 78× smaller
2. **Hyperparameters**: SFT LR 40× higher, DPO Beta 50× lower than Tulu
3. **Training method**: LoRA vs Full fine-tuning

**Conclusion**: D-STEER requires a **true SFT→DPO training relationship** with proper hyperparameters. Proceed to Phase 1 (Tulu 10% subset).

---

## Context

### iter1 Result (Tulu Reference)

Tulu SFT→DPO pair produced **monotonic AQI increase**:

| λ | AQI |
|---|-----|
| 0.0 | 58.0 |
| 0.25 | 68.0 |
| 0.50 | 71.5 |
| 0.75 | 72.1 |
| 1.0 | **79.8** |

**Δ = +21.8 points** ✅ D-STEER validated

### iter2 Goal

Train our own SFT→DPO pair and verify we get similar results before expanding to other models.

---

## Llama31_Tulu (AllenAI Official) - Full Training Configuration

### SFT Stage

| Parameter | Value |
|-----------|-------|
| **Base Model** | `meta-llama/Llama-3.1-8B` |
| **Dataset** | `allenai/tulu-3-sft-mixture` |
| **Dataset Size** | 939,344 samples |
| **Learning Rate** | 5e-6 |
| **LR Schedule** | Linear |
| **Warmup Ratio** | 0.03 (3%) |
| **Epochs** | 2 |
| **Effective Batch Size** | 128 |
| **Max Sequence Length** | 4096 |
| **Method** | Full fine-tuning |
| **Precision** | BF16 |
| **Output** | `allenai/Llama-3.1-Tulu-3-8B-SFT` |

### DPO Stage

| Parameter | Value |
|-----------|-------|
| **Base Model** | `allenai/Llama-3.1-Tulu-3-8B-SFT` |
| **Dataset** | `allenai/llama-3.1-tulu-3-8b-preference-mixture` |
| **Learning Rate** | 5e-7 |
| **LR Schedule** | Linear |
| **Warmup Ratio** | 0.1 (10%) |
| **Epochs** | 1 |
| **Effective Batch Size** | 128 |
| **Max Sequence Length** | 2048 |
| **Beta (KL Penalty)** | 5.0 |
| **Method** | Full fine-tuning |
| **Precision** | BF16 |
| **Output** | `allenai/Llama-3.1-Tulu-3-8B-DPO` |

### Comparison: Tulu Official vs PKU Custom (Failed)

| Parameter | Tulu Official | PKU Custom | Ratio |
|-----------|---------------|------------|-------|
| **SFT Dataset** | 939K | 12K | 78× |
| **SFT LR** | 5e-6 | 2e-4 | 40× lower |
| **SFT Epochs** | 2 | 1 | 2× |
| **DPO LR** | 5e-7 | 1e-5 | 20× lower |
| **DPO Beta** | 5.0 | 0.1 | 50× higher |
| **Method** | Full FT | LoRA | ~4× compute |
| **AQI Δ** | **+21.83** | **-2.98** | ❌ |

---

## Current Baseline (Your Scripts)

| Phase | Dataset | Time (A40 40GB) |
|-------|---------|-----------------|
| SFT | PKU-SafeRLHF (12K), 1 epoch | 40 mins |
| DPO | PKU-SafeRLHF (12K), 1 epoch | 2.2 hours |
| **Total** | | **~3 hours** |

### Gap Analysis

| Aspect | Your Scripts | Tulu 3 Official | Gap |
|--------|--------------|-----------------|-----|
| SFT Dataset | 12K | 939K | 78× |
| SFT Epochs | 1 | 2 | 2× |
| SFT LR | 2e-4 | 5e-6 | 40× lower |
| DPO LR | 1e-5 | 5e-7 | 20× lower |
| DPO Beta | 0.1 | 5.0 | 50× higher |
| Method | LoRA | Full FT | ~4× compute |

---

## Compute Scaling Analysis

### If Replicating Tulu Exactly (Full FT on Single A40)

| Phase | Scaling Factor | Estimated Time |
|-------|----------------|----------------|
| SFT | 78× data × 2× epochs × 4× full FT = 624× | ~17 days |
| DPO | 78× data × 4× full FT × 2× ref model = 624× | ~29 days |
| **Total** | | **~46 days** |

⚠️ **Problem**: Full fine-tuning needs ~60GB+ VRAM → Won't fit on A40 40GB

---

## Practical Options

### Option A: LoRA + Full Tulu Datasets

Use Tulu's datasets with LoRA (fits on A40):

| Phase | Config | Estimated Time |
|-------|--------|----------------|
| SFT | 939K samples, 2 epochs, LoRA, LR=5e-6 | ~52 hours |
| DPO | Full preference mix, 1 epoch, LoRA, LR=5e-7, β=5 | ~86 hours |
| **Total** | | **~6 days** |

### Option B: LoRA + 10% Tulu Datasets (Recommended for Validation)

Sample 10% of Tulu datasets:

| Phase | Config | Estimated Time |
|-------|--------|----------------|
| SFT | 94K samples, 2 epochs, LoRA, LR=5e-6 | ~5 hours |
| DPO | 10% preference mix, 1 epoch, LoRA, LR=5e-7, β=5 | ~9 hours |
| **Total** | | **~14 hours** |

### Option C: Multi-GPU Full Fine-tuning

Rent 8× A100 80GB cluster:

| Phase | Config | Time | Cost (~$2/GPU-hr) |
|-------|--------|------|-------------------|
| SFT | 939K, 2 epochs, DeepSpeed ZeRO-3 | ~8 hours | ~$128 |
| DPO | Full mix, 1 epoch, DeepSpeed ZeRO-3 | ~8 hours | ~$128 |
| **Total** | | **~16 hours** | **~$256** |

---

## Recommendation

### Phase 1: Quick Validation (Option B)

```
Goal: Verify training pipeline produces monotonic AQI curve
Time: ~14 hours on single A40
Cost: ~$15 (Lambda Labs A40 @ $1.10/hr)
```

**Success Criteria**: AQI should increase monotonically with λ (like Tulu)

### Phase 2: Full Reproduction (Option A or C)

Only if Phase 1 succeeds:
- Option A (~6 days) for budget-friendly
- Option C (~16 hours, ~$256) for exact reproduction

---

## Decision: Use open-instruct Instead of Custom Scripts

### Why Switch?

| Original Plan | New Plan |
|---------------|----------|
| Copy scripts from `finetuning_evaluation` | Use AllenAI's `open-instruct` directly |
| Write custom Python code | Just modify YAML configs |
| Risk of bugs, debugging | Zero risk (same code trained Tulu 3) |
| ~1-2 days coding | ~1 hour setup |

### Industry Standard Validation

Web search confirms this is the standard approach among ML researchers in 2025:

- **open-instruct** is the exact codebase that trained Tulu 3 (published at COLM 2025)
- **Framework > Custom Scripts**: "Teams can use existing supervised learning frameworks... training process is more stable" — [HuggingFace TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- **SFT → DPO is established workflow**: "Combined SFT-then-DPO workflow converges faster and yields higher-quality results" — [OpenAI Cookbook](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)

### Submodule Added

```bash
# Already added to MAHALS repo
git submodule add https://github.com/allenai/open-instruct.git literature/tulu_train
git submodule add https://github.com/allenai/olmes.git literature/tulu_eval
```

---

## Execution Plan (Using open-instruct)

### Step 1: Create Custom Config Files

Create configs in `literature/tulu_train/configs/train_configs/mahals/`:

**llama31_8b_sft_10pct.yaml:**
```yaml
model_name_or_path: meta-llama/Llama-3.1-8B
dataset_mixer:
    allenai/tulu-3-sft-mixture: 0.1  # 10% subset
learning_rate: 5.0e-06
num_train_epochs: 2
warmup_ratio: 0.03
lr_scheduler_type: linear
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # effective batch 128 on 8 GPUs
max_seq_length: 2048
```

**llama31_8b_dpo_10pct.yaml:**
```yaml
model_name_or_path: <path_to_sft_output>
dataset_mixer:
    allenai/llama-3.1-tulu-3-8b-preference-mixture: 0.1
learning_rate: 5.0e-07
num_epochs: 1
warmup_ratio: 0.1
beta: 5
max_seq_length: 2048
```

### Step 2: Run Training (on GPU machine)

```bash
cd literature/tulu_train

# SFT Training
./scripts/finetune_with_accelerate.sh \
    --config configs/train_configs/mahals/llama31_8b_sft_10pct.yaml

# DPO Training
./scripts/dpo_train_with_accelerate.sh \
    --config configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml
```

### Step 3: Register in MAHALS

```json
{
  "Llama31_Tulu10pct": {
    "base": "kapilw25/Llama31_Tulu10pct_SFT",
    "instruct": "kapilw25/Llama31_Tulu10pct_DPO",
    "hidden_dim": 4096,
    "num_layers": 32,
    "note": "Custom SFT→DPO pair (Tulu 10% subset, open-instruct)"
  }
}
```

### Step 4: Validate

```bash
python src/p03_same_arch_validation.py --mode sanity --models Llama31_Tulu10pct
```

### Step 5: Compare

| Model | AQI(λ=0) | AQI(λ=1) | Δ | Monotonic |
|-------|----------|----------|---|-----------|
| Llama31_Tulu (reference) | 58.01 | 79.84 | +21.83 | ✅ |
| Llama31_PKU_Custom (Phase 0) | 68.12 | 65.14 | -2.98 | ❌ |
| Llama31_Tulu10pct (Phase 1) | ? | ? | ? | ? |

---

## References

- [Tülu 3 Technical Blog](https://allenai.org/blog/tulu-3-technical)
- [Tülu 3 Paper (arXiv:2411.15124)](https://arxiv.org/pdf/2411.15124)
- [AllenAI open-instruct GitHub](https://github.com/allenai/open-instruct)
- [Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT)
- [Llama-3.1-Tulu-3-8B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO)

---

## Next Step

**→ See [plan_c.md](plan_c.md) for Phase 1 implementation details**
