# iter2 Plan D: FSDP2 Migration for DPO Training

> **Status**: PLANNED
> **Date**: Jan 30, 2026
> **Prerequisite**: Complete SFT training (Plan C), fix DPO `mixer_list` bug

---

## Why FSDP2 over DeepSpeed ZeRO-3?

| Aspect | DeepSpeed ZeRO-3 (Current) | FSDP2 (Proposed) |
|--------|---------------------------|------------------|
| **Sweet Spot** | 70B+ models | 1B-10B models |
| **Our Models** | Overkill for most | Better fit |
| **Expected Speedup** | Baseline | +20-50% TPS |
| **PyTorch Native** | No | Yes (torch 2.0+) |
| **Checkpoint Compat** | DeepSpeed format | Native PyTorch |
| **MoE Support** | Good | Improving (torch 2.2+) |
| **SSM Support** | Unknown | Limited (architecture risk) |

---

## Target Models from D-STEER Feasibility Matrix

### Tier 1: Standard Transformers (FSDP2 Ready)

| Model | Size | Architecture | FSDP2 Wrapper Class | Priority |
|-------|------|--------------|---------------------|----------|
| Llama 3.1 8B | 8B | Transformer | `LlamaDecoderLayer` | ✅ Current |
| Llama 3.2 1B | 1B | Transformer | `LlamaDecoderLayer` | High |
| Phi-4 | 14B | Transformer | `PhiDecoderLayer` | High |
| Gemma 3 2B | 2B | Transformer | `GemmaDecoderLayer` | Medium |
| DeepSeek-V3 | 671B | Transformer+MoE | Complex | Low |
| DeepSeek-R1 | 671B | Transformer+MoE | Complex | Low |

### Tier 2: MoE Models (FSDP2 with Caution)

| Model | Size | Architecture | Notes |
|-------|------|--------------|-------|
| Mixtral 8x7B | 46.7B | MoE-Transformer | Expensive, needs expert sharding |
| OLMoE | 6.9B | MoE | Smaller, more feasible |

### Tier 3: SSMs (Architecture Risk - Keep DeepSpeed)

| Model | Architecture | FSDP2 Status | Recommendation |
|-------|--------------|--------------|----------------|
| Jamba | Hybrid (SSM+Attn) | ⚠️ Untested | Stay with DeepSpeed |
| Mamba | Pure SSM | ⚠️ Untested | Stay with DeepSpeed |
| FalconMamba | Hybrid | ⚠️ Untested | Stay with DeepSpeed |
| Hymba | Hybrid | ⚠️ Untested | Stay with DeepSpeed |

---

## FSDP2 Configuration Files

### 1. Create Accelerate FSDP Config

**File**: `configs/ds_configs/fsdp2_llama.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8

fsdp_config:
  # Sharding Strategy (equivalent to ZeRO-3)
  fsdp_sharding_strategy: FULL_SHARD

  # Auto-wrap Transformer layers
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

  # Memory optimization
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_use_orig_params: true

  # No CPU offload (we have 80GB GPUs)
  fsdp_offload_params: false
  fsdp_cpu_ram_efficient_loading: true

  # Checkpointing
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
```

### 2. Create DPO Training Script for FSDP2

**File**: `scripts/dpo_train_with_fsdp2_mahals.sh`

```bash
#!/bin/bash

# MAHALS: DPO training with FSDP2 (PyTorch native)
#
# Usage:
# sh scripts/dpo_train_with_fsdp2_mahals.sh 8 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_gpus> <config_file>"
    exit 1
fi

NUM_GPUS="$1"
CONFIG_FILE="$2"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

echo "FSDP2 Mode: $NUM_GPUS GPUs"
echo "Config: $CONFIG_FILE"

accelerate launch \
    --config_file configs/ds_configs/fsdp2_llama.yaml \
    --num_processes $NUM_GPUS \
    open_instruct/dpo.py \
    "$CONFIG_FILE"
```

---

## Model-Specific FSDP2 Configs

### Llama Family (3.1 8B, 3.2 1B)

```yaml
fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

### Phi-4

```yaml
fsdp_transformer_layer_cls_to_wrap: PhiDecoderLayer
# Note: Verify class name in transformers library
```

### Gemma 3 2B

```yaml
fsdp_transformer_layer_cls_to_wrap: GemmaDecoderLayer
```

### Mixtral 8x7B (MoE - Special Handling)

```yaml
fsdp_transformer_layer_cls_to_wrap: MixtralDecoderLayer
# WARNING: MoE experts need careful sharding
# Consider: fsdp_sharding_strategy: HYBRID_SHARD for expert locality
```

---

## Migration Steps

### Phase 1: Validate FSDP2 on Llama 3.1 8B DPO (After SFT)

```bash
# Step 1: Fix mixer_list bug in dpo_utils.py (see Plan C bug #10)
# Change line 167: mixer_list: list[str] = field(default=None)

# Step 2: Create FSDP2 config
cp configs/ds_configs/fsdp2_llama.yaml configs/ds_configs/fsdp2_llama.yaml

# Step 3: Sanity test (2 GPUs)
sh scripts/dpo_train_with_fsdp2_mahals.sh 2 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee logs/dpo_fsdp2_2gpu_test.log

# Step 4: Compare metrics
# - TPS: Should be higher than DeepSpeed (~20-50%)
# - Memory: Should be similar (~36GB per GPU)
# - Loss: Should be identical trajectory

# Step 5: Full training (8 GPUs)
sh scripts/dpo_train_with_fsdp2_mahals.sh 8 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml 2>&1 | tee logs/dpo_fsdp2_training.log
```

### Phase 2: Extend to Other Models

| Model | Config File | Wrapper Class |
|-------|-------------|---------------|
| Llama 3.2 1B | `fsdp2_llama.yaml` | `LlamaDecoderLayer` |
| Phi-4 | `fsdp2_phi.yaml` | `PhiDecoderLayer` |
| Gemma 3 2B | `fsdp2_gemma.yaml` | `GemmaDecoderLayer` |

### Phase 3: MoE Models (If Needed)

```bash
# Mixtral requires hybrid sharding for expert parallelism
# Create separate config: fsdp2_mixtral.yaml
# Test thoroughly before full training
```

---

## Expected Performance Comparison

| Model | DeepSpeed ZeRO-3 TPS | FSDP2 TPS (Est.) | Speedup |
|-------|---------------------|------------------|---------|
| Llama 3.1 8B | ~6,500 | ~8,000-10,000 | +23-54% |
| Llama 3.2 1B | ~15,000 | ~20,000-25,000 | +33-67% |
| Phi-4 (14B) | ~4,000 | ~5,500-6,500 | +38-63% |
| Gemma 3 2B | ~12,000 | ~16,000-20,000 | +33-67% |

*Estimates based on FSDP vs DeepSpeed benchmarks for similar model sizes*

---

## Fallback Strategy

If FSDP2 causes issues:

1. **Checkpoint incompatibility**: Convert using `torch.distributed.checkpoint`
2. **OOM errors**: Reduce batch size or enable `fsdp_offload_params: true`
3. **MoE issues**: Fall back to DeepSpeed ZeRO-3 for Mixtral/OLMoE
4. **SSM models**: Always use DeepSpeed (untested with FSDP2)

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `configs/ds_configs/fsdp2_llama.yaml` | CREATE | FSDP2 config for Llama family |
| `configs/ds_configs/fsdp2_phi.yaml` | CREATE | FSDP2 config for Phi-4 |
| `configs/ds_configs/fsdp2_gemma.yaml` | CREATE | FSDP2 config for Gemma |
| `scripts/dpo_train_with_fsdp2_mahals.sh` | CREATE | FSDP2 training script |
| `open_instruct/dpo_utils.py:167` | MODIFY | Fix `mixer_list` default to `None` |

---

## Decision Matrix: When to Use What

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Architecture                        │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│ Transformer │ MoE         │ SSM/Hybrid  │ Size > 70B        │
│ (Standard)  │             │             │                   │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ FSDP2 ✓     │ DeepSpeed ✓ │ DeepSpeed ✓ │ DeepSpeed ✓       │
│             │ (or FSDP2   │ (untested   │ (better scaling)  │
│             │  with care) │  with FSDP) │                   │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```

---

## Next Steps

1. ⏳ Wait for SFT training to complete (Plan C)
2. 🔧 Fix DPO `mixer_list` bug
3. 📝 Create FSDP2 configs (this plan)
4. 🧪 Run FSDP2 sanity test on DPO
5. 📊 Compare DeepSpeed vs FSDP2 performance
6. 🚀 Full DPO training with winner
