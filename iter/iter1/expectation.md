# Phase 1: Expected Outputs

> **Scripts**: `src/m01_config.py`, `src/m02_measure_baseline_aqi.py`
> **Execution**: GPU Server Only (Nvidia CUDA)
> **GPU**: A10 24GB (sequential processing, 1 model at a time)

---

## m01_config.py

**Command**: `python src/m01_config.py`

**Expected Output**:
```
==================================================
Phase 1: Multi-Architecture AQI Config
==================================================
Models: ['Llama3_8B', 'Phi3_Mini', 'Mistral_7B', 'Qwen2_7B', 'Gemma2_9B']
Dataset: hasnat79/litmus
Output: /path/to/outputs/phase1_baseline_aqi
```

---

## m02_measure_baseline_aqi.py

### Sanity Mode (Quick Test)

**Command**: `python src/m02_measure_baseline_aqi.py --mode sanity`

**Expected Runtime**: ~30-60 minutes (100 samples per category)

**Expected Console Output**:
```
GPU: NVIDIA A10
Memory: 24.0 GB

============================================================
Phase 1: Multi-Architecture AQI Baseline (GPU Only)
============================================================
Mode: sanity | Samples: 100
Models: ['Llama3_8B', 'Phi3_Mini', 'Mistral_7B', 'Qwen2_7B', 'Gemma2_9B']
Output: outputs/phase1_baseline_aqi

Loading dataset: hasnat79/litmus
Loaded 700 samples

============================================================
Evaluating: Llama-3.1-8B
HF Repo: meta-llama/Llama-3.1-8B-Instruct
============================================================
...
AQI: 72.45

============================================================
Evaluating: Phi-3-Mini
HF Repo: microsoft/Phi-3-mini-4k-instruct
============================================================
...
AQI: 68.12

[...repeat for all 5 models...]

============================================================
PHASE 1 RESULTS: Baseline AQI Scores
============================================================
Model           AQI        CHI_norm     XB_norm
------------------------------------------------------------
Mistral_7B      75.23      0.82         0.68
Llama3_8B       72.45      0.78         0.67
Gemma2_9B       71.89      0.76         0.68
Qwen2_7B        69.34      0.74         0.65
Phi3_Mini       68.12      0.72         0.64
------------------------------------------------------------
BASELINE MODEL: Mistral_7B (AQI=75.23)
============================================================

Summary: outputs/phase1_baseline_aqi/phase1_summary.json
```

### Full Mode

**Command**: `python src/m02_measure_baseline_aqi.py --mode full`

**Expected Runtime**: ~2-3 hours (200 samples per category)

---

## Expected File Outputs

```
outputs/phase1_baseline_aqi/
├── phase1_summary.json          # Overall summary with baseline model
├── Llama3_8B/
│   ├── result.json              # AQI, CHI_norm, XB_norm
│   ├── embeddings_Llama3_8B.pkl # Cached embeddings
│   └── metrics_summary.json     # Per-axiom breakdown
├── Phi3_Mini/
│   ├── result.json
│   ├── embeddings_Phi3_Mini.pkl
│   └── metrics_summary.json
├── Mistral_7B/
│   ├── result.json
│   ├── embeddings_Mistral_7B.pkl
│   └── metrics_summary.json
├── Qwen2_7B/
│   ├── result.json
│   ├── embeddings_Qwen2_7B.pkl
│   └── metrics_summary.json
└── Gemma2_9B/
    ├── result.json
    ├── embeddings_Gemma2_9B.pkl
    └── metrics_summary.json
```

### phase1_summary.json

```json
{
  "baseline_model": "Mistral_7B",
  "baseline_aqi": 75.23,
  "all_scores": {
    "Mistral_7B": 75.23,
    "Llama3_8B": 72.45,
    "Gemma2_9B": 71.89,
    "Qwen2_7B": 69.34,
    "Phi3_Mini": 68.12
  },
  "timestamp": "2026-01-17T14:30:00.000000"
}
```

### Individual result.json

```json
{
  "model_key": "Llama3_8B",
  "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
  "aqi_score": 72.45,
  "chi_norm": 0.78,
  "xb_norm": 0.67,
  "n_samples": 700
}
```

---

## GPU Memory Usage (Sequential Processing)

Models are loaded **one at a time** with explicit unloading between evaluations:

```
┌────────────────────────────────────────────────────────────┐
│                    A10 24GB VRAM                           │
├────────────────────────────────────────────────────────────┤
│ Model           │ Size (fp16) │ + Overhead │ Peak VRAM    │
├─────────────────┼─────────────┼────────────┼──────────────┤
│ Llama3_8B       │ ~16 GB      │ ~4 GB      │ ~20 GB       │
│ Phi3_Mini       │ ~8 GB       │ ~3 GB      │ ~11 GB       │
│ Mistral_7B      │ ~14 GB      │ ~4 GB      │ ~18 GB       │
│ Qwen2_7B        │ ~14 GB      │ ~4 GB      │ ~18 GB       │
│ Gemma2_9B       │ ~18 GB      │ ~4 GB      │ ~22 GB       │
└─────────────────┴─────────────┴────────────┴──────────────┘

Processing Flow:
  Load Llama → Evaluate → Unload → torch.cuda.empty_cache()
  Load Phi3  → Evaluate → Unload → torch.cuda.empty_cache()
  ... (repeat for all 5 models)
```

**Peak VRAM**: ~22 GB (Gemma2_9B) — fits in A10 24GB

---

## Success Criteria

| Criterion | Expected |
|-----------|----------|
| All 5 models evaluated | Yes (sequential) |
| AQI scores in range [0-100] | Yes |
| Baseline model identified | Highest AQI |
| No CUDA OOM errors | A10 24GB, batch size = 4 |
| Embeddings cached | .pkl files saved |
| Peak VRAM usage | < 24 GB |

---

## Expected AQI Range (Hypothesis)

Based on D-STEER paper and general alignment literature:

| Model | Expected AQI Range | Rationale |
|-------|-------------------|-----------|
| Llama-3.1-8B | 70-80 | Strong RLHF from Meta |
| Phi-3-Mini | 65-75 | Microsoft's safety focus |
| Mistral-7B | 70-80 | Known for good alignment |
| Qwen2-7B | 65-75 | Alibaba's multilingual focus |
| Gemma-2-9B | 70-80 | Google's safety guidelines |

**Key Observation**: All models should show AQI > 60 (aligned). Differences of 5-15 points represent the **natural alignment differential** (Type 1 drift from plan.md).

---

## Next Steps (After Phase 1)

Once baseline AQI scores are established:

1. **Identify baseline model** (highest AQI) - use for steering vector extraction
2. **Document per-axiom breakdown** - identify which dimensions differ most
3. **Proceed to Phase 2** - Extract steering vectors using D-STEER approach
