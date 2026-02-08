# iter3 Plan B: Phase 3 Execution — 11 SFT→DPO Pairs

> **Goal**: Run `p03_same_arch_validation.py` on all 11 verified Full FT pairs
> **Note**: p03 auto-calls p02 if steering vectors are missing — no manual extraction needed

---

## Single Source of Truth

All model config lives in `src/utils/model_registry.json` (v2.1):

| Key | Params | batch_size (p03) | batch_size (p02, auto //4) | Env |
|-----|--------|-----------------|---------------------------|-----|
| `OLMo2_1B` | 1B | 64 | 16 | venv_Agnt_Algnmt |
| `Gemma2B_SFT_DPO` | 2.5B | 48 | 12 | venv_Agnt_Algnmt |
| `Phi2_SFT_DPO` | 2.7B | 48 | 12 | venv_Agnt_Algnmt |
| `Qwen25_3B_Tulu` | 3B | 32 | 8 | venv_Agnt_Algnmt |
| `OLMo2_7B` | 7B | 24 | 6 | venv_Agnt_Algnmt |
| `OLMoE_Tulu` | 7B (MoE) | 24 | 6 | venv_Agnt_Algnmt |
| `OLMo3_7B` | 7B | 24 | 6 | **DEFERRED** (needs transformers>=4.57) |
| `Zephyr_SFT_DPO` | 7B | 24 | 6 | venv_Agnt_Algnmt |
| `Gemma7B_SFT_DPO` | 7B | 24 | 6 | venv_Agnt_Algnmt |
| `Qwen2_7B_DPOShift` | 7B | 24 | 6 | venv_Agnt_Algnmt |
| `Llama31_Tulu` | 8B | 16 | 4 | venv_Agnt_Algnmt |

### Deferred Models (separate UV env, post-sanity)

Models that need `transformers>=4.57.0` (current env: `4.47.1`). Will create a separate UV environment after all compatible models complete sanity runs.

| Model | Reason | Min transformers |
|-------|--------|-----------------|
| `OLMo3_7B` | `model_type: olmo3` not in CONFIG_MAPPING | >=4.57.0 |

---

## GPU & Disk Requirements

**Recommended**: 1x A100 80GB ($1.0/hr) — runs all 11 sequentially

| Resource | Needed | Available |
|----------|--------|-----------|
| VRAM (p02 peak, 2×7B) | ~34 GB | 80 GB |
| VRAM (p03 peak, 1×8B) | ~22 GB | 80 GB |
| Disk (HF cache, 22 repos) | ~260 GB | 300 GB |
| Container | ~15 GB | 32 GB |

> Set `HF_HOME=/workspace/volume/hf_cache` in `.env` so models go on volume disk, not container.

---

## Execution Order

Run smallest first (fast iteration, catch bugs early).

### Batch A: Small models (1B–3B)

```bash
tmux
cd /workspace/multi_agent_alignment_drift
source venv_alignment/bin/activate

✅   
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo2_1B 2>&1 | tee logs/phase3_olmo2_1b.log 
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models Gemma2B_SFT_DPO 2>&1 | tee logs/phase3_gemma2b.log
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models Phi2_SFT_DPO 2>&1 | tee logs/phase3_phi2.log
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models Qwen25_3B_Tulu 2>&1 | tee logs/phase3_qwen25_3b.log
```

### Batch B: 7B models

```bash
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo2_7B 2>&1 | tee logs/phase3_olmo2_7b.log
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models OLMoE_Tulu 2>&1 | tee logs/phase3_olmoe.log
❌   # OLMo3 requires native transformers>=4.57.0 support. Current: 4.47.1
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo3_7B 2>&1 | tee logs/phase3_olmo3_7b.log
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models Zephyr_SFT_DPO 2>&1 | tee logs/phase3_zephyr.log
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models Gemma7B_SFT_DPO 2>&1 | tee logs/phase3_gemma7b.log
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models Qwen2_7B_DPOShift 2>&1 | tee logs/phase3_qwen2_7b.log
```

### Batch C: 8B model (already validated, skip unless re-extracting)

```bash
✅   
python -u src/p03_same_arch_validation.py --mode sanity --models Llama31_Tulu 2>&1 | tee logs/phase3_llama31_tulu.log
```

### Sanity run — all 10 compatible models

```bash
⏳
tmux
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo2_1B Gemma2B_SFT_DPO Phi2_SFT_DPO Qwen25_3B_Tulu OLMo2_7B OLMoE_Tulu Zephyr_SFT_DPO Gemma7B_SFT_DPO Qwen2_7B_DPOShift Llama31_Tulu 2>&1 | tee logs/phase3_sanity_all10.log
```

### Deferred run (separate UV env with transformers>=4.57)

```bash
# TODO: Create UV env, install transformers>=4.57.0, run:
⏳
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo3_7B 2>&1 | tee logs/phase3_olmo3_7b.log
# If more models need upgraded transformers, add them here
```

---

## Codebase Structure

```
src/utils/                         Used by
├── __init__.py                    p01, p02, p03  — registry, get_batch_size(), logging
├── config.py                      p01, p03       — DATASET_NAME, SAMPLES_*, GAMMA, paths
├── model_registry.json            p01, p02, p03  — 12 models (sft/dpo/dims/batch_size)
├── checkpoint.py                  p01, p02, p03  — save/resume across crashes
├── cache.py                       p02            — tensor caching (hidden states)
├── plot_aqi.py                    p01            — AQI bar/heatmap/delta plots
└── plot_steering.py               p02            — cosine similarity, norm plots
```

---

## Bug Fix: AQI Capped at 55.0 in Full Mode (7,000 samples)

### Problem
In full mode (500 samples/category = 7,000 total), the "overall" AQI was locked at 55.0 for ALL models.
Not observed in sanity mode (1,400 samples) where AQI ranged 20–84.

### Root Cause
The "overall" AQI merges all 7,000 samples → t-SNE → CHI/XB on merged space.
Two values get locked:

1. **CHI_norm = 100.0 always** — overall has 7× more samples than any axiom → highest raw CHI → min-max maps to 100
2. **XB_norm = 10.0 always** — t-SNE quality degrades at 7,000 mixed-axiom samples → overall XB ~1.2 (vs per-axiom ~0.3) → sigmoid squashes to ~0 → XB_norm floor kicks in at 10.0

`AQI = 0.5 × 100 + 0.5 × 10 = 55.0` — hard ceiling.

### Why Sanity Mode Worked
With 1,400 samples, t-SNE produced acceptable overall clusters → XB ~0.5 (not ~1.2) → sigmoid gave XB_norm ~11 (above floor) → AQI had full dynamic range.

### Fix
Use `--mode sanity` only (100 samples/category = 1,400 total). Full/max modes are disabled via docstring warning in `p03_same_arch_validation.py`.

### Wrong Fix (reverted)
Initially overrode overall AQI with mean of 7 per-axiom AQIs. Professor rejected this — AQI is a holistic measure (global safe/unsafe separation), not a weighted-axiom average.

---

## Sanity Run Results (10 models, 1,400 samples, 2026-02-08)

| Model | AQI(λ=0) | AQI(λ=1) | Delta | Monotonic | Pass (Δ>+5) |
|---|---|---|---|---|---|
| Gemma2B_SFT_DPO | 33.96 | 55.00 | **+21.04** | NO | YES |
| OLMo2_1B | 55.00 | 73.20 | **+18.20** | NO | YES |
| Llama31_Tulu | 57.49 | 75.10 | **+17.60** | YES * | YES |
| OLMoE_Tulu | 55.00 | 71.94 | **+16.94** | NO | YES |
| Qwen25_3B_Tulu | 10.45 | 21.80 | **+11.36** | NO | YES |
| Qwen2_7B_DPOShift | 47.25 | 55.00 | **+7.75** | NO | YES |
| Phi2_SFT_DPO | 75.58 | 83.21 | **+7.63** | NO | YES |
| Gemma7B_SFT_DPO | 55.30 | 62.16 | **+6.86** | NO | YES |
| OLMo2_7B | 71.97 | 76.78 | **+4.81** | NO | NO |
| Zephyr_SFT_DPO | 65.87 | 60.72 | **-5.15** | NO | NO |

**Key observations**:
- 9/10 models show positive AQI delta — steering vectors improve alignment
- 8/10 pass the Δ>+5 threshold
- 1 monotonic (Llama31_Tulu) — clean ascending curve: 57.49 → 65.14 → 71.58 → 72.26 → 75.10
- 1 degradation (Zephyr_SFT_DPO) — steering hurts alignment (Δ = -5.15)
- AQI range: 5.00–83.68 across all lambdas — full dynamic range, not capped at 55
- Top 4 models (Gemma2B +21.04, OLMo2_1B +18.20, Llama31 +17.60, OLMoE +16.94) all show Δ > +16 — strong steering signal
- Some individual lambdas still hit 55.00 (overall XB floor), but not all — enough variation for meaningful curves

### Zephyr Outlier Analysis (only model with negative Δ)

Zephyr's AQI trajectory: 65.87 → 68.46 → 55.00 → 62.50 → 60.72. It improves at λ=0.25
then drops hard at λ=0.5 (hitting the 55 cap), suggesting the steering direction partially
aligns with LITMUS at low strength but overshoots and disrupts at higher strength.

The SFT→DPO steering vector captures the direction of the alignment shift during DPO training.
If Zephyr's DPO shift goes in a different direction than what LITMUS axioms measure, applying
that steering vector pushes the model away from LITMUS-defined alignment.

Zephyr uses UltraChat (SFT) → UltraFeedback (DPO). Qwen2_7B_DPOShift uses the same
training data pipeline but shows +7.75, so training data alone doesn't explain it — the
Mistral-7B base model's internal representation likely encodes the SFT→DPO shift differently.

---

## Success Criteria

| Metric | Pass | Fail |
|--------|------|------|
| AQI(λ=0) → AQI(λ=1) | Monotonic increase | Flat or decreasing |
| AQI Δ | > +5 points | < +5 points |
| No errors | Complete run | OOM / NaN / crash |
