# iter3 Plan B: Phase 3 Execution — 11 SFT→DPO Pairs

> **Goal**: Run `p03_same_arch_validation.py` on all 11 verified Full FT pairs
> **Note**: p03 auto-calls p02 if steering vectors are missing — no manual extraction needed

---

## Single Source of Truth

All model config lives in `src/utils/model_registry.json` (v2.1):

| Key | Params | batch_size (p03) | batch_size (p02, auto //4) |
|-----|--------|-----------------|---------------------------|
| `OLMo2_1B` | 1B | 64 | 16 |
| `Gemma2B_SFT_DPO` | 2.5B | 48 | 12 |
| `Phi2_SFT_DPO` | 2.7B | 48 | 12 |
| `Qwen25_3B_Tulu` | 3B | 32 | 8 |
| `OLMo2_7B` | 7B | 24 | 6 |
| `OLMoE_Tulu` | 7B (MoE) | 24 | 6 |
| `OLMo3_7B` | 7B | 24 | 6 |
| `Zephyr_SFT_DPO` | 7B | 24 | 6 |
| `Gemma7B_SFT_DPO` | 7B | 24 | 6 |
| `Qwen2_7B_DPOShift` | 7B | 24 | 6 |
| `Llama31_Tulu` | 8B | 16 | 4 |

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

python -u src/p03_same_arch_validation.py --mode sanity --models OLMo2_1B 2>&1 | tee logs/phase3_olmo2_1b.log

python -u src/p03_same_arch_validation.py --mode sanity --models Gemma2B_SFT_DPO 2>&1 | tee logs/phase3_gemma2b.log

python -u src/p03_same_arch_validation.py --mode sanity --models Phi2_SFT_DPO 2>&1 | tee logs/phase3_phi2.log

python -u src/p03_same_arch_validation.py --mode sanity --models Qwen25_3B_Tulu 2>&1 | tee logs/phase3_qwen25_3b.log
```

### Batch B: 7B models

```bash
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo2_7B 2>&1 | tee logs/phase3_olmo2_7b.log

python -u src/p03_same_arch_validation.py --mode sanity --models OLMoE_Tulu 2>&1 | tee logs/phase3_olmoe.log

python -u src/p03_same_arch_validation.py --mode sanity --models OLMo3_7B 2>&1 | tee logs/phase3_olmo3_7b.log

python -u src/p03_same_arch_validation.py --mode sanity --models Zephyr_SFT_DPO 2>&1 | tee logs/phase3_zephyr.log

python -u src/p03_same_arch_validation.py --mode sanity --models Gemma7B_SFT_DPO 2>&1 | tee logs/phase3_gemma7b.log

python -u src/p03_same_arch_validation.py --mode sanity --models Qwen2_7B_DPOShift 2>&1 | tee logs/phase3_qwen2_7b.log
```

### Batch C: 8B model (already validated, skip unless re-extracting)

```bash
python -u src/p03_same_arch_validation.py --mode sanity --models Llama31_Tulu 2>&1 | tee logs/phase3_llama31_tulu.log
```

### Full run (after sanity passes)

```bash
python -u src/p03_same_arch_validation.py --mode full --models OLMo2_1B Gemma2B_SFT_DPO Phi2_SFT_DPO Qwen25_3B_Tulu OLMo2_7B OLMoE_Tulu OLMo3_7B Zephyr_SFT_DPO Gemma7B_SFT_DPO Qwen2_7B_DPOShift Llama31_Tulu 2>&1 | tee logs/phase3_full_all11.log
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

## Success Criteria

| Metric | Pass | Fail |
|--------|------|------|
| AQI(λ=0) → AQI(λ=1) | Monotonic increase | Flat or decreasing |
| AQI Δ | > +5 points | < +5 points |
| No errors | Complete run | OOM / NaN / crash |
