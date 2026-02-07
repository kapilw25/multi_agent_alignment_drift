# iter3 Plan B: Phase 3 Execution — 11 SFT→DPO Pairs

> **Goal**: Run `p03_same_arch_validation.py` on all 11 verified Full FT pairs
> **Note**: p03 auto-calls p02 if steering vectors are missing — no manual extraction needed

---

## GPU Requirements

| Batch | Key | Params | BF16 (1 model) | Min GPU |
|-------|-----|--------|----------------|---------|
| A | `OLMo2_1B` | 1B | ~2 GB | Any GPU |
| A | `Gemma2B_SFT_DPO` | 2.5B | ~5 GB | Any GPU |
| A | `Phi2_SFT_DPO` | 2.7B | ~5.4 GB | Any GPU |
| A | `Qwen25_3B_Tulu` | 3B | ~6 GB | Any GPU |
| B | `OLMo2_7B` | 7B | ~14 GB | A40 40GB |
| B | `OLMoE_Tulu` | 7B total | ~14 GB | A40 40GB |
| B | `OLMo3_7B` | 7B | ~14 GB | A40 40GB |
| B | `Zephyr_SFT_DPO` | 7B | ~14 GB | A40 40GB |
| B | `Gemma7B_SFT_DPO` | 7B | ~14 GB | A40 40GB |
| B | `Qwen2_7B_DPOShift` | 7B | ~15 GB | A40 40GB |
| C | `Llama31_Tulu` | 8B | ~16 GB | A40 40GB |

> p03 loads 1 model (SFT + hook). If steering vector missing, auto-calls p02 (loads SFT+DPO, ~2x VRAM).

---

## Execution Order

Run smallest first (fast iteration, catch bugs early).

### Batch A: Small models (Any GPU)

```bash
tmux
cd /workspace/multi_agent_alignment_drift
source venv_alignment/bin/activate

# 1B — fastest
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo2_1B 2>&1 | tee logs/phase3_olmo2_1b.log

# 2.5B
python -u src/p03_same_arch_validation.py --mode sanity --models Gemma2B_SFT_DPO 2>&1 | tee logs/phase3_gemma2b.log

# 2.7B
python -u src/p03_same_arch_validation.py --mode sanity --models Phi2_SFT_DPO 2>&1 | tee logs/phase3_phi2.log

# 3B
python -u src/p03_same_arch_validation.py --mode sanity --models Qwen25_3B_Tulu 2>&1 | tee logs/phase3_qwen25_3b.log
```

### Batch B: 7B models (A40 40GB+)

```bash
python -u src/p03_same_arch_validation.py --mode sanity --models OLMo2_7B 2>&1 | tee logs/phase3_olmo2_7b.log

python -u src/p03_same_arch_validation.py --mode sanity --models OLMoE_Tulu 2>&1 | tee logs/phase3_olmoe.log

python -u src/p03_same_arch_validation.py --mode sanity --models OLMo3_7B 2>&1 | tee logs/phase3_olmo3_7b.log

python -u src/p03_same_arch_validation.py --mode sanity --models Zephyr_SFT_DPO 2>&1 | tee logs/phase3_zephyr.log

python -u src/p03_same_arch_validation.py --mode sanity --models Gemma7B_SFT_DPO 2>&1 | tee logs/phase3_gemma7b.log

python -u src/p03_same_arch_validation.py --mode sanity --models Qwen2_7B_DPOShift 2>&1 | tee logs/phase3_qwen2_7b.log
```

### Batch C: 8B model (A40 40GB+)

```bash
# Already validated (Δ=+21.8) — re-run only if steering vector re-extracted
python -u src/p03_same_arch_validation.py --mode sanity --models Llama31_Tulu 2>&1 | tee logs/phase3_llama31_tulu.log
```

### Full run (after sanity passes)

```bash
# All 11 at once — full mode
python -u src/p03_same_arch_validation.py --mode full --models OLMo2_1B Gemma2B_SFT_DPO Phi2_SFT_DPO Qwen25_3B_Tulu OLMo2_7B OLMoE_Tulu OLMo3_7B Zephyr_SFT_DPO Gemma7B_SFT_DPO Qwen2_7B_DPOShift Llama31_Tulu 2>&1 | tee logs/phase3_full_all11.log
```

---

## Success Criteria

| Metric | Pass | Fail |
|--------|------|------|
| AQI(λ=0) → AQI(λ=1) | Monotonic increase | Flat or decreasing |
| AQI Δ | > +5 points | < +5 points |
| No errors | Complete run | OOM / NaN / crash |
