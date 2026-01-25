# Plan B: Tulu SFT→DPO Quick Validation

> **Status**: ✅ COMPLETE - SUCCESS
> **Date**: Jan 25, 2026
> **Goal**: Validate D-STEER hypothesis using genuine SFT→DPO pair before full iter2 training

---

## Approach Analysis

| Option | Changes | Pros | Cons |
|--------|---------|------|------|
| **A: Add Tulu to existing registry** | 1 file | Fast, minimal code changes | Semantically confusing (base=SFT, instruct=DPO) |
| B: Extend registry schema | 4 files | Clean architecture | More work, premature if Tulu also fails |
| C: New registry file + flag | 4+ files | Most flexible | Overkill for POC |

**Decision**: Option A (Quick Validation)

**Rationale**: If Tulu (true SFT→DPO pair) also fails → problem is deeper than model pairs. If it works → validates hypothesis, then refactor.

---

## Change Made

Added to `src/utils/model_registry.json`:

```json
"Llama31_Tulu": {
  "base": "allenai/Llama-3.1-Tulu-3-8B-SFT",
  "instruct": "allenai/Llama-3.1-Tulu-3-8B-DPO",
  "display_name": "Llama-3.1-Tulu-8B",
  "hidden_dim": 4096,
  "num_layers": 32,
  "note": "D-STEER reference SFT→DPO pair (genuine alignment direction)"
}
```

---

## Execution Command

```bash
# Single command - m04 auto-generates steering vectors via m03 if missing
python -u src/p03_same_arch_validation.py --mode sanity --models Llama31_Tulu 2>&1 | tee logs/phase3_tulu.log
```

**Note**: m04 detects missing steering vectors and calls m03 internally (see `generate_missing_steering_vectors()` in m04).

---

## Expected Outcomes

| Scenario | AQI(λ=0) vs AQI(λ=1) | Interpretation | Next Step |
|----------|----------------------|----------------|-----------|
| **Success** | AQI increases monotonically | D-STEER validated, Base→Instruct was wrong paradigm | Expand to more SFT→DPO pairs (iter2) |
| **Failure** | AQI flat or decreases | Problem deeper than model pairs | Investigate AQI metric, dataset, or steering formula |

---

## Why Tulu?

1. **Exact D-STEER reference**: Paper used `allenai/Llama-3.1-Tulu-3-8B-SFT` → `allenai/Llama-3.1-Tulu-3-8B-DPO`
2. **Controlled difference**: SFT and DPO differ ONLY in DPO training (same base, same SFT)
3. **Open weights**: Both models publicly available on HuggingFace

---

## Key Difference from iter1

| Aspect | iter1 (Failed) | Plan B (Tulu) |
|--------|----------------|---------------|
| **Pair** | Base → Instruct | SFT → DPO |
| **What differs** | Everything (pretraining, SFT, RLHF, format) | Only DPO alignment |
| **Steering captures** | Mixed signal (instruction-following + alignment + noise) | Pure alignment direction |

---

## Actual Results (Jan 25, 2026)

### Llama31_Tulu (SFT→DPO) - ✅ SUCCESS

| λ | AQI Score |
|---|-----------|
| 0.00 | 58.0 |
| 0.25 | 68.0 |
| 0.50 | 71.5 |
| 0.75 | 72.1 |
| 1.00 | **79.8** |

**Result**: **Monotonic Increase** of +21.8 points

### Comparison with Base→Instruct Models

| Model | Pair Type | AQI(λ=0) | AQI(λ=1) | Δ | Monotonic |
|-------|-----------|----------|----------|---|-----------|
| **Llama31_Tulu** | SFT→DPO | 58.0 | 79.8 | **+21.8** | ✅ Yes |
| Falcon_7B | Base→Instruct | 29.5 | 15.3 | -14.2 | ❌ No |
| Mistral_7B | Base→Instruct | 69.3 | 57.9 | -11.4 | ❌ No |
| Zephyr_7B | Base→Instruct | 65.9 | 67.8 | +1.9 | ❌ No |

### Conclusion

**D-STEER hypothesis validated.** The problem was the model pairs, not the steering formula.

- SFT→DPO pairs capture **pure alignment direction**
- Base→Instruct pairs capture **mixed signals** (instruction-following + format + alignment + noise)

### Next Steps

1. Expand to more SFT→DPO pairs in iter2
2. Investigate cross-architecture steering with SFT→DPO vectors
3. Consider training custom SFT→DPO pairs for architectures without public pairs
