# Plan B: Tulu SFT→DPO Quick Validation

> **Status**: Ready to execute
> **Date**: Jan 24, 2026
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
