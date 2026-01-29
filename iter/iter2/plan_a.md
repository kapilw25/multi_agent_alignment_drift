# iter2: Custom SFT→DPO Training Approach

> **Purpose**: Expand model coverage by training our own SFT→DPO pairs
> **Comparison**: See [iter1](../iter1/plan.md) for Base→Instruct approach (POC)
> **Status**: Future work (after POC validates steering hypothesis)

---

## iter1 vs iter2 Comparison

| Aspect | iter1 (Base→Instruct) | iter2 (Custom SFT→DPO) |
|--------|----------------------|------------------------|
| **Formula** | `v = h_instruct - h_base` | `v = h_DPO - h_SFT` |
| **Constraint** | Requires existing pair | Requires open weights only |
| **Feasible Models** | 4-9 / 15 | 8-12 / 15 |
| **Training Required** | No | Yes (weeks of GPU) |
| **Matches D_STEER** | Approximation | Exact methodology |

---

## Models Unlocked by SFT→DPO Training

| Model | iter1 (Base→Instruct) | iter2 (SFT→DPO) | Change |
|-------|----------------------|-----------------|--------|
| Phi-4 | Only instruct exists | Train DPO on it | **Unlocked** |
| Gemma 3 2B | Verify availability | Now feasible | **Unlocked** |
| DeepSeek-R1 | Paradigm mismatch | R1→R1-DPO | **Unlocked** |
| OLMoE | Verify pair | Now feasible | **Unlocked** |
| Mamba | No pair exists | Can train pair | **Unlocked** |
| GPT/Claude | API-only | Still impossible | No change |

---

## POC Recommendation

**Use iter1 (Base→Instruct) for POC first:**

| Reason | Explanation |
|--------|-------------|
| **Speed** | No training required → validate hypothesis in days, not weeks |
| **Already built** | Phase 1 + 2.1 complete with 6 models |
| **Risk mitigation** | If steering doesn't work, SFT→DPO investment wasted |
| **Sufficient coverage** | 4 architectures already feasible |

---

## Recommended Path

```
NOW (POC)                              LATER (Paper)
─────────────────────────              ─────────────────────────
iter1: Base→Instruct                   iter2: SFT→DPO

• 6 models already done                • Train 2-3 key models
• Build Phase 2.2 (validation)         • Add to paper as ablation
• Prove steering works                 • Expand to more categories

Time: ~1 week                          Time: ~3-4 weeks
```

---

## iter2 Training Plan (Future)

When POC validates steering, train SFT→DPO for breadth-first coverage:

| Category | Model | HF Endpoints to Create |
|----------|-------|------------------------|
| **LLMs** | Llama 3.1 8B | `{username}/Llama-3.1-8B-SFT` + `{username}/Llama-3.1-8B-DPO` |
| **SLMs** | Llama 3.2 1B | `{username}/Llama-3.2-1B-SFT` + `{username}/Llama-3.2-1B-DPO` |
| **MoE** | Mixtral 8x7B | `{username}/Mixtral-8x7B-SFT` + `{username}/Mixtral-8x7B-DPO` |
| **SSMs** | Jamba | `{username}/Jamba-SFT` + `{username}/Jamba-DPO` |

**Training Recipe (Consistent Across All):**
```
SFT Dataset:  OpenAssistant or dolly-15k
DPO Dataset:  Anthropic/hh-rlhf or ultrafeedback
Hyperparams:  β=0.1, lr=5e-7, epochs=1
```

---

## Decision Gate

```
┌─────────────────────────────────────────────────────────────────┐
│  IF Phase 2.2 (iter1) shows AQI increases with λ:               │
│     → POC validated                                             │
│     → Proceed to iter2 (SFT→DPO training)                       │
│                                                                 │
│  IF Phase 2.2 (iter1) fails:                                    │
│     → Steering doesn't work with Base→Instruct                  │
│     → Rethink approach before investing in training             │
└─────────────────────────────────────────────────────────────────┘
```
