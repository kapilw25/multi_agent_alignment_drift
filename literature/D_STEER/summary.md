# D-STEER Summary

> **Key Insight**: D-STEER extracts steering vectors from existing model differences — **no training required**.

---

## How D-STEER Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    D-STEER: Steering Vector Extraction                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐              ┌──────────────────┐                    │
│   │   Base Model     │              │   DPO Model      │                    │
│   │   (Pre-RLHF)     │              │   (Post-RLHF)    │                    │
│   └────────┬─────────┘              └────────┬─────────┘                    │
│            │                                 │                              │
│            ▼                                 ▼                              │
│   ┌──────────────────┐              ┌──────────────────┐                    │
│   │  Hidden states   │              │  Hidden states   │                    │
│   │  h_base          │              │  h_dpo           │                    │
│   └────────┬─────────┘              └────────┬─────────┘                    │
│            │                                 │                              │
│            └─────────────┬───────────────────┘                              │
│                          ▼                                                  │
│                 ┌─────────────────┐                                         │
│                 │ Steering Vector │                                         │
│                 │ v* = h_dpo - h_base │                                     │
│                 └─────────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Code

```python
# 1. Extract hidden states for BOTH models on same input
h_base = get_all_hidden_states(base_model, tokenizer_base, conversation)
h_dpo = get_all_hidden_states(dpo_model, tokenizer_dpo, conversation)

# 2. Steering vector = difference (NO TRAINING!)
steering_vector = h_dpo - h_base

# 3. Save for later use
torch.save(steering_vector, "steering_vector.pth")
```

---

## Applying the Steering Vector

```python
# INCREASE alignment (push toward DPO behavior):
h_steered = h_base + λ * steering_vector    # λ > 0

# DECREASE alignment (reverse DPO / dealign):
h_steered = h_dpo - λ * steering_vector     # λ > 0

# Lambda controls steering strength:
#   λ = 0.0  →  no change
#   λ = 0.5  →  partial steering
#   λ = 1.0  →  full steering
```

---

## D-STEER vs Intentional Dealignment Training

| Aspect | D-STEER | Intentional Dealignment |
|--------|---------|-------------------------|
| **Method** | Extract vector from model differences | Train on rejected responses |
| **Training Required** | NO | YES |
| **Speed** | Fast (inference only) | Slow (finetuning) |
| **Reversible** | YES (change λ sign) | NO (need retrain) |
| **Output** | Steering vector (reusable) | Permanently dealigned model |
| **Control** | Continuous (any λ value) | Binary (aligned/dealigned) |

---

## Why D-STEER Matters for Multi-Agent Alignment

1. **No training needed** — extract steering vectors from existing Base/Instruct model pairs
2. **Controllable** — lambda lets you tune alignment level continuously
3. **Cross-architecture question** — Can a steering vector from Mistral work on Llama?
4. **Reversible** — Can always undo steering by negating lambda

---

## References

- **Paper**: "Preference Alignment Techniques Learn to Behave, not to Believe" (arXiv:2512.11838)
- **Dataset**: `Anthropic/hh-rlhf` (chosen vs rejected responses)
- **Key Formula**: `v* = h_dpo - h_base` (steering vector)
