# Multi-Agent Alignment Drift - Research Plan

> **Goal**: Resolve alignment drift across multi-architecture LLMs (Llama, Phi, etc.) working in the same environment
> **Status**: Phase 1 - Baseline Measurement
> **Last Updated**: Jan 17, 2026

---

## Research Questions

1. How do we **measure** alignment drift across different LLM architectures?
2. Can steering vectors transfer cross-architecture?
3. What game-theoretic equilibrium exists for multi-agent alignment?

---                                                                                                                
Updated Phase Structure                                                                                            
┌───────┬────────────────────────────────┬──────────┬─────────────────────────────────────────────┐                
│ Phase │              Name              │  Effort  │                   Purpose                   │                
├───────┼────────────────────────────────┼──────────┼─────────────────────────────────────────────┤                
│ 1     │ Measure Baseline AQI           │ EASY     │ Establish natural differences               │                
├───────┼────────────────────────────────┼──────────┼─────────────────────────────────────────────┤                
│ 2     │ Extract Steering Vectors       │ MEDIUM   │ Get v_llama, v_mistral, etc.                │                
├───────┼────────────────────────────────┼──────────┼─────────────────────────────────────────────┤                
│ 3     │ Same-Architecture Validation   │ MEDIUM   │ Prove steering works                        │                
├───────┼────────────────────────────────┼──────────┼─────────────────────────────────────────────┤                
│ 4     │ Cross-Architecture Steering    │ HARD     │ Test transfer between architectures         │                
├───────┼────────────────────────────────┼──────────┼─────────────────────────────────────────────┤                
│ 4.5   │ Alignment Recovery (OPTIONAL)  │ HIGH     │ Prove steering can recover broken alignment │                
├───────┼────────────────────────────────┼──────────┼─────────────────────────────────────────────┤                
│ 5     │ Multi-Agent Collaboration Test │ HARD     │ Test collaboration-induced drift            │                
├───────┼────────────────────────────────┼──────────┼─────────────────────────────────────────────┤                
│ 6     │ Game-Theoretic Equilibrium     │ RESEARCH │ Nash equilibrium formulation                │                
└───────┴────────────────────────────────┴──────────┴─────────────────────────────────────────────┘                
--- 

## Two Types of Alignment Drift

### Type 1: Baseline Differential (Natural)

> **Source**: Different architectures trained on different RLHF preferences
> **When**: Exists BEFORE collaboration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NATURAL ALIGNMENT DIFFERENTIAL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Llama-3 (Meta)       Phi-3 (Microsoft)      Mistral (Mistral AI)         │
│   ┌─────────────┐      ┌─────────────┐        ┌─────────────┐              │
│   │ Trained on  │      │ Trained on  │        │ Trained on  │              │
│   │ Meta's RLHF │      │ MS's data   │        │ Different   │              │
│   │ preferences │      │ preferences │        │ preferences │              │
│   └──────┬──────┘      └──────┬──────┘        └──────┬──────┘              │
│          │                    │                      │                      │
│          ▼                    ▼                      ▼                      │
│      AQI = 72             AQI = 68               AQI = 75                   │
│                                                                             │
│   These ARE ALREADY DIFFERENT — this IS the "drift" (differential)         │
│   You don't need to CREATE it — it already exists naturally.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Type 2: Collaboration-Induced Drift

> **Source**: Multi-agent interaction degrades alignment
> **When**: Happens DURING collaboration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COLLABORATION-INDUCED DRIFT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BEFORE COLLABORATION              AFTER COLLABORATION                     │
│                                                                             │
│   ┌─────────┐  ┌─────────┐          ┌─────────┐  ┌─────────┐               │
│   │ Agent A │  │ Agent B │          │ Agent A │  │ Agent B │               │
│   │ AQI=75  │  │ AQI=68  │   ───►   │ AQI=70  │  │ AQI=65  │               │
│   └─────────┘  └─────────┘          └─────────┘  └─────────┘               │
│        │            │                    │            │                     │
│        └─────┬──────┘                    └─────┬──────┘                     │
│              ▼                                 ▼                            │
│         Collaborate                    Both degraded!                       │
│         on tasks                       Alignment leaked                     │
│                                                                             │
│   Question: Does interaction cause alignment to degrade?                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## D-STEER Approach (No Training Required)

> See: `literature/D_STEER/summary.md`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    D-STEER: Steering Vector Extraction                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For EACH architecture (Llama, Mistral, etc.):                             │
│                                                                             │
│   ┌──────────────────┐              ┌──────────────────┐                    │
│   │   Base Model     │              │  Instruct Model  │                    │
│   │  (Llama-3-8B)    │              │ (Llama-3-8B-Inst)│                    │
│   └────────┬─────────┘              └────────┬─────────┘                    │
│            │                                 │                              │
│            └─────────────┬───────────────────┘                              │
│                          ▼                                                  │
│                 ┌─────────────────┐                                         │
│                 │ Steering Vector │                                         │
│                 │ v_llama = Δh    │                                         │
│                 └─────────────────┘                                         │
│                                                                             │
│   Key Question: Can v_mistral steer Llama? (Cross-architecture transfer)   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases (Easy Wins First)

### Phase 1: Measure Baseline AQI (EASY WIN)

> **Effort**: Low | **Value**: High | **Dependencies**: None

**Goal**: Establish AQI baseline for each architecture using existing tools.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: Baseline Measurement                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │
│   │  Llama  │   │   Phi   │   │ Mistral │   │  Qwen   │   │  Gemma  │      │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘      │
│        │             │             │             │             │           │
│        ▼             ▼             ▼             ▼             ▼           │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │              SAME Evaluation Suite (AQI)                        │      │
│   │  • Use existing: finetuning_evaluation/05_evaluation/AQI/       │      │
│   │  • Dataset: hasnat79/litmus                                     │      │
│   │  • Metrics: CHI, XB → AQI score [0-100]                        │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│        │             │             │             │             │           │
│        ▼             ▼             ▼             ▼             ▼           │
│    AQI = ??      AQI = ??      AQI = ??      AQI = ??      AQI = ??       │
│                                    ↑                                        │
│                         HIGHEST = BASELINE MODEL                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Run `evaluation.py` on 5 architectures (Llama-3-8B, Phi-3, Mistral-7B, Qwen2-7B, Gemma-2-9B)
- [ ] Record AQI scores per model
- [ ] Identify baseline model (highest AQI)
- [ ] Document per-axiom breakdown (7 dimensions)

**Tools**: `finetuning_evaluation/comparative_study/05_evaluation/AQI/evaluation.py`

---

### Phase 2: Extract Steering Vectors (MEDIUM)

> **Effort**: Medium | **Value**: High | **Dependencies**: Phase 1

**Goal**: Extract steering vectors for each architecture using D-STEER approach.

**Tasks**:
- [ ] For each architecture, get Base + Instruct model pair
- [ ] Extract hidden states on `Anthropic/hh-rlhf` prompts
- [ ] Compute steering vector: `v = h_instruct - h_base`
- [ ] Save steering vectors per architecture

**Tools**: `literature/D_STEER/steering/Create_steering_vector_and_AQI_eval.ipynb`

---

### Phase 3: Same-Architecture Steering Validation (MEDIUM)

> **Effort**: Medium | **Value**: Medium | **Dependencies**: Phase 2

**Goal**: Verify steering works within same architecture before trying cross-architecture.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAME-ARCHITECTURE STEERING TEST                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Llama-3-8B (Base)                                                         │
│        │                                                                    │
│        ▼                                                                    │
│   Apply v_llama with different λ values                                     │
│        │                                                                    │
│        ├── λ = 0.0  →  AQI = ?? (should match base)                        │
│        ├── λ = 0.5  →  AQI = ?? (should be between)                        │
│        └── λ = 1.0  →  AQI = ?? (should match instruct)                    │
│                                                                             │
│   Validation: Does AQI increase monotonically with λ?                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Apply steering vectors at λ = {0.0, 0.25, 0.5, 0.75, 1.0}
- [ ] Measure AQI at each λ value
- [ ] Plot AQI vs λ curve for each architecture
- [ ] Verify monotonic increase (sanity check)

---

### Phase 4: Cross-Architecture Steering (HARD - Key Experiment)

> **Effort**: High | **Value**: Very High | **Dependencies**: Phase 3

**Goal**: Test if steering vectors transfer between architectures.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-ARCHITECTURE STEERING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Baseline Model: Mistral (AQI = 75, highest)                               │
│                                                                             │
│   Extract v_mistral from Mistral                                            │
│        │                                                                    │
│        ├────► Apply to Llama (Base)     →  AQI = ??                        │
│        ├────► Apply to Phi (Base)       →  AQI = ??                        │
│        ├────► Apply to Qwen (Base)      →  AQI = ??                        │
│        └────► Apply to Gemma (Base)     →  AQI = ??                        │
│                                                                             │
│   Key Questions:                                                            │
│   • Does cross-architecture steering work at all?                          │
│   • How much alignment improvement vs same-architecture?                   │
│   • Which architectures are most compatible?                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Apply baseline model's steering vector to all other architectures
- [ ] Measure AQI improvement (or degradation)
- [ ] Compare cross-arch vs same-arch steering effectiveness
- [ ] Identify architecture compatibility patterns

---

### Phase 4.5: Alignment Recovery Stress Test (OPTIONAL)

> **Effort**: High | **Value**: Validation | **Dependencies**: Phase 3

**Goal**: Prove steering can RECOVER alignment from an intentionally degraded model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALIGNMENT RECOVERY EXPERIMENT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │ Llama-3-8B   │────►│ Train on     │────►│ Llama-3-8B   │               │
│   │ Instruct     │     │ rejected     │     │ Dealigned    │               │
│   │ (AQI = 72)   │     │ responses    │     │ (AQI = 35)   │               │
│   └──────────────┘     └──────────────┘     └──────────────┘               │
│                                                    │                        │
│                                                    ▼                        │
│                                        ┌──────────────────────┐            │
│                                        │  Apply v_mistral     │            │
│                                        │  (cross-arch) OR     │            │
│                                        │  Apply v_llama       │            │
│                                        │  (same-arch)         │            │
│                                        └──────────────────────┘            │
│                                                    │                        │
│                                                    ▼                        │
│                                             ┌──────────────┐               │
│                                             │ Llama-3-8B   │               │
│                                             │ Recovered?   │               │
│                                             │ (AQI = ??)   │               │
│                                             └──────────────┘               │
│                                                                             │
│   Key Questions:                                                            │
│   • Can steering recover alignment after intentional degradation?          │
│   • Same-arch vs cross-arch recovery effectiveness?                        │
│   • What's the max AQI drop that can be recovered?                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why This Matters**: If steering can recover a *deliberately broken* model, it proves the approach is robust enough for real-world alignment drift scenarios.

**Tasks**:
- [ ] Train LoRA adapter on rejected responses (create dealigned model)
- [ ] Measure AQI drop (before/after dealignment)
- [ ] Apply same-architecture steering vector → measure recovery
- [ ] Apply cross-architecture steering vector → measure recovery
- [ ] Compare recovery rates

**Tools**: May need to create `dealign_training.py` using `Anthropic/hh-rlhf` rejected responses

---

### Phase 5: Multi-Agent Collaboration Test (HARD)

> **Effort**: High | **Value**: High | **Dependencies**: Phase 4

**Goal**: Test if steered agents maintain alignment during collaboration.

**Tasks**:
- [ ] Design multi-agent task requiring collaboration
- [ ] Measure AQI before and after collaboration
- [ ] Detect collaboration-induced drift (Type 2)
- [ ] Test if pre-steering prevents drift

---

### Phase 6: Game-Theoretic Equilibrium (RESEARCH)

> **Effort**: Very High | **Value**: Research Contribution | **Dependencies**: Phase 5

**Goal**: Find Nash equilibrium for multi-agent alignment states.

**Tasks**:
- [ ] Model alignment as game-theoretic payoff
- [ ] Formulate equilibrium conditions
- [ ] Validate with Purdue 7-dimension framework
- [ ] Publish findings

---

## Literature Review (TODO)

1. Game theoretic equilibrium in multi-agent systems
2. Multi-Agent Alignment benchmarks
3. Purdue paper - Anthropic dataset → 7 dimensions → conflict of direction analysis

---

## References

- [D-STEER Paper](https://arxiv.org/abs/2512.11838) - "DPO as Steering Vector Perturbation" (Dec 2025)
- [Agent Drift: Quantifying Behavioral Degradation](https://arxiv.org/abs/2601.04170) - Jan 2026
- [AAAI 2026 Bridge Program on LLM-Based Multi-Agent Collaboration](https://multiagents.org/2026/)
- [Anthropic Alignment Auditing Agents](https://alignment.anthropic.com/2025/automated-auditing/)

---

## Code Assets

| Asset | Location | Purpose |
|-------|----------|---------|
| AQI Evaluation | `finetuning_evaluation/05_evaluation/AQI/` | Measure alignment quality |
| D-STEER Steering | `literature/D_STEER/steering/` | Extract steering vectors |
| Eval Utilities | `finetuning_evaluation/05_evaluation/eval_utils/` | Shared tools |
