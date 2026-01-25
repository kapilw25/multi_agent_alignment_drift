# MAHALS: Multi-Agent Homophilic Alignment via Latent Steering

> **Goal**: Achieve alignment homophily across multi-architecture LLMs through latent steering
> **Narrative**: See [goal_polished.md](../goal_polished.md) for threat model and high-level framing
> **Status**: Phase 1 Complete, Phase 2 Ready
> **Last Updated**: Jan 18, 2026

---

## Research Questions

1. How do we **measure** alignment drift across different LLM architectures?
2. Can steering vectors transfer cross-architecture?
3. What game-theoretic equilibrium exists for multi-agent alignment?

---

## Phase Structure

| Phase | Name | Effort | Purpose | Script |
|:-----:|:-----|:------:|:--------|:-------|
| **1** | **Baseline Selection** | EASY | Identify per-axiom weaknesses, select baseline | `p01_measure_baseline_aqi.py` |
| **2** | **Extract Steering Vectors** | MEDIUM | Get v_llama, v_mistral, etc. | `p02_extract_steering_vectors.py` |
| **3** | **Same-Architecture Validation** | MEDIUM | Prove steering works (AQI vs λ) | `p03_same_arch_validation.py` |
| **4** | **Cross-Architecture Steering** | HARD | Achieve homophily across architectures | `m05_*.py` (TBD) |
| **5** | **Alignment Recovery** (OPTIONAL) | HIGH | Recover alignment after intentional degradation | - |
| **6** | **Multi-Agent Interaction** | | | |
| 6.1 | Collaboration Test | HARD | Test steered agents maintain alignment | - |
| 6.2 | Game-Theoretic Equilibrium | RESEARCH | Nash equilibrium formulation | - |

---

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-AGENT ALIGNMENT DRIFT PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  PHASE 1                    PHASE 2                    PHASE 3                 PHASE 4
  Baseline AQI               Steering Vectors           Same-Arch Validation    Cross-Arch Steering
  ─────────────              ────────────────           ──────────────          ───────────────

  ┌─────────┐               ┌─────────┐ ┌─────────┐
  │  LLM_1  │──┐            │  Base   │ │Instruct │
  │ (AQI_1) │  │            │ Model   │ │ Model   │
  └─────────┘  │            └────┬────┘ └────┬────┘
               │                 │           │
  ┌─────────┐  │                 └─────┬─────┘
  │  LLM_2  │  │                       ▼
  │ (AQI_2) │──┤  AQI        ┌─────────────────┐        ┌─────────────────┐
  └─────────┘  │  Eval──────►│ Steering Vector │───────►│ Apply to Base   │
               │             │ v = h_i - h_b   │        │ at λ = 0→1      │
  ┌─────────┐  │             └─────────────────┘        └────────┬────────┘
  │  LLM_3  │  │                                                  │
  │ (AQI_3) │──┤             Per architecture:                    ▼
  └─────────┘  │             v_1, v_2, ...v_N          ┌─────────────────┐
               │                                        │ AQI vs λ curve  │
  ┌─────────┐  │                                        │ (monotonic?)    │
  │   ...   │  │                                        └────────┬────────┘
  │         │──┤                                                  │
  └─────────┘  │                                                  ▼
               │                                        ┌─────────────────┐
  ┌─────────┐  │                                        │ Cross-Arch:     │
  │  LLM_N  │──┘                                        │ v_best → all    │
  │ (AQI_N) │                                           │ other LLMs      │
  └─────────┘                                           └─────────────────┘

       │                           │                          │
       ▼                           ▼                          ▼
  ┌─────────┐               ┌─────────────┐           ┌─────────────┐
  │ Baseline│               │ N Steering  │           │ Transfer    │
  │ = Best  │               │ Vectors     │           │ Matrix      │
  │  AQI    │               │ (.pt files) │           │ (NxN)       │
  └─────────┘               └─────────────┘           └─────────────┘
```

---

## Two Types of Alignment Drift

### Type 1: Baseline Differential (Natural)
> **Source**: Different architectures trained on different RLHF preferences
> **When**: Exists BEFORE collaboration

Different training → Different alignment baselines → Measurable via AQI

### Type 2: Collaboration-Induced Drift
> **Source**: Multi-agent interaction degrades alignment
> **When**: Happens DURING collaboration

Agent A (AQI=75) + Agent B (AQI=68) → After collaboration → Both degraded

---

## Phase Details

### Phase 1: Baseline Selection [COMPLETE]

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: BASELINE AQI MEASUREMENT                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐         ┌─────────┐             │
│   │  LLM_1  │   │  LLM_2  │   │  LLM_3  │   ...   │  LLM_N  │             │
│   │Instruct │   │Instruct │   │Instruct │         │Instruct │             │
│   └────┬────┘   └────┬────┘   └────┬────┘         └────┬────┘             │
│        │             │             │                    │                  │
│        └─────────────┴──────┬──────┴────────────────────┘                  │
│                             ▼                                              │
│                 ┌───────────────────────┐                                  │
│                 │   LITMUS Dataset      │                                  │
│                 │   (hasnat79/litmus)   │                                  │
│                 │   200 per category    │                                  │
│                 └───────────┬───────────┘                                  │
│                             ▼                                              │
│                 ┌───────────────────────┐                                  │
│                 │   AQI Evaluation      │                                  │
│                 │   CHI + XB → AQI      │                                  │
│                 └───────────┬───────────┘                                  │
│                             ▼                                              │
│        ┌─────────┬─────────┬─────────┬─────────┐                          │
│        ▼         ▼         ▼         ▼         ▼                          │
│     AQI_1     AQI_2     AQI_3     ...      AQI_N                          │
│                             │                                              │
│                             ▼                                              │
│                    Baseline = max(AQI)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Script**: `src/p01_measure_baseline_aqi.py`

**Configuration** (`src/m01_config.py`):
- Dataset: `hasnat79/litmus`
- Samples: 100 per category (sanity) / 200 per category (full)
- Models: Llama3_8B, Mistral_7B, Qwen2_7B, Gemma2_9B, Falcon_7B, Zephyr_7B

**Commands**:
```bash
python -u src/p01_measure_baseline_aqi.py --mode sanity 2>&1 | tee logs/phase1_sanity.log
python -u src/p01_measure_baseline_aqi.py --mode full 2>&1 | tee logs/phase1_full.log
```

**Output**: `outputs/phase1_baseline_aqi/`
- Per-model AQI scores
- Baseline model identification (highest AQI)
- Bar plots comparing all models

#### Phase 1 Results Summary

| Model | AQI | Status |
|-------|-----|--------|
| **Mistral_7B** | 55.0 | **Baseline** |
| Zephyr_7B | 55.0 | Tied |
| Llama3_8B | 34.8 | Needs steering |
| Gemma2_9B | 34.3 | Needs steering |
| Qwen2_7B | 15.0 | Needs steering |
| Falcon_7B | 5.0 | Needs steering |

> **Threat Model**: Per-axiom vulnerabilities identified in `aqi_axiom_heatmap.png`.
> See [expectation.md](./expectation.md#per-axiom-vulnerability-analysis-threat-model) for detailed analysis.

---

### Phase 2: Extract Steering Vectors [COMPLETE]

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: STEERING VECTOR EXTRACTION                      │
│                         (D-STEER Approach - Batched)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For EACH architecture (LLM_1, LLM_2, ... LLM_N):                          │
│                                                                             │
│   ┌──────────────────┐              ┌──────────────────┐                    │
│   │   Base Model     │              │  Instruct Model  │                    │
│   │   (LLM_i-base)   │              │  (LLM_i-instruct)│                    │
│   └────────┬─────────┘              └────────┬─────────┘                    │
│            │                                 │                              │
│            │    ┌───────────────────────┐    │                              │
│            │    │  Anthropic/hh-rlhf    │    │                              │
│            │    │  (chosen + rejected)  │    │                              │
│            │    │  1000 samples (full)  │    │                              │
│            │    └───────────┬───────────┘    │                              │
│            │                │                │                              │
│            ▼                ▼                ▼                              │
│   ┌─────────────────────────────────────────────────────┐                  │
│   │         Batched Hidden State Extraction              │                  │
│   │         (batch_size from MODEL_BATCH_SIZES)          │                  │
│   │                                                      │                  │
│   │   [1/4] BASE - chosen      [2/4] BASE - rejected    │                  │
│   │   [3/4] INSTRUCT - chosen  [4/4] INSTRUCT - rejected│                  │
│   └─────────────────────────┬───────────────────────────┘                  │
│                             ▼                                              │
│                 ┌─────────────────────┐                                    │
│                 │   Steering Vector   │                                    │
│                 │ v = mean(h_inst -   │                                    │
│                 │         h_base)     │                                    │
│                 └─────────┬───────────┘                                    │
│                           │                                                │
│                           ▼                                                │
│                 ┌─────────────────────┐                                    │
│                 │  SVD Decomposition  │  (optional, --no-svd to skip)      │
│                 │  component 3        │                                    │
│                 └─────────┬───────────┘                                    │
│                           │                                                │
│                           ▼                                                │
│   ┌───────────────────────────────────────────────────────────┐           │
│   │  steering_vector.pth          (chosen, per layer)         │           │
│   │  steering_vector_rejected.pth (rejected, per layer)       │           │
│   │  steering_vector_svd3.pth     (SVD component 3)           │           │
│   │  cosine_similarity.png        (instruct vs base)          │           │
│   │  steering_vector_norms.png    (magnitude per layer)       │           │
│   └───────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Script**: `src/p02_extract_steering_vectors.py`

**Approach**: D-STEER (no training required)
```
v = mean(h_instruct - h_base)   # per layer, hidden_dim
```

**Dataset**: `Anthropic/hh-rlhf` (chosen + rejected pairs)
- Sanity mode: 100 samples
- Full mode: 1000 samples

**Batch Processing** (synced with m02, A100 80GB optimized):
```python
MODEL_BATCH_SIZES = {
    "Llama3_8B": 16,   "Mistral_7B": 16,  "Qwen2_7B": 16,
    "Gemma2_9B": 8,    "Falcon_7B": 16,   "Zephyr_7B": 16,
}
```

> **⚠️ POC Note**: Current implementation uses **Base→Instruct** pairs (general alignment direction).
> For pure D-STEER replication, future work should use **SFT→DPO** pairs:
> - **Option A**: Find public SFT→DPO model pairs for each architecture
> - **Option B**: Train SFT→DPO pipelines (see `finetuning_evaluation/comparative_study/`)

**Model Pairs** (`src/utils/model_registry.json`):
| Key | Base Model | Instruct Model |
|-----|-----------|----------------|
| Mistral_7B | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 |
| Llama3_8B | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct |
| Qwen2_7B | Qwen/Qwen2-7B | Qwen/Qwen2-7B-Instruct |
| Gemma2_9B | google/gemma-2-9b | google/gemma-2-9b-it |
| Falcon_7B | tiiuae/falcon-7b | tiiuae/falcon-7b-instruct |
| Zephyr_7B | mistralai/Mistral-7B-v0.1 | HuggingFaceH4/zephyr-7b-beta |

**Commands**:
```bash
python -u src/p02_extract_steering_vectors.py --mode sanity 2>&1 | tee logs/phase2_sanity.log
python -u src/p02_extract_steering_vectors.py --mode full 2>&1 | tee logs/phase2_full.log
python -u src/p02_extract_steering_vectors.py --no-svd --models Mistral_7B  # skip SVD
```

**Output**: `outputs/phase2_steering_vectors/`
```
{model}/
├── steering_vector.pth           # Chosen responses (num_layers × hidden_dim)
├── steering_vector_rejected.pth  # Rejected responses
├── steering_vector_svd3.pth      # SVD component 3 (if enabled)
├── cosine_similarity.png         # Instruct vs Base similarity
├── steering_vector_norms.png     # L2 norm per layer
├── metadata.json                 # Stats, batch_size, timestamps
└── explained_variances/          # SVD variance plots per layer
```

**Summary Plots** (cross-model):
- `cosine_similarity_comparison.png` - All models comparison
- `steering_norms_comparison.png` - Norm curves overlay
- `model_dimensions.png` - Architecture comparison

#### Phase 2 Results Summary

| Model | Layers | Dim | CosSim | Steering Potential |
|-------|--------|-----|--------|-------------------|
| **Zephyr_7B** | 32 | 4096 | 0.77 | **Highest** |
| **Falcon_7B** | 32 | 4544 | 0.78 | **High** |
| Mistral_7B | 32 | 4096 | 0.83 | Good (baseline) |
| Gemma2_9B | 42 | 3584 | 0.92 | Moderate |
| Llama3_8B | 32 | 4096 | 0.95 | Low |
| Qwen2_7B | 28 | 3584 | 0.99 | **Minimal** |

**Key Finding**: Lower CosSim = larger `h_instruct - h_base` gap = more extractable alignment signal.

> See [expectation.md](./expectation.md#plot-analysis) for detailed plot analysis.

---

### Phase 3: Same-Architecture Validation [IN PROGRESS]

**Goal**: Verify steering works within same architecture

**Script**: `src/p03_same_arch_validation.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAME-ARCHITECTURE STEERING TEST                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LLM_i (Base)                                                              │
│        │                                                                    │
│        ▼                                                                    │
│   Apply v_i with different λ values                                         │
│        │                                                                    │
│        ├── λ = 0.0  →  AQI = ?? (should match base)                        │
│        ├── λ = 0.5  →  AQI = ?? (should be between)                        │
│        └── λ = 1.0  →  AQI = ?? (should match instruct)                    │
│                                                                             │
│   Validation: Does AQI increase monotonically with λ?                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Details**:

1. **Load Base Model**: Load `{model_key}` base model (e.g., `mistralai/Mistral-7B-v0.3`)

2. **Load Steering Vector**:
   ```python
   v = torch.load(f"outputs/phase2_steering_vectors/{model_key}/steering_vector.pth")
   # Shape: [num_layers, hidden_dim] e.g., [32, 4096] for Mistral
   ```

3. **Steering Formula** (hook into forward pass):
   ```python
   # For each layer l:
   h_steered[l] = h_base[l] + λ * v[l]
   ```

4. **λ Values**: `[0.0, 0.25, 0.5, 0.75, 1.0]`
   - More granular than [0, 0.5, 1] to catch non-monotonic behavior

5. **Evaluation**: For each λ, measure AQI on LITMUS dataset (same as Phase 1)

6. **Output**:
   ```
   outputs/phase3_same_arch_validation/
   ├── {model_key}/
   │   ├── aqi_vs_lambda.png          # AQI vs λ curve
   │   ├── aqi_vs_lambda.json         # Raw data
   │   └── per_axiom_curves.png       # Per-axiom breakdown
   └── summary.json                   # All models comparison
   ```

**Expected**: Monotonic AQI increase with λ

**Recommended Models for Phase 3** (combining Phase 1 AQI + Phase 2 CosSim):

| Priority | Model | CosSim | AQI | Why |
|----------|-------|--------|-----|-----|
| 1 | **Zephyr_7B** | 0.77 | 55.0 | Highest steering potential + tied baseline |
| 2 | **Falcon_7B** | 0.78 | 5.0 | High potential + most room to improve |
| 3 | **Mistral_7B** | 0.83 | 55.0 | Baseline - validate steering doesn't degrade |

**Rationale**:
- Low CosSim = large alignment signal extractable
- Falcon has lowest AQI (5.0) so steering improvement will be most visible
- Qwen2 (CosSim=0.99) unlikely to respond to steering

---

### Phase 4: Cross-Architecture Steering [TODO]

**Goal**: Test steering vector transfer between architectures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-ARCHITECTURE STEERING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Baseline Model: LLM_best (highest AQI)                                    │
│                                                                             │
│   Extract v_best from LLM_best                                              │
│        │                                                                    │
│        ├────► Apply to LLM_1 (Base)     →  AQI = ??                        │
│        ├────► Apply to LLM_2 (Base)     →  AQI = ??                        │
│        ├────► Apply to LLM_3 (Base)     →  AQI = ??                        │
│        └────► Apply to LLM_N (Base)     →  AQI = ??                        │
│                                                                             │
│   Key Questions:                                                            │
│   • Does cross-architecture steering work at all?                          │
│   • How much alignment improvement vs same-architecture?                   │
│   • Which architectures are most compatible?                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Questions**:
- Does cross-architecture steering work?
- Which architectures are most compatible?
- How does it compare to same-architecture steering?

---

### Phase 5: Alignment Recovery [OPTIONAL]

**Goal**: Prove steering can recover alignment from intentionally degraded model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALIGNMENT RECOVERY EXPERIMENT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │   LLM_i      │────►│ Train on     │────►│   LLM_i      │               │
│   │   Instruct   │     │ rejected     │     │  Dealigned   │               │
│   │  (AQI = X)   │     │ responses    │     │  (AQI = Y)   │               │
│   └──────────────┘     └──────────────┘     └──────────────┘               │
│                                                    │                        │
│                                                    ▼                        │
│                                        ┌──────────────────────┐            │
│                                        │  Apply v_best        │            │
│                                        │  (cross-arch) OR     │            │
│                                        │  Apply v_i           │            │
│                                        │  (same-arch)         │            │
│                                        └──────────────────────┘            │
│                                                    │                        │
│                                                    ▼                        │
│                                             ┌──────────────┐               │
│                                             │   LLM_i      │               │
│                                             │  Recovered?  │               │
│                                             │  (AQI = ??)  │               │
│                                             └──────────────┘               │
│                                                                             │
│   Key Questions:                                                            │
│   • Can steering recover alignment after intentional degradation?          │
│   • Same-arch vs cross-arch recovery effectiveness?                        │
│   • What's the max AQI drop that can be recovered?                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Experiment**:
1. Train LoRA on rejected responses (dealign)
2. Apply steering vector
3. Measure recovery rate

---

### Phase 6: Multi-Agent Interaction [TODO]

#### Phase 6.1: Collaboration Test

**Goal**: Test if steered agents maintain alignment during collaboration

**Experiment**:
- Design multi-agent task
- Measure AQI before/after collaboration
- Test if pre-steering prevents drift

---

#### Phase 6.2: Game-Theoretic Equilibrium [RESEARCH]

**Goal**: Find Nash equilibrium for multi-agent alignment states

---

## Code Assets

| Asset | Location | Purpose |
|-------|----------|---------|
| Config | `src/m01_config.py` | Phase 1 settings |
| AQI Evaluation | `src/p01_measure_baseline_aqi.py` | Measure alignment quality |
| Steering Extraction | `src/p02_extract_steering_vectors.py` | Extract steering vectors |
| Model Registry | `src/utils/model_registry.json` | Model pairs config |
| AQI Package | `src/AQI/` | Shared AQI utilities |

---

## References

- [D-STEER Paper](https://arxiv.org/abs/2512.11838) - "DPO as Steering Vector Perturbation" (Dec 2025)
- [Agent Drift: Quantifying Behavioral Degradation](https://arxiv.org/abs/2601.04170) - Jan 2026
- [AAAI 2026 Bridge Program on LLM-Based Multi-Agent Collaboration](https://multiagents.org/2026/)
