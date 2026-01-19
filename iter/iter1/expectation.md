# MAHALS: Expected Outputs

> **Project**: Multi-Agent Homophilic Alignment via Latent Steering
> **Execution**: GPU Server Only (Nvidia CUDA)
> **GPU**: A100 80GB (optimized batch sizes per model)

---

## Phase 1: Baseline Selection

> **Scripts**: `src/m01_config.py`, `src/m02_measure_baseline_aqi.py`

### m01_config.py

**Command**: `python src/m01_config.py`

**Expected Output**:
```
==================================================
Phase 1: Multi-Architecture AQI Config
==================================================
Models: ['Llama3_8B', 'Mistral_7B', 'Qwen2_7B', 'Gemma2_9B', 'Falcon_7B', 'Zephyr_7B']
Dataset: hasnat79/litmus
Output: /path/to/outputs/phase1_baseline_aqi
```

---

### m02_measure_baseline_aqi.py

#### Sanity Mode (Quick Test)

**Command**: `python -u src/m02_measure_baseline_aqi.py --mode sanity 2>&1 | tee logs/phase1_sanity.log`

**Expected Runtime**: ~30-60 minutes (100 samples per category)

**Expected Console Output**:
```
GPU: NVIDIA A100-SXM4-80GB
Memory: 80.0 GB

============================================================
Phase 1: Multi-Architecture AQI Baseline (GPU Only)
============================================================
Mode: sanity | Samples: 100
Models: ['Llama3_8B', 'Mistral_7B', 'Qwen2_7B', 'Gemma2_9B', 'Falcon_7B', 'Zephyr_7B']
Output: outputs/phase1_baseline_aqi

Loading dataset: hasnat79/litmus
Loaded 700 samples

============================================================
Evaluating: Llama-3.1-8B
HF Repo: meta-llama/Llama-3.1-8B-Instruct
============================================================
Using batch_size=16 for Llama3_8B
...
AQI: 72.45

[...repeat for all 6 models...]

============================================================
PHASE 1 RESULTS: Baseline AQI Scores
============================================================
Model           AQI        CHI_norm     XB_norm
------------------------------------------------------------
Falcon_7B       45.23      0.52         0.38
Mistral_7B      35.12      0.42         0.28
...
------------------------------------------------------------
BASELINE MODEL: Falcon_7B (AQI=45.23)
============================================================

Summary: outputs/phase1_baseline_aqi/phase1_summary.json

============================================================
Generating AQI Plots
============================================================
  aqi_bar_plot: outputs/phase1_baseline_aqi/aqi_bar_plot.png
  aqi_axiom_heatmap: outputs/phase1_baseline_aqi/aqi_axiom_heatmap.png
```

#### Full Mode

**Command**: `python -u src/m02_measure_baseline_aqi.py --mode full 2>&1 | tee logs/phase1_full.log`

**Expected Runtime**: ~2-3 hours (200 samples per category)

---

### Phase 1 File Outputs

```
outputs/phase1_baseline_aqi/
├── phase1_summary.json              # Overall summary with baseline model
├── phase1_aqi_checkpoint.json       # Checkpoint for resume
├── aqi_bar_plot.png                 # Bar plot comparing all models
├── aqi_axiom_heatmap.png            # Heatmap of AQI by axiom × model
├── Llama3_8B/
│   ├── result.json                  # AQI, CHI_norm, XB_norm
│   ├── embeddings_Llama3_8B.pkl     # Cached embeddings
│   └── metrics_summary.json         # Per-axiom breakdown
├── Mistral_7B/
│   └── ...
├── Qwen2_7B/
│   └── ...
├── Gemma2_9B/
│   └── ...
├── Falcon_7B/
│   └── ...
└── Zephyr_7B/
    └── ...
```

---

## Phase 2.1: Extract Steering Vectors

> **Script**: `src/m03_extract_steering_vectors.py`

### Sanity Mode

**Command**: `python -u src/m03_extract_steering_vectors.py --mode sanity 2>&1 | tee logs/phase2_sanity.log`

**Expected Runtime**: ~1-2 hours (100 samples, 6 models × 2 model pairs each)

**Expected Console Output**:
```
GPU: NVIDIA A100-SXM4-80GB
Memory: 80.0 GB

============================================================
Phase 2: Extract Steering Vectors (D_STEER)
============================================================
Mode: sanity | Samples: 100
Dataset: Anthropic/hh-rlhf
SVD: Enabled (component 3)
Cache: Enabled
Models: ['Llama3_8B', 'Mistral_7B', 'Qwen2_7B', 'Gemma2_9B', 'Falcon_7B', 'Zephyr_7B']
Output: outputs/phase2_steering_vectors

Loading Anthropic/hh-rlhf dataset (100 samples)...
Extracted 100 valid samples

============================================================
Extracting: Mistral-7B-v0.3
  Base:     mistralai/Mistral-7B-v0.3
  Instruct: mistralai/Mistral-7B-Instruct-v0.3
============================================================

[1/3] Loading BASE model...
  Loading: mistralai/Mistral-7B-v0.3
  Using Flash Attention 2
GPU Memory: 45.2GB free / 80.0GB total

[2/3] Loading INSTRUCT model...
  Loading: mistralai/Mistral-7B-Instruct-v0.3
  Using Flash Attention 2
GPU Memory: 17.8GB free / 80.0GB total

[3/3] Computing steering vectors (batch_size=16)...
Extracting hidden states from 100 samples (batch_size=16)...
  Prepared 100 chosen + 100 rejected texts
[1/4] BASE model - chosen responses...
[2/4] BASE model - rejected responses...
[3/4] INSTRUCT model - chosen responses...
[4/4] INSTRUCT model - rejected responses...

Caching hidden states...
Hidden states shape: torch.Size([100, 32, 4096])

Shape of h_delta (chosen): torch.Size([32, 4096])
Norm of h_delta (chosen): 12.3456

Applying SVD decomposition (component 3)...
SVD steering vector shape: torch.Size([32, 4096])

Steering vector saved: outputs/phase2_steering_vectors/Mistral_7B
Mean cosine similarity (chosen): 0.9234
Mean cosine similarity (rejected): 0.9187

[...repeat for all 6 models...]

======================================================================
PHASE 2 RESULTS: Steering Vectors Extracted (D_STEER)
======================================================================
Model           Layers   Dim      CosSim(C)    CosSim(R)
----------------------------------------------------------------------
Mistral_7B      32       4096     0.9234       0.9187
Llama3_8B       32       4096     0.9156       0.9098
...
----------------------------------------------------------------------
Output: outputs/phase2_steering_vectors
======================================================================

Summary: outputs/phase2_steering_vectors/phase2_summary.json

============================================================
Generating Steering Vector Plots
============================================================
  cosine_similarity_comparison: outputs/phase2_steering_vectors/cosine_similarity_comparison.png
  steering_norms_comparison: outputs/phase2_steering_vectors/steering_norms_comparison.png
```

### Full Mode

**Command**: `python -u src/m03_extract_steering_vectors.py --mode full 2>&1 | tee logs/phase2_full.log`

**Expected Runtime**: ~6-8 hours (1000 samples)

---

### Phase 2.1 File Outputs

```
outputs/phase2_steering_vectors/
├── phase2_summary.json                    # Overall summary
├── phase2_steering_checkpoint.json        # Checkpoint for resume
├── cosine_similarity_comparison.png       # All models comparison
├── steering_norms_comparison.png          # Norm curves overlay
├── model_dimensions.png                   # Architecture comparison
├── Mistral_7B/
│   ├── steering_vector.pth                # Chosen responses (32 × 4096)
│   ├── steering_vector_rejected.pth       # Rejected responses
│   ├── steering_vector_svd3.pth           # SVD component 3
│   ├── cosine_similarity.png              # Instruct vs Base similarity
│   ├── steering_vector_norms.png          # L2 norm per layer
│   ├── metadata.json                      # Stats, batch_size, timestamps
│   ├── hidden_states_cache/               # Cached tensors (if enabled)
│   │   ├── h_base_chosen.pt
│   │   ├── h_base_rejected.pt
│   │   ├── h_instruct_chosen.pt
│   │   └── h_instruct_rejected.pt
│   └── explained_variances/               # SVD variance plots
│       ├── layer_0_explained_variance.png
│       ├── layer_1_explained_variance.png
│       └── ...
├── Llama3_8B/
│   └── ...
├── Qwen2_7B/
│   └── ...
├── Gemma2_9B/
│   └── ...
├── Falcon_7B/
│   └── ...
└── Zephyr_7B/
    └── ...
```

---

## GPU Memory Usage (A100 80GB)

Models loaded **sequentially** with batch processing optimized for A100:

```
┌────────────────────────────────────────────────────────────────────┐
│                       A100 80GB VRAM                                │
├────────────────────────────────────────────────────────────────────┤
│ Model        │ Size (bf16) │ Batch Size │ Phase 1 Peak │ Phase 2 Peak │
├──────────────┼─────────────┼────────────┼──────────────┼──────────────┤
│ Llama3_8B    │ ~16 GB      │ 16         │ ~25 GB       │ ~45 GB*      │
│ Mistral_7B   │ ~14 GB      │ 16         │ ~23 GB       │ ~40 GB*      │
│ Qwen2_7B     │ ~14 GB      │ 16         │ ~23 GB       │ ~40 GB*      │
│ Gemma2_9B    │ ~18 GB      │ 8          │ ~28 GB       │ ~50 GB*      │
│ Falcon_7B    │ ~14 GB      │ 16         │ ~23 GB       │ ~40 GB*      │
│ Zephyr_7B    │ ~14 GB      │ 16         │ ~23 GB       │ ~40 GB*      │
└──────────────┴─────────────┴────────────┴──────────────┴──────────────┘

* Phase 2 loads BASE + INSTRUCT models simultaneously
```

---

## Checkpointing

Both Phase 1 and Phase 2.1 support **resumable runs**:

```
# If interrupted, re-run the same command:
python -u src/m02_measure_baseline_aqi.py --mode sanity 2>&1 | tee logs/phase1_sanity.log

# You'll see:
============================================================
Checkpoint found: phase1_aqi_checkpoint.json
Completed models: ['Llama3_8B', 'Mistral_7B']
Pending models: ['Qwen2_7B', 'Gemma2_9B', 'Falcon_7B', 'Zephyr_7B']
============================================================
[R] Resume from checkpoint
[F] Fresh start (delete checkpoint)
[S] Skip (use cached results)
Select option:
```

---

## Success Criteria

### Phase 1: Baseline Selection

| Criterion | Expected |
|-----------|----------|
| All 6 models evaluated | Yes (sequential) |
| AQI scores computed | Per-model and per-axiom |
| Baseline model identified | Highest overall AQI |
| Heatmap generated | Shows per-axiom vulnerabilities |
| Checkpointing works | Resume on crash |

### Phase 2.1: Extract Steering Vectors

| Criterion | Expected |
|-----------|----------|
| All 6 model pairs processed | Base + Instruct for each |
| Steering vectors saved | .pth files per model |
| SVD decomposition applied | Component 3 by default |
| Cosine similarity computed | Instruct vs Base |
| Hidden states cached | For faster re-runs |

---

## Model Pairs (from model_registry.json)

| Key | Base Model | Instruct Model |
|-----|-----------|----------------|
| Mistral_7B | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 |
| Llama3_8B | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct |
| Qwen2_7B | Qwen/Qwen2-7B | Qwen/Qwen2-7B-Instruct |
| Gemma2_9B | google/gemma-2-9b | google/gemma-2-9b-it |
| Falcon_7B | tiiuae/falcon-7b | tiiuae/falcon-7b-instruct |
| Zephyr_7B | mistralai/Mistral-7B-v0.1 | HuggingFaceH4/zephyr-7b-beta |

---

## Next Steps

| After Phase | Next Action |
|-------------|-------------|
| Phase 1 | Identify weakest LLMs per axiom from heatmap → Proceed to Phase 2.1 |
| Phase 2.1 | Use steering vectors in Phase 2.2 (Same-Architecture Validation) |
