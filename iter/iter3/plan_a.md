# iter3: Pre-Existing SFT→DPO Pairs from HuggingFace

> **Status**: Planning
> **Date**: Feb 6, 2026
> **Goal**: Skip training — use community SFT→DPO pairs to expand D-STEER validation across architectures and scales
> **Prerequisite**: iter2 validated that TRUE SFT→DPO pairs produce monotonic AQI (MAHALS Δ=+23.8)

---

## D-STEER Requirement

> **`v = mean(h_DPO - h_SFT)`** — the steering vector captures changes across ALL parameters.
> Both SFT and DPO must be **full fine-tuning** (not LoRA/PEFT/QLoRA).
> LoRA-only pairs (PKU Custom) failed in iter2 Phase 0 (Δ=-2.98).

---

## Architecture Diversity — Honest Assessment

| Base Architecture | Confirmed Pairs | Sizes Available | Arch Type |
|-------------------|----------------|-----------------|-----------|
| **Llama 2** (dense Transformer) | 3 + (14 v2.5 ablations) | 7B, 13B, 70B | Dense Transformer |
| **Llama 3/3.1** (dense Transformer) | 7 | 8B, 70B, 405B | Dense Transformer |
| **OLMo-2** (dense Transformer) | 4 | 1B, 7B, 13B, 32B | Dense Transformer |
| **OLMo-3/3.1** (dense Transformer) | 4 | 7B, 32B | Dense Transformer |
| **OLMoE** (MoE) | 1 | 1B/7B | MoE |
| **Gemma 1** (Google, non-AllenAI) | 2 | 2.5B, 7B | Dense Transformer |
| **Phi-2** (parallel attn+MLP, non-AllenAI) | 1 | 2.7B | Dense Transformer (parallel) |
| **Mistral** (Zephyr, non-AllenAI) | 1 | 7B | Dense Transformer |
| **Mixtral** (non-AllenAI) | 1 (risky) | 8x7B | MoE |
| **Qwen2/2.5** (non-AllenAI) | 3 | 0.5B, 3B, 7B | Dense Transformer |
| **SSM / Hybrid** | 0 | — | — |

> **21 AllenAI pairs + 8 non-AllenAI = 29 total.** 26/29 are sequential dense Transformer, 1 is parallel dense Transformer (Phi-2), 2 are MoE (OLMoE confirmed, Mixtral risky). Zero SSM/Hybrid.
>
> **What we DO get**: scale diversity (0.5B → 405B), recipe diversity (Tulu 2/2.5/3, Dolci, Think, UltraFeedback, OpenAssistant, deita+dpo-mix, DPO-Shift), model family diversity (Llama 2/3/3.1, OLMo-2/3, Gemma, Mistral, Phi-2, **Qwen2/2.5**). First **Qwen** model family — different tokenizer (151K vocab), GQA, RoPE theta=1M.
>
> **What we DON'T get**: architecture type diversity beyond Transformer + MoE. Gemma 2/3 not available (no base→SFT→DPO pipeline exists).

---

## All 21 Confirmed AllenAI SFT↔DPO Pairs (Grouped by Architecture)

### Group A: OLMo-3 / 3.1 — Dense Transformer (4 pairs)

| # | Registry Key | SFT | DPO | Size | Recipe | Variant |
|---|-------------|-----|-----|------|--------|---------|
| 1 | `OLMo3_7B` | `allenai/Olmo-3-7B-Instruct-SFT` | `allenai/Olmo-3-7B-Instruct-DPO` | 7B | Dolci | Instruct |
| 2 | `OLMo3_7B_Think` | `allenai/Olmo-3-7B-Think-SFT` | `allenai/Olmo-3-7B-Think-DPO` | 7B | Dolci | Think (reasoning) |
| 3 | `OLMo31_32B` | `allenai/Olmo-3.1-32B-Instruct-SFT` | `allenai/Olmo-3.1-32B-Instruct-DPO` | 32B | Dolci | Instruct |
| 4 | `OLMo3_32B_Think` | `allenai/Olmo-3-32B-Think-SFT` | `allenai/Olmo-3-32B-Think-DPO` | 32B | Dolci | Think (reasoning) |

> **Think variants**: Reasoning-focused SFT→DPO. Tests whether D-STEER steering vector captures reasoning alignment vs safety/helpfulness alignment.

### Group B: OLMo-2 — Dense Transformer (4 pairs)

| # | Registry Key | SFT | DPO | Size | Recipe |
|---|-------------|-----|-----|------|--------|
| 5 | `OLMo2_1B` | `allenai/OLMo-2-0425-1B-SFT` | `allenai/OLMo-2-0425-1B-DPO` | 1B | Tulu 3 |
| 6 | `OLMo2_7B` | `allenai/OLMo-2-1124-7B-SFT` | `allenai/OLMo-2-1124-7B-DPO` | 7B | Tulu 3 |
| 7 | `OLMo2_13B` | `allenai/OLMo-2-1124-13B-SFT` | `allenai/OLMo-2-1124-13B-DPO` | 13B | Tulu 3 |
| 8 | `OLMo2_32B` | `allenai/OLMo-2-0325-32B-SFT` | `allenai/OLMo-2-0325-32B-DPO` | 32B | Tulu 3 |

> **Scale ladder**: 1B → 7B → 13B → 32B on same architecture family + same recipe. Ideal for scaling law analysis of D-STEER.

### Group C: OLMoE — MoE (1 pair)

| # | Registry Key | SFT | DPO | Active/Total | Recipe |
|---|-------------|-----|-----|-------------|--------|
| 9 | `OLMoE_Tulu` | `allenai/OLMoE-1B-7B-0125-SFT` | `allenai/OLMoE-1B-7B-0125-DPO` | 1B/7B (64 experts, 8 active) | Tulu 3 |

> **Only confirmed MoE pair with full FT.** The one genuinely different architecture type.

### Group D: Llama 3.1 Tulu 3 — Dense Transformer (3 pairs)

| # | Registry Key | SFT | DPO | Size | Recipe |
|---|-------------|-----|-----|------|--------|
| 10 | `Llama31_Tulu` | `allenai/Llama-3.1-Tulu-3-8B-SFT` | `allenai/Llama-3.1-Tulu-3-8B-DPO` | 8B | Tulu 3 |
| 11 | `Llama31_Tulu_70B` | `allenai/Llama-3.1-Tulu-3-70B-SFT` | `allenai/Llama-3.1-Tulu-3-70B-DPO` | 70B | Tulu 3 |
| 12 | `Llama31_Tulu_405B` | `allenai/Llama-3.1-Tulu-3-405B-SFT` | `allenai/Llama-3.1-Tulu-3-405B-DPO` | 405B | Tulu 3 |

> **#10 already VALIDATED** (Δ=+21.8 monotonic). #11 and #12 are scale-up tests. 405B needs ~810 GB VRAM (BF16) — multi-node only.

### Group E: Llama 3/3.1 Tulu 2 — Dense Transformer (4 pairs)

| # | Registry Key | SFT | DPO | Size | Recipe |
|---|-------------|-----|-----|------|--------|
| 13 | `Llama31_Tulu2_8B` | `allenai/llama-3.1-tulu-2-8b` | `allenai/llama-3.1-tulu-2-dpo-8b` | 8B | Tulu 2 |
| 14 | `Llama31_Tulu2_70B` | `allenai/llama-3.1-tulu-2-70b` | `allenai/llama-3.1-tulu-2-dpo-70b` | 70B | Tulu 2 |
| 15 | `Llama3_Tulu2_8B` | `allenai/llama-3-tulu-2-8b` | `allenai/llama-3-tulu-2-dpo-8b` | 8B | Tulu 2 |
| 16 | `Llama3_Tulu2_70B` | `allenai/llama-3-tulu-2-70b` | `allenai/llama-3-tulu-2-dpo-70b` | 70B | Tulu 2 |

> **Recipe comparison**: Same Llama 3.1 8B base, Tulu 2 vs Tulu 3 recipe (#13 vs #10). Direct test of whether D-STEER is recipe-sensitive.

### Group F: Llama 2 Tulu 2 — Dense Transformer (3 pairs)

| # | Registry Key | SFT (no "-sft" suffix) | DPO | Size | Recipe |
|---|-------------|------------------------|-----|------|--------|
| 17 | `Llama2_Tulu2_7B` | `allenai/tulu-2-7b` | `allenai/tulu-2-dpo-7b` | 7B | Tulu 2 |
| 18 | `Llama2_Tulu2_13B` | `allenai/tulu-2-13b` | `allenai/tulu-2-dpo-13b` | 13B | Tulu 2 |
| 19 | `Llama2_Tulu2_70B` | `allenai/tulu-2-70b` | `allenai/tulu-2-dpo-70b` | 70B | Tulu 2 |

> **Oldest pair family.** Llama 2 architecture + Tulu 2 recipe. Tests D-STEER generalization to older model generations.

### Preview Versions (skip — use stable releases above)

| # | DPO | SFT | Size | Why Skip |
|---|-----|-----|------|----------|
| 20 | `OLMo-2-1124-13B-DPO-Preview` | `OLMo-2-1124-13B-SFT-Preview` | 13B | Superseded by #7 |
| 21 | `OLMo-2-1124-7B-DPO-Preview` | `OLMo-2-1124-7B-SFT-Preview` | 7B | Superseded by #6 |

---

## Non-AllenAI Pairs (Verified in Previous Session)

| # | Registry Key | SFT | DPO | Size | Org | D-STEER |
|---|-------------|-----|-----|------|-----|---------|
| 22 | `Zephyr_SFT_DPO` | `alignment-handbook/zephyr-7b-sft-full` | `HuggingFaceH4/zephyr-7b-beta` | 7B | HuggingFace | **YES** (Full FT both) |
| 23 | `Phi2_SFT_DPO` | `lxuechen/phi-2-sft` | `lxuechen/phi-2-dpo` | 2.7B | Stanford (lxuechen) | **YES** (Full FT both) |
| 24 | `Gemma7B_SFT_DPO` | `lewtun/gemma-7b-sft-full-deita-10k-v0` | `lewtun/gemma-7b-dpo-full-mix1-beta-0.05-epoch-3` | 7B | HuggingFace (lewtun) | **YES** (Full FT both) |
| 25 | `Gemma2B_SFT_DPO` | `Columbia-NLP/gemma-2b-zephyr-sft` | `Columbia-NLP/gemma-2b-zephyr-dpo` | 2.5B | Columbia NLP | **YES** (Full FT both) |
| 26 | `Mixtral_Hermes` | `NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT` | `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO` | 8x7B | NousResearch | **RISK** (DPO=QLoRA) |
| 27 | `Qwen2_7B_DPOShift` | `NoManDeRY/DPO-Shift-Qwen-2-7B-UltraChat200K-SFT` | `NoManDeRY/DPO-Shift-Qwen-2-7B-Ultrafeedback-fixed-1.0` | 7B | NoManDeRY (DPO-Shift paper) | **YES** (Full FT both, 8GPU, batch 128) |
| 28 | `Qwen25_3B_Tulu` | `phunguyen01/II-Tulu-3B-SFT` | `phunguyen01/Qwen-Tulu-3B-DPO` | 3B | phunguyen01 | **YES** (Full FT both, Tulu-3 data) |
| 29 | `Qwen25Math_7B_InfiAlign` | `InfiX-ai/InfiAlign-Qwen-7B-SFT` | `InfiX-ai/InfiAlign-Qwen-7B-DPO` | 7B | InfiX-ai | **YES** (Full FT both, math-biased base) |

---

## Bonus: Tulu v2.5 DPO Data Ablations (14 models, same SFT base)

All 14 are 13B (Llama 2), all share `allenai/tulu-2-13b` as SFT base (needs verification), each trained with different DPO dataset:

| # | DPO Model | DPO Dataset | D-STEER Value |
|---|-----------|-------------|---------------|
| 1 | `tulu-v2.5-dpo-13b-uf-mean` | UltraFeedback (mean) | Reference |
| 2 | `tulu-v2.5-dpo-13b-uf-overall` | UltraFeedback (overall) | DPO data ablation |
| 3 | `tulu-v2.5-dpo-13b-hh-rlhf` | Anthropic HH-RLHF | DPO data ablation |
| 4 | `tulu-v2.5-dpo-13b-hh-rlhf-60k` | HH-RLHF (60K subset) | DPO data ablation |
| 5 | `tulu-v2.5-dpo-13b-nectar` | Nectar | DPO data ablation |
| 6 | `tulu-v2.5-dpo-13b-nectar-60k` | Nectar (60K subset) | DPO data ablation |
| 7 | `tulu-v2.5-dpo-13b-stackexchange` | StackExchange | DPO data ablation |
| 8 | `tulu-v2.5-dpo-13b-stackexchange-60k` | StackExchange (60K) | DPO data ablation |
| 9 | `tulu-v2.5-dpo-13b-shp2` | SHP-2 | DPO data ablation |
| 10 | `tulu-v2.5-dpo-13b-helpsteer` | HelpSteer | DPO data ablation |
| 11 | `tulu-v2.5-dpo-13b-capybara` | Capybara | DPO data ablation |
| 12 | `tulu-v2.5-dpo-13b-argilla-orca-pairs` | Argilla Orca Pairs | DPO data ablation |
| 13 | `tulu-v2.5-dpo-13b-alpacafarm-gpt4-pref` | AlpacaFarm GPT-4 | DPO data ablation |
| 14 | `tulu-v2.5-dpo-13b-alpacafarm-human-pref` | AlpacaFarm Human | DPO data ablation |

> **Paper value**: Same SFT, 14 different DPO datasets → 14 different steering vectors. Tests whether D-STEER AQI varies by DPO preference source. If it does, this proves DPO data quality matters for steerability.
>
> **Needs verification**: Confirm `tulu-2-13b` is the SFT base for all 14. Also confirm Tulu 2 uses full FT (not LoRA).

---

## Unmatched DPO (No SFT Found)

| DPO | Issue |
|-----|-------|
| `allenai/open-instruct-llama2-sharegpt-dpo-7b` | No matching SFT checkpoint published |
| `allenai/Llama-3.1-Tulu-3-8B-DPO-RM-RB2` | Reward model (Text Classification), not text generation |

---

## Already Validated (iter1 + iter2)

| Registry Key | SFT | DPO | D-STEER Result |
|-------------|-----|-----|---------------|
| `Llama31_Tulu` | `allenai/Llama-3.1-Tulu-3-8B-SFT` | `allenai/Llama-3.1-Tulu-3-8B-DPO` | **Δ=+21.8 Monotonic** |
| `Llama31_MAHALS` | `anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS` | `anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS` | **Δ=+23.8 Monotonic** |

## Failed Pairs (iter2 Phase 0 — why LoRA fails)

| Registry Key | SFT Method | DPO Method | D-STEER Result | Lesson |
|-------------|-----------|-----------|---------------|--------|
| `Llama31_PKU_Custom` | LoRA (rank 16) | LoRA (rank 16) | **Δ=-2.98 FAILED** | LoRA captures too weak a signal for steering |

---

## Dropped Models

| Model | Why Dropped |
|-------|-------------|
| ~~`Qwen25_RLHFlow`~~ | Full FT but **math domain only** (15K math samples, rule-based reward). D-STEER AQI measures safety/helpfulness, not math. Wrong domain. |

## Non-Feasible Models (No SFT→DPO Pair Exists on HF)

| Model | What Exists | Why No Pair |
|-------|-------------|-------------|
| **Phi-3/4** | Only instruct | SFT+DPO baked into single release. 94 community Phi DPO models searched — only 1 viable pair found (`lxuechen/phi-2-sft` → `phi-2-dpo`, Phi-2 only). No Phi-3/4 base→SFT→DPO pipeline exists; all community Phi-3/4 DPOs start from already-instruct-tuned models. |
| **Gemma 2/3** | base + instruct | 703 community DPO models searched. No Gemma 2/3 base→SFT→DPO pipeline exists. Google doesn't release SFT checkpoints. Best community attempts (princeton-nlp, togethercomputer) start from already-instruct-tuned `gemma-2-9b-it`. Only Gemma 1 (7B, 2B) has viable pairs (lewtun, Columbia-NLP). |
| **DeepSeek-R1** | Distill models | RL paradigm, not SFT→DPO. 40+ community DPO models searched — all built on distillations (not base→SFT→DPO). Only 2/40 documented training method. Best candidate (STAR-1) has broken SFT link + distillation base. Math-domain models (imagination-research) same problem as Qwen25_RLHFlow. |
| **Qwen (most)** | 359 models searched | ~176 are Japanese LLM course (`dpo-qwen-cot-merged`, LoRA merged), ~35 JayHyeon sweeps (0.5B too small), 50+ empty model cards, 1 confirmed LoRA (dorukardahan), 5+ start from instruct (not base→SFT→DPO). Only 3 pairs qualify out of 359. |
| **Falcon-H1** | pre-DPO exists (Hybrid Transformer+Mamba) | Only 90M params — too tiny for D-STEER signal. 16 Falcon DPO models searched: 2 are tiiuae pre-DPO (90M, Full FT, **SSM/Hybrid arch**), 2 LoRA, 10 quantization copies, 2 course projects. No larger (0.6B+) Falcon-H1 has pre-DPO checkpoint. 0.6B uses GRPO (not DPO). |
| **DeepSeek-V3** | 671B final only | Too large, no SFT checkpoint |
| **Jamba** | Instruct only | SFT+DPO combined |
| **Mamba** | Base only | No instruct/SFT/DPO exists |
| **FalconMamba** | base + instruct | Instruct is SFT-only, no DPO |
| **Hymba** | base + instruct | SFT+DPO combined |

---

## Detailed Training Parameters (Verified via Model Cards, Configs, Logs)

### #1 Mixtral_Hermes (NousResearch) — RISK

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** | **QLoRA adapter** |
| Base Model | `mistralai/Mixtral-8x7B-v0.1` | SFT checkpoint |
| Dataset | `teknium/OpenHermes-2.5` (1M+ entries, GPT-4 generated) | Preference data |
| Framework | Axolotl | Axolotl |
| Precision | BF16 | BF16 |

> **Source**: [teknium on HF](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO/discussions/3): *"The sft phase was full finetune"*. DPO is QLoRA — [NousResearch tweet](https://x.com/NousResearch/status/1746988416779309143): *"qlora adapter for the DPO"*.

### #2 OLMoE_Tulu (AllenAI) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** | **Full fine-tuning** |
| Base Model | `allenai/OLMoE-1B-7B-0125` | SFT checkpoint |
| Dataset | 608K samples (Tulu 3 mix) | OLMoE preference mix |
| Infrastructure | 32 GPUs, DeepSpeed ZeRO-3 | 32 GPUs, DeepSpeed ZeRO-3 |

> **Source**: [SFT logs](https://github.com/allenai/OLMoE/blob/main/logs/olmoe-sft-logs.txt), [DPO logs](https://github.com/allenai/OLMoE/blob/main/logs/olmoe-dpo-logs.txt) — ZeRO-3 across 32 GPUs, no LoRA.

### #3 OLMo2_1B (AllenAI) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** | **Full fine-tuning** (`use_lora: false`) |
| Learning Rate | 1e-5 | 5e-7 |
| Epochs | 2 | 1 |
| Beta | — | 5 |

> **Source**: [OLMo-2 7B DPO config](https://github.com/allenai/open-instruct/blob/main/configs/train_configs/olmo2/olmo2_1124_7b_dpo.yaml) — explicit `use_lora: false`. Same pipeline for 1B.

### #4 OLMo3_7B (AllenAI) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** | **Full fine-tuning** |
| SFT Framework | **OLMo-Core** (no LoRA support) | open-instruct |
| Dataset | Dolci-Instruct (math, code, chat) | Dolci-DPO |

> **Source**: [OLMo-Core](https://github.com/allenai/OLMo-core) — no LoRA/PEFT support in framework.

### #5 Zephyr_SFT_DPO (HuggingFace) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** ("sft-full") | **Full fine-tuning** (no LoRA in config) |
| Base Model | `mistralai/Mistral-7B-v0.1` | SFT checkpoint |
| Dataset | UltraChat 200K | UltraFeedback 64K |
| LR | 2e-5 | 5e-7 |
| Beta | — | 0.01 |

> **Source**: [SFT card](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full), [DPO config](https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_full.yaml).
>
> **Ablation value**: Current registry Zephyr uses base→DPO (Δ=+1.9, non-monotonic). New pair uses SFT→DPO on **same DPO model**. Direct test of whether alignment direction matters.

### #6 Phi2_SFT_DPO (Stanford / lxuechen) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** (no LoRA mentioned, LR 2e-5) | **Full fine-tuning** (explicitly stated) |
| Base Model | `microsoft/phi-2` (2.7B, parallel attn+MLP) | `lxuechen/phi-2-sft` |
| Dataset | OpenAssistant (single-turn instruction) | UltraFeedback 10K (binarized) |
| LR | 2e-5 (peak, cosine decay, 3% warmup) | 3e-5 (peak, cosine decay, 3% warmup) |
| Epochs | 2 | 2 |
| Batch Size | 64 | 64 |
| Context Length | 512 | 1024 |
| Beta | — | 0.1 |

> **Source**: [SFT card](https://huggingface.co/lxuechen/phi-2-sft), [DPO card](https://huggingface.co/lxuechen/phi-2-dpo). By Xuechen Li (Stanford). DPO card explicitly says "Full fine-tuning with DPO". SFT card: no LoRA/rank/alpha mentioned, LR 2e-5 is standard full FT.
>
> **Architecture value**: Phi-2 uses **parallel attention + MLP blocks** (like PaLM), different from sequential Llama/OLMo/Mistral. First non-standard-Transformer layout in our test set. hidden_dim=2560, num_layers=32.
>
> **94 Phi DPO models searched**: Only this pair qualifies. All others: LoRA (Yhyu13), empty model cards (SongTonyLi 13+ variants, yaswanth Phi-4), no SFT checkpoint (hydroxai), or task-specific (Datle1610 KQA).

---

### #7 Gemma7B_SFT_DPO (HuggingFace / lewtun) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** ("full" in name) | **Full fine-tuning** ("full" in name) |
| Base Model | `google/gemma-7b` | SFT checkpoint |
| Dataset | HuggingFaceH4/deita-10k-v0-sft (10K) | argilla/dpo-mix-7k (7.5K) |
| LR | 2e-5 (cosine, 10% warmup) | 5e-7 (cosine, 10% warmup) |
| Epochs | 3 | 3 |
| Batch Size | 128 (8 GPUs) | 128 (8 GPUs) |
| Beta | — | 0.05 |

> **Source**: [SFT card](https://huggingface.co/lewtun/gemma-7b-sft-full-deita-10k-v0), [DPO card](https://huggingface.co/lewtun/gemma-7b-dpo-full-mix1-beta-0.05-epoch-3). By Lewis Tunstall (co-author of alignment-handbook + Zephyr). Same framework.
>
> **Architecture**: Gemma 1 7B — hidden_dim=3072, num_layers=28. Different dims from Llama (4096/32). First Google model in test set.
>
> **703 Gemma DPO models searched**: Only 2 pairs qualify (this + Columbia-NLP 2B). All Gemma 2/3 DPOs start from instruct-tuned models. Google doesn't release SFT checkpoints.

---

### #8 Gemma2B_SFT_DPO (Columbia NLP) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** | **Full fine-tuning** |
| Base Model | `google/gemma-2b` | SFT checkpoint |
| Dataset | HuggingFaceH4/deita-10k-v0-sft (10K) | argilla/dpo-mix-7k (7.5K) |
| Hyperparams | Not disclosed ("carefully selected") | Not disclosed |

> **Source**: [SFT card](https://huggingface.co/Columbia-NLP/gemma-2b-zephyr-sft), [DPO card](https://huggingface.co/Columbia-NLP/gemma-2b-zephyr-dpo). By Columbia NLP research lab.
>
> **Architecture**: Gemma 1 2B — hidden_dim=2048, num_layers=18. Small model for fast D-STEER iteration.

---

### #9 Qwen2_7B_DPOShift (NoManDeRY) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** | **Full fine-tuning** |
| Base Model | `Qwen/Qwen2-7B` | SFT checkpoint |
| Dataset | UltraChat 200K | UltraFeedback Binarized |
| LR | 2e-5 | Not disclosed |
| Epochs | 1 | Not disclosed |
| Batch Size | 128 (8 GPUs, 4/GPU × 4 GA) | Not disclosed |
| Optimizer | Adam | Not disclosed |
| Scheduler | Cosine (10% warmup) | Not disclosed |
| Files | 4 safetensors, no adapter_config.json (~15 GB) | 4 safetensors, no adapter_config.json (~15 GB) |

> **Source**: [SFT card](https://huggingface.co/NoManDeRY/DPO-Shift-Qwen-2-7B-UltraChat200K-SFT), [DPO card](https://huggingface.co/NoManDeRY/DPO-Shift-Qwen-2-7B-Ultrafeedback-fixed-1.0). Paper: [DPO-Shift (arxiv:2502.07599)](https://arxiv.org/abs/2502.07599).
>
> **Architecture**: Qwen2ForCausalLM — hidden=3584, 28 layers, GQA (28 heads / 4 KV), vocab=151936. Different tokenizer + dims from Llama.
>
> **359 Qwen DPO models searched**: ~176 are Japanese LLM course exercise (`dpo-qwen-cot-merged`, LoRA merged), ~35 are JayHyeon hyperparameter sweeps (0.5B), 1 confirmed LoRA (dorukardahan), 50+ empty model cards. Only 3 pairs qualify.

---

### #10 Qwen25_3B_Tulu (phunguyen01) — CONFIRMED

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** | **Full fine-tuning** |
| Base Model | `Qwen/Qwen2.5-3B` | SFT checkpoint |
| Dataset | allenai/tulu-3-sft-mixture (939K) | Not disclosed (TRL DPO) |
| LR | 5e-6 | Not disclosed |
| Epochs | 2 | Not disclosed |
| Batch Size | 128 (8 GPUs) | Not disclosed |
| Framework | Axolotl 0.5.3 | TRL 0.12.1 |
| Files | 2 safetensors + training_config.yaml, no adapter_config.json (~2.87 GB) | 2 safetensors + training_config.yaml, no adapter_config.json (~2.88 GB) |

> **Source**: [SFT card](https://huggingface.co/phunguyen01/II-Tulu-3B-SFT), [DPO card](https://huggingface.co/phunguyen01/Qwen-Tulu-3B-DPO).
>
> **Architecture**: Qwen2ForCausalLM — hidden=2048, 36 layers, GQA (16 heads / 2 KV), vocab=151936.
>
> **D-STEER value**: Uses Tulu-3 SFT mixture — same data family as our validated Llama31_Tulu reference. Cross-architecture test of same recipe.

---

### #11 Qwen25Math_7B_InfiAlign (InfiX-ai) — CONFIRMED (math-biased base)

| Parameter | SFT | DPO |
|-----------|-----|-----|
| **Method** | **Full fine-tuning** (2-stage curriculum) | **Full fine-tuning** |
| Base Model | `Qwen/Qwen2.5-Math-7B` (math-specialized) | SFT checkpoint |
| Dataset | InfiR-SFT-165K (math/code/science/general) | 10K curated (3.5K math, 3.5K code, 3K science) |
| LR | Not disclosed | 5e-7 |
| Epochs | Not disclosed | 3 |
| Batch Size | Not disclosed | 16 |
| Beta | — | 0.1 |
| Files | 4 safetensors, no adapter_config.json (~15 GB) | 4 safetensors, no adapter_config.json (~15 GB) |

> **Source**: [SFT card](https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-SFT), [DPO card](https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-DPO).
>
> **Caveat**: Base model is `Qwen2.5-Math-7B` (math-specialized), not vanilla `Qwen2.5-7B`. SFT data is multi-domain (not math-only), but base model bias adds risk. D-STEER steering vector may capture math reasoning alignment more than safety/helpfulness.

---

### AllenAI Models Not Yet Individually Verified

> All AllenAI OLMo-2/3, Llama Tulu 2/3 models use the same open-instruct / OLMo-Core pipeline.
> Training configs consistently show `use_lora: false` or use DeepSpeed ZeRO-3 (unnecessary for LoRA).
> **High confidence all are full FT**, but individual model cards should be checked before Phase 3 runs.

---

## Explanation

### Why Full FT Matters for D-STEER

| Training Method | Params Modified | Steering Signal | D-STEER Result |
|----------------|----------------|-----------------|----------------|
| Full FT → Full FT | 100% of weights | Dense, rich | Δ=+21.8 to +23.8 (monotonic) |
| LoRA → LoRA | ~0.4% of weights | Sparse, weak | Δ=-2.98 (failed) |
| Full FT → QLoRA | 100% / ~0.4% | Mixed — unknown | ? (Mixtral will test) |

### Why Zephyr Upgrade Is a Paper Result

Current: `Mistral-7B-v0.1` (base) → `zephyr-7b-beta` (DPO) = Δ=+1.9, non-monotonic.
New: `zephyr-7b-sft-full` (SFT) → `zephyr-7b-beta` (DPO) = same DPO model, different steering vector.

If new pair produces monotonic AQI → proves **alignment direction (SFT→DPO) matters more than model choice**.

### Scale Analysis Opportunities

| Scale | Models Available | Pairs |
|-------|-----------------|-------|
| **0.5B** | JayHyeon Qwen (35+ variants, UltraFeedback) | bonus |
| **1B** | OLMo-2 1B | 1 |
| **2.5B** | Gemma 2B (Columbia-NLP) | 1 |
| **2.7B** | Phi-2 (parallel attn+MLP) | 1 |
| **3B** | Qwen25_3B_Tulu (Tulu-3 data) | 1 |
| **7B** | OLMo-2 7B, OLMo-3 7B, OLMo-3 7B Think, Llama2 Tulu2 7B, Zephyr, Gemma 7B, **Qwen2_7B_DPOShift**, **Qwen25Math_7B_InfiAlign** | 8 |
| **8B** | Llama31 Tulu3, Llama31 Tulu2, Llama3 Tulu2 | 3 |
| **13B** | OLMo-2 13B, Llama2 Tulu2 13B, (+ 14 v2.5 ablations) | 2 (+14) |
| **32B** | OLMo-2 32B, OLMo-3.1 32B, OLMo-3 32B Think | 3 |
| **70B** | Llama31 Tulu3 70B, Llama31 Tulu2 70B, Llama3 Tulu2 70B, Llama2 Tulu2 70B | 4 |
| **405B** | Llama31 Tulu3 405B | 1 |
| **MoE** | OLMoE 1B/7B, Mixtral 8x7B (risky) | 1+1 |

### Risk: Different Training Recipes

**Mitigation** — group results by confidence level:
- **High confidence** (Tulu 3 recipe, full FT): Llama31_Tulu, MAHALS, OLMoE_Tulu, OLMo2_*, Llama31_Tulu_70B/405B, Qwen25_3B_Tulu
- **Medium confidence** (full FT, different recipe): OLMo3_*, Zephyr_SFT_DPO, Tulu 2 models, Qwen2_7B_DPOShift (UltraChat/UltraFeedback)
- **Medium-low confidence** (full FT, math-biased base): Qwen25Math_7B_InfiAlign
- **Low confidence** (QLoRA DPO): Mixtral_Hermes

---

## Action Plan

### Phase 1: Priority models (small, fast, high-confidence)
1. Register in `model_registry.json`: OLMo2_1B, OLMo2_7B, OLMoE_Tulu, OLMo3_7B, Zephyr_SFT_DPO, Phi2_SFT_DPO, Gemma7B_SFT_DPO, Gemma2B_SFT_DPO, Qwen2_7B_DPOShift, Qwen25_3B_Tulu
2. Run Phase 3 on each
3. Zephyr ablation: Zephyr_7B (base→DPO) vs Zephyr_SFT_DPO (SFT→DPO)
4. Qwen25Math_7B_InfiAlign — run if Phase 1 succeeds on other Qwen pairs (math-biased base = risk)

### Phase 2: Scale-up (if Phase 1 validates)
5. Register: OLMo2_13B, OLMo2_32B, OLMo31_32B, Llama31_Tulu_70B
6. Run Phase 3 — tests scaling behavior

### Phase 3: Recipe comparison
7. Register: Llama31_Tulu2_8B, Llama3_Tulu2_8B, Llama2_Tulu2_7B, Llama2_Tulu2_13B
8. Run Phase 3 — tests Tulu 2 vs Tulu 3 recipe sensitivity

### Phase 4: Bonus (if time permits)
9. Think variants: OLMo3_7B_Think, OLMo3_32B_Think — reasoning vs safety alignment
10. Tulu v2.5 ablations — DPO data source sensitivity (14 models, same SFT)
11. Mixtral_Hermes — QLoRA DPO experiment
12. Llama31_Tulu_405B — only if multi-node GPU available

---

## VRAM Requirements (BF16 inference, Phase 3 needs SFT + DPO loaded)

| Model | Params | BF16/model | 2 models loaded | Min GPU |
|-------|--------|-----------|-----------------|---------|
| Gemma2B_SFT_DPO | ~2.5B | ~5 GB | ~10 GB | Any GPU |
| Phi2_SFT_DPO | ~2.7B | ~5.4 GB | ~11 GB | Any GPU |
| OLMo2_1B | ~1B | ~2 GB | ~4 GB | Any GPU |
| Qwen25_3B_Tulu | ~3B | ~6 GB | ~12 GB | Any GPU |
| Gemma7B_SFT_DPO | ~7B | ~14 GB | ~28 GB | A40 40GB |
| Qwen2_7B_DPOShift | ~7B | ~15 GB | ~30 GB | A40 40GB |
| Qwen25Math_7B_InfiAlign | ~7B | ~15 GB | ~30 GB | A40 40GB |
| OLMo2_7B | ~7B | ~14 GB | ~28 GB | A40 40GB |
| OLMoE_Tulu | 7B total | ~14 GB | ~28 GB | A40 40GB |
| OLMo3_7B | ~7B | ~14 GB | ~28 GB | A40 40GB |
| Zephyr_SFT_DPO | ~7B | ~14 GB | ~28 GB | A40 40GB |
| OLMo2_13B | ~13B | ~26 GB | ~52 GB | A100 80GB |
| OLMo2_32B | ~32B | ~64 GB | ~128 GB | 2x A100 80GB |
| OLMo31_32B | ~32B | ~64 GB | ~128 GB | 2x A100 80GB |
| Llama31_Tulu_70B | ~70B | ~140 GB | ~280 GB | 4x A100 80GB |
| Mixtral_Hermes | ~47B | ~94 GB | ~188 GB | 3x A100 80GB |
| Llama31_Tulu_405B | ~405B | ~810 GB | ~1.6 TB | Multi-node |

---

## Sources

**AllenAI Model Collections:**
- [Tulu 3 Models](https://huggingface.co/collections/allenai/tulu-3-models-673b8e0dc3512e30e7dc54f5)
- [OLMo-2 Models](https://huggingface.co/collections/allenai/olmo-2-702b8c5e5a9d54e816c2ba95)
- [OLMo-3 Models](https://huggingface.co/allenai?search=olmo-3)
- [AllenAI DPO search](https://huggingface.co/allenai/models?search=dpo)
- [AllenAI SFT search](https://huggingface.co/allenai/models?search=sft)

**Training Configs & Logs:**
- [Tulu 3 DPO config](https://github.com/allenai/open-instruct/blob/main/configs/train_configs/tulu3/tulu3_dpo_8b.yaml) — `use_lora: false`
- [OLMo-2 DPO config](https://github.com/allenai/open-instruct/blob/main/configs/train_configs/olmo2/olmo2_1124_7b_dpo.yaml) — `use_lora: false`
- [OLMoE logs](https://github.com/allenai/OLMoE/blob/main/logs/) — ZeRO-3, 32 GPUs
- [OLMo-Core](https://github.com/allenai/OLMo-core) — full FT framework, no LoRA support
- [Zephyr DPO config](https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_full.yaml) — no LoRA

**Non-AllenAI:**
- [NousResearch Mixtral discussion](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO/discussions/3) — SFT=Full FT, DPO=QLoRA
- [Zephyr SFT card](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)
- [Alignment Handbook](https://github.com/huggingface/alignment-handbook)
- [Phi-2 SFT](https://huggingface.co/lxuechen/phi-2-sft) | [Phi-2 DPO](https://huggingface.co/lxuechen/phi-2-dpo) — Full FT, Stanford (Xuechen Li), 94 Phi DPO models searched
- [Gemma 7B SFT](https://huggingface.co/lewtun/gemma-7b-sft-full-deita-10k-v0) | [Gemma 7B DPO](https://huggingface.co/lewtun/gemma-7b-dpo-full-mix1-beta-0.05-epoch-3) — Full FT, HuggingFace (Lewis Tunstall)
- [Gemma 2B SFT](https://huggingface.co/Columbia-NLP/gemma-2b-zephyr-sft) | [Gemma 2B DPO](https://huggingface.co/Columbia-NLP/gemma-2b-zephyr-dpo) — Full FT, Columbia NLP. 703 Gemma DPO models searched

**Qwen:**
- [Qwen2 7B SFT](https://huggingface.co/NoManDeRY/DPO-Shift-Qwen-2-7B-UltraChat200K-SFT) | [Qwen2 7B DPO](https://huggingface.co/NoManDeRY/DPO-Shift-Qwen-2-7B-Ultrafeedback-fixed-1.0) — Full FT, 8 GPU. Paper: [DPO-Shift (arxiv:2502.07599)](https://arxiv.org/abs/2502.07599)
- [Qwen2.5 3B SFT](https://huggingface.co/phunguyen01/II-Tulu-3B-SFT) | [Qwen2.5 3B DPO](https://huggingface.co/phunguyen01/Qwen-Tulu-3B-DPO) — Full FT, Tulu-3 data, Axolotl+TRL
- [Qwen2.5-Math 7B SFT](https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-SFT) | [Qwen2.5-Math 7B DPO](https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-DPO) — Full FT, multi-domain, math-biased base
- 359 Qwen DPO models searched total: [HuggingFace search](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending&search=qwen+++dpo)
