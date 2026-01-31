---
license: llama3.1
base_model: anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS
library_name: transformers
tags:
- llama
- dpo
- tulu
- mahals
- alignment
- preference-learning
datasets:
- allenai/llama-3.1-tulu-3-8b-preference-mixture
language:
- en
pipeline_tag: text-generation
---

# Llama-3.1-8B-Tulu10pct-DPO-MAHALS

Direct Preference Optimization (DPO) model trained on 10% of the Tulu-3 preference mixture for the MAHALS research project.

## Model Details

| Attribute | Value |
|-----------|-------|
| **Base Model** | [anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS](https://huggingface.co/anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS) |
| **Training Method** | Direct Preference Optimization (DPO) |
| **Dataset** | [allenai/llama-3.1-tulu-3-8b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture) (10%) |
| **Framework** | [allenai/open-instruct](https://github.com/allenai/open-instruct) |
| **License** | [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/) |

## Training Pipeline

```
meta-llama/Llama-3.1-8B
        ↓ SFT (10% Tulu-3 SFT mixture)
anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS
        ↓ DPO (10% Tulu-3 preference mixture)
anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS  ← This model
```

## Inference Requirements

| Precision | VRAM Required | Compatible GPUs |
|-----------|---------------|-----------------|
| BF16/FP16 | ~20 GB | A100 40GB, RTX 4090/3090, A10 |
| INT8 | ~10 GB | T4, RTX 3080 |
| INT4 | ~6 GB | RTX 3060, consumer GPUs |

## Usage

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [{"role": "user", "content": "What is machine learning?"}]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### vLLM Serving

```bash
vllm serve anonymousML123/Llama-3.1-8B-Tulu10pct-DPO-MAHALS --max_model_len=2048
```

## Chat Template

This model uses the Tulu chat template:

```
<|user|>
Your question here
<|assistant|>
Model response<|endoftext|>
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-7 |
| Effective Batch Size | 32 |
| Gradient Accumulation | 16 |
| Max Sequence Length | 2048 |
| Epochs | 1 |
| LR Schedule | Linear |
| Warmup Ratio | 0.1 |
| DPO Beta | 5 |
| DPO Loss Type | dpo_norm |
| Optimizer | AdamW |
| Precision | BF16 |

## DPO Training Metrics

| Step | Loss | Notes |
|------|------|-------|
| 1 | 0.693 | Initial (log(2)) |
| 103 | 0.634 | -8.6% |
| 197 | 0.521 | -25% |
| 844 | TBD | Final |

## Training Data

10% random sample from [allenai/llama-3.1-tulu-3-8b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture), which contains preference pairs for:
- Helpfulness
- Harmlessness
- Honesty
- Instruction following

## Intended Use

This model is intended for research on multi-agent alignment and preference learning. It is part of the MAHALS (Multi-Agent Hierarchical Alignment) research project.

## Limitations

- Trained on 10% of data (reduced capability vs full Tulu-3)
- English only
- May exhibit biases present in training data
- Not suitable for production without further evaluation

## Citation

```bibtex
@misc{mahals2026,
  title={MAHALS: Multi-Agent Hierarchical Alignment},
  author={Anonymous},
  year={2026},
  note={Under review}
}
```

## Acknowledgments

Built using [AllenAI's open-instruct](https://github.com/allenai/open-instruct) framework.
