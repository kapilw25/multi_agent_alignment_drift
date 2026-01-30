---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B
library_name: transformers
tags:
- llama
- sft
- tulu
- mahals
- alignment
datasets:
- allenai/tulu-3-sft-mixture
language:
- en
pipeline_tag: text-generation
---

# Llama-3.1-8B-Tulu10pct-SFT-MAHALS

Supervised Fine-Tuned (SFT) Llama 3.1 8B model trained on 10% of the Tulu-3 SFT mixture for the MAHALS research project.

## Model Details

| Attribute | Value |
|-----------|-------|
| **Base Model** | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| **Training Method** | Supervised Fine-Tuning (SFT) |
| **Dataset** | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) (10%) |
| **Framework** | [allenai/open-instruct](https://github.com/allenai/open-instruct) |
| **License** | [Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/) |

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

model_id = "anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS"

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
vllm serve anonymousML123/Llama-3.1-8B-Tulu10pct-SFT-MAHALS --max_model_len=4096
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
| Learning Rate | 5e-6 |
| Effective Batch Size | 128 |
| Gradient Accumulation | 16 |
| Max Sequence Length | 4096 |
| Epochs | 2 |
| LR Schedule | Linear |
| Warmup Ratio | 0.03 |
| Optimizer | AdamW |
| Precision | BF16 |

## Training Data

10% random sample (~94K examples) from [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), which includes:
- FLAN v2
- Open Assistant
- ShareGPT
- Code instructions
- Math instructions

## Intended Use

This model is intended for research on multi-agent alignment and instruction following. It is part of the MAHALS (Multi-Agent Hierarchical Alignment) research project.

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
