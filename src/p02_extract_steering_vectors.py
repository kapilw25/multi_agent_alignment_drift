"""
Phase 2.1: Extract Steering Vectors for Each Architecture (GPU Only)
D-STEER approach with checkpointing and hidden states caching. POC uses Base→Instruct pairs.

    python -u src/p02_extract_steering_vectors.py --mode sanity 2>&1 | tee logs/phase2_sanity.log
    python -u src/p02_extract_steering_vectors.py --mode full 2>&1 | tee logs/phase2_full.log
    python -u src/p02_extract_steering_vectors.py --mode sanity --models Mistral_7B Llama3_8B 2>&1 | tee logs/phase2_custom.log

Modes:
    --mode sanity: 100 samples (~1-2 hrs)
    --mode full:  1000 samples (~6-8 hrs)

Resources (A100 80GB, 6 models): Disk ~250GB | VRAM ~50GB peak | Batch: 4 (2 models loaded)

On GPU server, set in .env:
    HF_HOME=/workspace/volume/hf_cache
    TRANSFORMERS_CACHE=/workspace/volume/hf_cache
"""

import os
import sys
import gc
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from utils import load_model_registry, get_model_info
from utils.plot_steering import (
    plot_cosine_similarity,
    plot_steering_vector_norms,
    create_all_plots as create_steering_plots,
)
from utils.checkpoint import CheckpointManager, show_checkpoint_menu
from utils.cache import TensorCache
from m01_config import get_batch_size, BATCH_SIZE

# Load environment
load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

# =============================================================================
# CONFIGURATION (matching D_STEER notebook)
# =============================================================================

@dataclass
class Config:
    """Configuration matching D_STEER notebook."""
    # Dataset
    dataset_id: str = "Anthropic/hh-rlhf"

    # Steering vector settings
    steering_vector_num_samples_sanity: int = 100
    steering_vector_num_samples_full: int = 1000  # D_STEER uses 10000
    steering_layers: List[int] = None  # Will be set per model (last 5 layers)

    # SVD settings
    steering_vector_svd: bool = True
    steering_vector_svd_component: int = 3

    # Other
    use_chat_template: bool = False
    device: str = "cuda"
    random_seed: int = 42

    def __post_init__(self):
        if self.steering_layers is None:
            self.steering_layers = [-5, -4, -3, -2, -1]  # Last 5 layers (relative)


MODEL_REGISTRY = load_model_registry()
PHASE2_MODEL_KEYS = list(MODEL_REGISTRY.keys())
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2_steering_vectors"


# =============================================================================
# GPU UTILITIES
# =============================================================================

def require_cuda():
    """Require CUDA GPU - exit if not available."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. No M1/CPU fallback.")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def cleanup_gpu():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory():
    """Print current GPU memory status."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU Memory: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")


# =============================================================================
# DATA LOADING (matching D_STEER notebook)
# =============================================================================

def extract_hh_prompt_and_answer(text: str) -> Tuple[str, str]:
    """
    Extract prompt and answer from HH-RLHF format.
    Matching D_STEER notebook exactly.

    Returns:
        prompt_text: all turns except the final Assistant turn
        last_assistant: the final Assistant message (chosen/rejected)
    """
    # Split into alternating tokens: ["", "Human:", "msg", "Assistant:", "msg", ...]
    parts = re.split(r"(Human:|Assistant:)", text)

    # Build list [(role, msg), ...]
    turns = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            role = parts[i].rstrip(":")
            msg = parts[i + 1].strip()
            turns.append((role, msg))

    # Extract last assistant message
    last_assistant = None
    if turns and turns[-1][0] == "Assistant":
        last_assistant = turns[-1][1]
        turns = turns[:-1]  # remove it from the prompt

    # Reconstruct prompt text
    prompt_text = "\n".join(f"{role}: {msg}" for role, msg in turns)

    return prompt_text, last_assistant


def load_hh_rlhf_data(num_samples: int, seed: int = 42) -> List[Dict]:
    """
    Load Anthropic/hh-rlhf dataset with chosen/rejected pairs.
    Matching D_STEER notebook.
    """
    print(f"\nLoading Anthropic/hh-rlhf dataset ({num_samples} samples)...")

    dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    # Shuffle and select
    dataset = dataset.shuffle(seed=seed)

    samples = []
    for i, ex in enumerate(tqdm(dataset, desc="Processing samples", total=min(num_samples * 2, len(dataset)))):
        if len(samples) >= num_samples:
            break

        try:
            chosen_prompt, chosen_answer = extract_hh_prompt_and_answer(ex["chosen"])
            rejected_prompt, rejected_answer = extract_hh_prompt_and_answer(ex["rejected"])

            # Skip if missing data
            if not chosen_answer or not rejected_answer:
                continue

            samples.append({
                "chosen_prompt": chosen_prompt,
                "chosen_answer": chosen_answer,
                "rejected_prompt": rejected_prompt,
                "rejected_answer": rejected_answer,
            })
        except Exception as e:
            continue

    print(f"Extracted {len(samples)} valid samples")
    return samples


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer(model_name: str, device: str = "cuda") -> Tuple:
    """Load model and tokenizer. Matching D_STEER notebook."""
    print(f"  Loading: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force GPU-only loading
    max_memory = {0: "70GiB", "cpu": "0GiB"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # D_STEER uses bfloat16
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="flash_attention_2",
        )
        print("  Using Flash Attention 2")
    except Exception as e:
        print(f"  Flash Attention unavailable, using eager: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="eager",
        )

    model.eval()
    return model, tokenizer


def unload_model(model):
    """Free GPU memory."""
    if model is not None:
        del model
        cleanup_gpu()


# =============================================================================
# HIDDEN STATE EXTRACTION (matching D_STEER notebook)
# =============================================================================

def get_all_hidden_states(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extracts hidden states from all layers for the last token of a prompt.
    Matching D_STEER notebook exactly.

    Returns:
        torch.Tensor: Hidden states of shape [num_layers, hidden_dim]
                      Index 0 to num_layers-1: transformer layer outputs
                      (embedding layer excluded, matching D_STEER)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of length (num_layers + 1)
    # Stack into tensor: (num_layers + 1) x seq_len x hidden_dim
    hs = torch.stack(outputs.hidden_states, dim=0)

    # Get last token hidden states: (num_layers + 1) x hidden_dim
    hs_last_token = hs[:, :, -1, :]

    # Skip embedding layer (index 0), matching D_STEER
    all_hidden_states = hs_last_token[1:, 0, :]  # num_layers x hidden_dim

    return all_hidden_states


def get_hidden_states_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract hidden states from all layers for the last token of each text in batch.
    Batched version of get_all_hidden_states() for better GPU utilization.

    Args:
        model: The model to extract hidden states from
        tokenizer: The tokenizer
        texts: List of input texts
        batch_size: Number of texts to process at once
        device: Device to use

    Returns:
        torch.Tensor: Hidden states of shape [num_samples, num_layers, hidden_dim]
    """
    all_hidden_states = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Batches", leave=False):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is tuple of (num_layers + 1) tensors
        # Each tensor: (batch_size, seq_len, hidden_dim)

        # Get last token position for each sample (accounting for padding)
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # Last token index

        # Stack hidden states: (num_layers + 1, batch_size, seq_len, hidden_dim)
        hs = torch.stack(outputs.hidden_states, dim=0)

        # Extract last token hidden states for each sample in batch
        batch_hidden = []
        for b_idx in range(len(batch_texts)):
            last_idx = seq_lengths[b_idx].item()
            # Get hidden states at last token position, skip embedding layer
            sample_hs = hs[1:, b_idx, int(last_idx), :]  # num_layers x hidden_dim
            batch_hidden.append(sample_hs.cpu())

        all_hidden_states.extend(batch_hidden)

        # Clear GPU cache periodically
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    # Stack all samples: (num_samples, num_layers, hidden_dim)
    return torch.stack(all_hidden_states)


# =============================================================================
# STEERING VECTOR COMPUTATION (matching D_STEER notebook)
# =============================================================================

def compute_steering_vectors_dsteer(
    base_model,
    instruct_model,
    tokenizer_base,
    tokenizer_instruct,
    samples: List[Dict],
    config: Config,
    output_dir: Path,
    batch_size: int = 8,
    use_cache: bool = True,
) -> Dict:
    """
    Compute steering vectors using D_STEER approach with batched processing.

    Formula: h_delta = mean(h_instruct - h_base)
    Computes for both chosen and rejected responses.

    Args:
        batch_size: Number of samples to process at once (for A100 80GB)
        use_cache: Whether to use cached hidden states if available
    """
    device = config.device

    # Initialize cache for hidden states
    cache = TensorCache("hidden_states", output_dir)
    cache_keys = ["h_base_chosen", "h_base_rejected", "h_instruct_chosen", "h_instruct_rejected"]

    # Check if all hidden states are cached
    if use_cache and cache.exists_all(cache_keys):
        print(f"\nLoading cached hidden states from {output_dir}...")
        h_base_chosen_list = cache.load("h_base_chosen")
        h_base_rejected_list = cache.load("h_base_rejected")
        h_instruct_chosen_list = cache.load("h_instruct_chosen")
        h_instruct_rejected_list = cache.load("h_instruct_rejected")
        print(f"Loaded {len(h_base_chosen_list)} samples from cache")
    else:
        print(f"\nExtracting hidden states from {len(samples)} samples (batch_size={batch_size})...")

        # Prepare all texts upfront
        chosen_conversations = []
        rejected_conversations = []

        for sample in samples:
            chosen_prompt = sample["chosen_prompt"] + "\nAssistant:"
            chosen_conversation = chosen_prompt + sample["chosen_answer"]

            rejected_prompt = sample["rejected_prompt"] + "\nAssistant:"
            rejected_conversation = rejected_prompt + sample["rejected_answer"]

            chosen_conversations.append(chosen_conversation)
            rejected_conversations.append(rejected_conversation)

        print(f"  Prepared {len(chosen_conversations)} chosen + {len(rejected_conversations)} rejected texts")

        # Extract hidden states in batches (4 passes: base/instruct x chosen/rejected)
        print("\n[1/4] BASE model - chosen responses...")
        h_base_chosen_list = get_hidden_states_batch(
            base_model, tokenizer_base, chosen_conversations, batch_size, device
        )
        torch.cuda.empty_cache()

        print("[2/4] BASE model - rejected responses...")
        h_base_rejected_list = get_hidden_states_batch(
            base_model, tokenizer_base, rejected_conversations, batch_size, device
        )
        torch.cuda.empty_cache()

        print("[3/4] INSTRUCT model - chosen responses...")
        h_instruct_chosen_list = get_hidden_states_batch(
            instruct_model, tokenizer_instruct, chosen_conversations, batch_size, device
        )
        torch.cuda.empty_cache()

        print("[4/4] INSTRUCT model - rejected responses...")
        h_instruct_rejected_list = get_hidden_states_batch(
            instruct_model, tokenizer_instruct, rejected_conversations, batch_size, device
        )
        torch.cuda.empty_cache()

        # Cache the hidden states
        if use_cache:
            print("\nCaching hidden states...")
            cache.save_batch({
                "h_base_chosen": h_base_chosen_list,
                "h_base_rejected": h_base_rejected_list,
                "h_instruct_chosen": h_instruct_chosen_list,
                "h_instruct_rejected": h_instruct_rejected_list,
            })

    print(f"\nHidden states shape: {h_base_chosen_list.shape}")

    # Compute mean for cosine similarity
    h_base_chosen_mean = torch.mean(h_base_chosen_list, dim=0)
    h_base_rejected_mean = torch.mean(h_base_rejected_list, dim=0)
    h_instruct_chosen_mean = torch.mean(h_instruct_chosen_list, dim=0)
    h_instruct_rejected_mean = torch.mean(h_instruct_rejected_list, dim=0)

    # Compute differences (D_STEER formula)
    stacked_differences_chosen = h_instruct_chosen_list - h_base_chosen_list
    stacked_differences_rejected = h_instruct_rejected_list - h_base_rejected_list

    # Mean steering vectors
    h_delta_chosen = torch.mean(stacked_differences_chosen, dim=0)     # num_layers x hidden_dim
    h_delta_rejected = torch.mean(stacked_differences_rejected, dim=0)

    # Save main steering vectors
    torch.save(h_delta_chosen, output_dir / "steering_vector.pth")
    torch.save(h_delta_rejected, output_dir / "steering_vector_rejected.pth")

    # Compute cosine similarity
    chosen_similarity = torch.cosine_similarity(h_instruct_chosen_mean, h_base_chosen_mean, dim=1).to(torch.float32)
    rejected_similarity = torch.cosine_similarity(h_instruct_rejected_mean, h_base_rejected_mean, dim=1).to(torch.float32)

    # Plot cosine similarity (matching D_STEER)
    plot_cosine_similarity(
        chosen_similarity, rejected_similarity,
        output_path=str(output_dir / "cosine_similarity.png")
    )
    plt.close()

    print(f"\nShape of h_delta (chosen): {h_delta_chosen.shape}")
    print(f"Shape of h_delta (rejected): {h_delta_rejected.shape}")
    print(f"Norm of h_delta (chosen): {torch.norm(h_delta_chosen, p=2).item():.4f}")
    print(f"Norm of h_delta (rejected): {torch.norm(h_delta_rejected, p=2).item():.4f}")

    # Optional SVD decomposition
    steering_vector_final = h_delta_chosen
    if config.steering_vector_svd:
        print(f"\nApplying SVD decomposition (component {config.steering_vector_svd_component})...")
        steering_vector_final = apply_svd_decomposition(
            stacked_differences_chosen,
            h_delta_chosen,
            config.steering_vector_svd_component,
            output_dir
        )

    return {
        "steering_vector": steering_vector_final,
        "steering_vector_chosen": h_delta_chosen,
        "steering_vector_rejected": h_delta_rejected,
        "chosen_similarity": chosen_similarity,
        "rejected_similarity": rejected_similarity,
        "num_samples": len(samples),
        "num_layers": h_delta_chosen.shape[0],
        "hidden_dim": h_delta_chosen.shape[1],
    }


# =============================================================================
# SVD DECOMPOSITION (matching D_STEER notebook)
# =============================================================================

def apply_svd_decomposition(
    stacked_differences: torch.Tensor,
    h_delta: torch.Tensor,
    component: int,
    output_dir: Path,
) -> torch.Tensor:
    """
    Apply SVD decomposition to steering vectors.
    Matching D_STEER notebook.
    """
    num_layers = h_delta.shape[0]
    steering_vector_component = {}

    # Create output directory for explained variance plots
    explained_var_dir = output_dir / "explained_variances"
    explained_var_dir.mkdir(parents=True, exist_ok=True)

    for layer in tqdm(range(num_layers), desc="SVD per layer"):
        # Convert bfloat16 to float32 (numpy doesn't support bfloat16)
        X = stacked_differences[:, layer, :].float().numpy()  # num_samples x hidden_dim

        # Center the data
        X_centered = X - X.mean(axis=0)

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance
        explained_var = (S ** 2) / (S ** 2).sum()
        cumulative_var = np.cumsum(explained_var)

        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(explained_var[:20], marker='o', label='Individual explained variance')
        plt.plot(cumulative_var[:20], linestyle='--', label='Cumulative explained variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title(f'Explained Variance (Layer {layer})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(explained_var_dir / f"layer_{layer}_explained_variance.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Get steering vector for specified component
        # Scale by original steering vector norm (matching D_STEER)
        steering_vector = torch.tensor(Vt[component], dtype=torch.float32) * h_delta[layer].float().norm()
        steering_vector_component[(layer, component)] = steering_vector

    # Stack into tensor
    stacked = torch.stack([steering_vector_component[(layer, component)] for layer in range(num_layers)])

    # Save SVD steering vector
    torch.save(stacked, output_dir / f"steering_vector_svd{component}.pth")

    print(f"SVD steering vector shape: {stacked.shape}")

    return stacked




# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_steering_vectors(
    model_keys: List[str],
    config: Config,
    num_samples: int,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    use_cache: bool = True,
) -> Dict:
    """Extract steering vectors for specified models using D_STEER approach."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint manager
    ckpt = CheckpointManager("phase2_steering", output_dir)

    # Check for existing checkpoint
    if ckpt.exists():
        choice = show_checkpoint_menu(ckpt, model_keys)
        if choice == "complete":
            # All models done, return cached results
            print("Using cached results from completed run.")
            return ckpt.get_results()
        elif choice == "resume":
            # Get pending models
            pending_models = ckpt.get_pending_models(model_keys)
            all_results = ckpt.get_results()
            print(f"Resuming: {len(pending_models)} models remaining")
            model_keys = pending_models
        else:
            # Restart fresh
            all_results = {}
    else:
        all_results = {}

    # Load data once (Anthropic/hh-rlhf with chosen/rejected)
    samples = load_hh_rlhf_data(num_samples, seed=config.random_seed)

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            print(f"WARNING: {model_key} not in registry, skipping")
            continue

        model_info = get_model_info(model_key)
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Mark current model in checkpoint
        ckpt.set_current_model(model_key)

        print(f"\n{'=' * 60}")
        print(f"Extracting: {model_info['display_name']}")
        print(f"  Base:     {model_info['base']}")
        print(f"  Instruct: {model_info['instruct']}")
        print(f"{'=' * 60}")

        try:
            # === Load both models ===
            print("\n[1/3] Loading BASE model...")
            base_model, tokenizer_base = load_model_and_tokenizer(model_info["base"])
            print_gpu_memory()

            print("\n[2/3] Loading INSTRUCT model...")
            instruct_model, tokenizer_instruct = load_model_and_tokenizer(model_info["instruct"])
            print_gpu_memory()

            # === Compute steering vectors ===
            # Get per-model batch size for phase 2 (two models loaded)
            model_batch_size = get_batch_size(model_key, phase=2)
            print(f"\n[3/3] Computing steering vectors (batch_size={model_batch_size})...")
            result = compute_steering_vectors_dsteer(
                base_model,
                instruct_model,
                tokenizer_base,
                tokenizer_instruct,
                samples,
                config,
                model_output_dir,
                batch_size=model_batch_size,
                use_cache=use_cache,
            )

            # Unload models
            unload_model(base_model)
            unload_model(instruct_model)
            del tokenizer_base, tokenizer_instruct
            cleanup_gpu()

            # Plot steering vector norms
            plot_steering_vector_norms(
                result["steering_vector"],
                output_path=str(model_output_dir / "steering_vector_norms.png")
            )
            plt.close()

            # Save metadata
            meta_path = model_output_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "model_key": model_key,
                    "display_name": model_info["display_name"],
                    "base_model": model_info["base"],
                    "instruct_model": model_info["instruct"],
                    "num_samples": result["num_samples"],
                    "num_layers": result["num_layers"],
                    "hidden_dim": result["hidden_dim"],
                    "batch_size": model_batch_size,
                    "svd_enabled": config.steering_vector_svd,
                    "svd_component": config.steering_vector_svd_component if config.steering_vector_svd else None,
                    "mean_cosine_sim_chosen": result["chosen_similarity"].mean().item(),
                    "mean_cosine_sim_rejected": result["rejected_similarity"].mean().item(),
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)

            model_result = {
                "model_key": model_key,
                "display_name": model_info["display_name"],
                "num_layers": result["num_layers"],
                "hidden_dim": result["hidden_dim"],
                "mean_cosine_sim_chosen": result["chosen_similarity"].mean().item(),
                "mean_cosine_sim_rejected": result["rejected_similarity"].mean().item(),
                "output_dir": str(model_output_dir),
            }

            all_results[model_key] = model_result

            # Save checkpoint after each successful model
            ckpt.add_completed_model(model_key, model_result)

            print(f"\nSteering vector saved: {model_output_dir}")
            print(f"Mean cosine similarity (chosen): {result['chosen_similarity'].mean().item():.4f}")
            print(f"Mean cosine similarity (rejected): {result['rejected_similarity'].mean().item():.4f}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            cleanup_gpu()
            print_gpu_memory()

    # Mark checkpoint as complete
    ckpt.mark_complete()

    return all_results


def print_summary(results: Dict, output_dir: Path):
    """Print summary of extracted steering vectors."""
    if not results:
        print("\nNo results.")
        return

    print(f"\n{'=' * 70}")
    print("PHASE 2 RESULTS: Steering Vectors Extracted (D_STEER)")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'Layers':<8} {'Dim':<8} {'CosSim(C)':<12} {'CosSim(R)':<12}")
    print("-" * 70)

    for r in results.values():
        print(f"{r['model_key']:<15} {r['num_layers']:<8} {r['hidden_dim']:<8} "
              f"{r['mean_cosine_sim_chosen']:<12.4f} {r['mean_cosine_sim_rejected']:<12.4f}")

    print("-" * 70)
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}")

    # Save summary
    summary_path = output_dir / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "models": list(results.keys()),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nSummary: {summary_path}")

    # Generate comparison plots
    print(f"\n{'=' * 60}")
    print("Generating Steering Vector Plots")
    print(f"{'=' * 60}")
    plot_paths = create_steering_plots(output_dir)
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract Steering Vectors (D_STEER)")
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-svd", action="store_true", help="Disable SVD decomposition")
    parser.add_argument("--svd-component", type=int, default=3, help="SVD component to use")
    parser.add_argument("--no-cache", action="store_true", help="Disable hidden states caching")

    args = parser.parse_args()

    require_cuda()

    # Create config
    config = Config()
    config.steering_vector_svd = not args.no_svd
    config.steering_vector_svd_component = args.svd_component

    # Determine samples
    if args.samples:
        num_samples = args.samples
    elif args.mode == "sanity":
        num_samples = config.steering_vector_num_samples_sanity
    else:
        num_samples = config.steering_vector_num_samples_full

    model_keys = args.models if args.models else PHASE2_MODEL_KEYS
    model_keys = [m for m in model_keys if m in MODEL_REGISTRY]
    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    if not model_keys:
        print(f"ERROR: No valid models. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    use_cache = not args.no_cache

    print(f"\n{'=' * 60}")
    print("Phase 2: Extract Steering Vectors (D_STEER)")
    print(f"{'=' * 60}")
    print(f"Mode: {args.mode} | Samples: {num_samples}")
    print(f"Dataset: {config.dataset_id}")
    print(f"SVD: {'Enabled (component ' + str(config.steering_vector_svd_component) + ')' if config.steering_vector_svd else 'Disabled'}")
    print(f"Cache: {'Enabled' if use_cache else 'Disabled'}")
    print(f"Models: {model_keys}")
    print(f"Output: {output_dir}")

    results = extract_steering_vectors(model_keys, config, num_samples, output_dir, use_cache=use_cache)
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
