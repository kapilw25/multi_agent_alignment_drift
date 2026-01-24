"""
Phase 3: Same-Architecture Steering Validation (GPU Only)
Validates steering vectors by applying them to base models at different lambda values.
Measures AQI vs lambda to verify monotonic alignment improvement. D_STEER-style implementation.

LITMUS Dataset (hasnat79/litmus): ~20,439 total samples, ~2,919 min per axiom
  Structure: 7 axioms × 2 safety_labels × samples_per_category

    python -u src/m04_same_arch_validation.py --mode sanity 2>&1 | tee logs/phase3_sanity.log
    python -u src/m04_same_arch_validation.py --mode full 2>&1 | tee logs/phase3_full.log
    python -u src/m04_same_arch_validation.py --mode max 2>&1 | tee logs/phase3_max.log
    python -u src/m04_same_arch_validation.py --mode sanity --models Zephyr_7B Falcon_7B 2>&1 | tee logs/phase3_custom.log
    python -u src/m04_same_arch_validation.py --mode sanity --lambdas 0.0 0.5 1.0 2>&1 | tee logs/phase3_sparse.log
    python -u src/m04_same_arch_validation.py --mode sanity --steering-layers -5 -4 -3 -2 -1 2>&1 | tee logs/phase3_last5.log
    python -u src/m04_same_arch_validation.py --mode sanity --all-layers --no-preserve-norm 2>&1 | tee logs/phase3_all.log

Modes:
    --mode sanity: 100 samples/category →  1,400 total (~1-2 hrs)
    --mode full:   500 samples/category →  7,000 total (~5-8 hrs)
    --mode max:   2000 samples/category → 28,000 total (~15-20 hrs)

Resources (A100 80GB, 3 models): Disk ~60GB | VRAM ~25GB peak | Batch: 16 (base model only)

On GPU server, set in .env:
    HF_HOME=/workspace/volume/hf_cache
    TRANSFORMERS_CACHE=/workspace/volume/hf_cache
"""

import os
import sys
import gc
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

# =============================================================================
# PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
AQI_DIR = SRC_DIR / "AQI"

# Load environment before importing transformers
load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

# Add paths for importing from AQI suite
sys.path.insert(0, str(AQI_DIR / "05_evaluation"))
sys.path.insert(0, str(AQI_DIR / "0a_AQI_EVAL_utils" / "src"))
sys.path.insert(0, str(AQI_DIR / "0c_utils"))
sys.path.insert(0, str(SRC_DIR))

# Import from AQI suite (for AQI calculation)
from eval_utils import (
    cleanup_gpu,
)
from aqi.aqi_dealign_xb_chi import (
    set_seed,
    load_and_balance_dataset,
    analyze_by_axiom,
    create_metrics_summary,
    process_model_data,
)

# Import local utilities
from utils import load_model_registry, get_model_info
from utils.checkpoint import CheckpointManager, show_checkpoint_menu
from m01_config import (
    DATASET_NAME, GAMMA, DIM_REDUCTION, RANDOM_SEED,
    SAMPLES_SANITY, SAMPLES_FULL, SAMPLES_MAX, get_batch_size,
)

# Lazy imports for transformers (after dotenv)
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for same-architecture validation."""
    # Lambda values for steering strength
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Steering vector source
    steering_vector_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "phase2_steering_vectors")
    steering_vector_type: str = "steering_vector.pth"  # or steering_vector_svd3.pth

    # Steering layers: which layers to apply steering (D_STEER uses last 5)
    # Use negative indices for relative positioning: [-5,-4,-3,-2,-1] = last 5 layers
    # None = all layers (not recommended - dilutes effect)
    steering_layers: Optional[List[int]] = field(default_factory=lambda: [-5, -4, -3, -2, -1])

    # Preserve norm: normalize steered hidden states to match original magnitude
    # Prevents hidden state explosion (recommended: True)
    preserve_norm: bool = True

    # Recommended models (from Phase 1 + Phase 2.1 analysis)
    recommended_models: List[str] = field(default_factory=lambda: [
        "Zephyr_7B",   # Highest steering potential (CosSim=0.77) + tied baseline
        "Falcon_7B",   # High potential (CosSim=0.78) + most room to improve
        "Mistral_7B",  # Baseline - validate steering doesn't degrade
    ])

    # Other
    device: str = "cuda"
    random_seed: int = 42


MODEL_REGISTRY = load_model_registry()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3_same_arch_validation"


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


def print_gpu_memory():
    """Print current GPU memory status."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU Memory: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")


# =============================================================================
# AUTO-GENERATE MISSING STEERING VECTORS
# =============================================================================

def generate_missing_steering_vectors(
    missing_models: List[str],
    steering_vector_dir: Path,
    mode: str = "sanity",
) -> bool:
    """
    Automatically run m03_extract_steering_vectors.py for missing models.

    Args:
        missing_models: List of model keys missing steering vectors
        steering_vector_dir: Directory where steering vectors should be stored
        mode: Evaluation mode (sanity/full/max)

    Returns:
        True if generation succeeded, False otherwise
    """
    # m03 only supports sanity/full modes, map max→full
    m03_mode = "full" if mode == "max" else mode

    print(f"\n{'=' * 60}")
    print("AUTO-GENERATING MISSING STEERING VECTORS")
    print(f"{'=' * 60}")
    print(f"Missing models: {missing_models}")
    print(f"m04 mode: {mode} → m03 mode: {m03_mode}")

    # Check for stale checkpoint (marked complete but files missing)
    checkpoint_path = steering_vector_dir / "phase2_steering_checkpoint.json"
    if checkpoint_path.exists():
        print(f"\nRemoving stale checkpoint: {checkpoint_path}")
        checkpoint_path.unlink()

    # Build command
    m03_script = PROJECT_ROOT / "src" / "m03_extract_steering_vectors.py"
    cmd = [
        sys.executable, "-u", str(m03_script),
        "--mode", m03_mode,
        "--models", *missing_models,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    try:
        # Run m03 script and stream output
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
        )

        # Verify steering vectors were created
        still_missing = []
        for model_key in missing_models:
            sv_path = steering_vector_dir / model_key / "steering_vector.pth"
            if not sv_path.exists():
                still_missing.append(model_key)

        if still_missing:
            print(f"\nERROR: Steering vectors still missing after generation: {still_missing}")
            return False

        print(f"\n{'=' * 60}")
        print("STEERING VECTORS GENERATED SUCCESSFULLY")
        print(f"{'=' * 60}\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: m03_extract_steering_vectors.py failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to run m03_extract_steering_vectors.py: {e}")
        return False


# =============================================================================
# STEERING HOOK IMPLEMENTATION
# =============================================================================

class SteeringHook:
    """
    Hook manager for applying steering vectors to model hidden states.

    Implements: h_steered[l] = h_base[l] + lambda * v[l]
    With optional norm preservation to prevent hidden state explosion.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        steering_vector: torch.Tensor,
        lambda_value: float = 1.0,
        steering_layers: Optional[List[int]] = None,
        preserve_norm: bool = True,
    ):
        """
        Initialize steering hook.

        Args:
            model: The language model to steer
            steering_vector: Tensor of shape [num_layers, hidden_dim]
            lambda_value: Steering strength (0=no steering, 1=full steering)
            steering_layers: List of layer indices to steer (negative = from end).
                            Default None = last 5 layers [-5,-4,-3,-2,-1]
            preserve_norm: If True, normalize steered hidden states to match original magnitude.
                          Prevents OOD hidden states from large steering vectors.
        """
        self.model = model
        self.steering_vector = steering_vector.to(model.device)
        self.lambda_value = lambda_value
        # If steering_layers is None, use all layers. Otherwise use specified layers.
        # Default in Config is [-5,-4,-3,-2,-1] (last 5 layers like D_STEER)
        self._use_all_layers = steering_layers is None
        self.steering_layers = steering_layers if steering_layers is not None else []
        self.preserve_norm = preserve_norm
        self.hooks = []
        self.layer_mapping = {}
        self.active_layer_indices = []  # Absolute indices of layers being steered

        # Detect model architecture and get decoder layers
        self._setup_layer_mapping()

    def _setup_layer_mapping(self):
        """Setup layer mapping based on model architecture."""
        # Common architectures and their layer paths
        if hasattr(self.model, 'model'):
            # LlamaForCausalLM, MistralForCausalLM, etc.
            if hasattr(self.model.model, 'layers'):
                self.decoder_layers = self.model.model.layers
                self.layer_attr = 'model.layers'
            elif hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
                # Some models use model.decoder.layers
                self.decoder_layers = self.model.model.decoder.layers
                self.layer_attr = 'model.decoder.layers'
            else:
                raise ValueError(f"Cannot find decoder layers in model architecture")
        elif hasattr(self.model, 'transformer'):
            # FalconForCausalLM uses transformer.h
            if hasattr(self.model.transformer, 'h'):
                self.decoder_layers = self.model.transformer.h
                self.layer_attr = 'transformer.h'
            else:
                raise ValueError(f"Cannot find decoder layers in Falcon-style architecture")
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")

        num_model_layers = len(self.decoder_layers)
        num_sv_layers = self.steering_vector.shape[0]

        if num_model_layers != num_sv_layers:
            print(f"WARNING: Model has {num_model_layers} layers, steering vector has {num_sv_layers}")

        self.num_layers = min(num_model_layers, num_sv_layers)

        # Determine which layers to steer
        if self._use_all_layers:
            # Steer all layers (not recommended but supported)
            self.active_layer_indices = list(range(self.num_layers))
        else:
            # Convert negative layer indices to absolute indices
            # e.g., [-5,-4,-3,-2,-1] with 32 layers -> [27,28,29,30,31]
            self.active_layer_indices = []
            for idx in self.steering_layers:
                abs_idx = idx if idx >= 0 else self.num_layers + idx
                if 0 <= abs_idx < self.num_layers:
                    self.active_layer_indices.append(abs_idx)
                else:
                    print(f"WARNING: Layer index {idx} (abs: {abs_idx}) out of range [0, {self.num_layers-1}], skipping")

            self.active_layer_indices = sorted(set(self.active_layer_indices))

        print(f"Model layers: {num_model_layers}, Steering vector layers: {num_sv_layers}")
        print(f"Steering {len(self.active_layer_indices)} layers: {self.active_layer_indices} via {self.layer_attr}")
        print(f"Preserve norm: {self.preserve_norm}")

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Apply steering: h_steered = h_base + lambda * v
            # steering_vector[layer_idx] has shape [hidden_dim]
            # hidden_states has shape [batch_size, seq_len, hidden_dim]
            sv = self.steering_vector[layer_idx].to(hidden_states.dtype)

            # Compute steered hidden states
            steered_hidden = hidden_states + self.lambda_value * sv

            # Optionally preserve original norm to prevent OOD hidden states
            # This is critical when steering vectors have large norms (see steering_norms_comparison.png)
            if self.preserve_norm:
                original_norm = hidden_states.norm(dim=-1, keepdim=True)
                steered_norm = steered_hidden.norm(dim=-1, keepdim=True)
                # Avoid division by zero
                steered_hidden = steered_hidden * (original_norm / (steered_norm + 1e-8))

            if rest is not None:
                return (steered_hidden,) + rest
            return steered_hidden

        return hook

    def register_hooks(self):
        """Register forward hooks on specified decoder layers."""
        self.remove_hooks()  # Clear any existing hooks

        for layer_idx in self.active_layer_indices:
            layer = self.decoder_layers[layer_idx]
            hook = layer.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} steering hooks on layers {self.active_layer_indices} (lambda={self.lambda_value}, preserve_norm={self.preserve_norm})")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_lambda(self, lambda_value: float):
        """Update lambda value (requires re-registering hooks)."""
        self.lambda_value = lambda_value
        if self.hooks:
            self.register_hooks()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_base_model(model_name: str, device: str = "cuda") -> Tuple:
    """Load base model and tokenizer."""
    print(f"  Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = {0: "70GiB", "cpu": "0GiB"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def load_steering_vector(model_key: str, config: Config) -> torch.Tensor:
    """Load steering vector for a model."""
    sv_path = config.steering_vector_dir / model_key / config.steering_vector_type
    if not sv_path.exists():
        raise FileNotFoundError(f"Steering vector not found: {sv_path}")

    sv = torch.load(sv_path, map_location="cpu")
    print(f"  Loaded steering vector: {sv.shape}")
    return sv


# =============================================================================
# AQI EVALUATION WITH STEERING
# =============================================================================

def evaluate_aqi_with_steering(
    model: AutoModelForCausalLM,
    tokenizer,
    steering_vector: torch.Tensor,
    lambda_value: float,
    dataset_df,
    model_key: str,
    output_dir: Path,
    batch_size: int = 8,
    steering_layers: Optional[List[int]] = None,
    preserve_norm: bool = True,
) -> Dict:
    """
    Evaluate AQI with steering applied at a specific lambda value.

    Args:
        model: Base model
        tokenizer: Tokenizer
        steering_vector: Steering vector tensor
        lambda_value: Steering strength
        dataset_df: LITMUS dataset DataFrame
        model_key: Model identifier
        output_dir: Output directory for this lambda evaluation
        batch_size: Batch size for processing
        steering_layers: List of layer indices to steer (negative = from end)
        preserve_norm: If True, normalize steered hidden states to match original magnitude

    Returns:
        Dict with AQI scores and metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use steering hook context manager with D_STEER-style configuration
    with SteeringHook(
        model, steering_vector, lambda_value,
        steering_layers=steering_layers,
        preserve_norm=preserve_norm,
    ) as hook:
        # Get model display name
        model_info = get_model_info(model_key)
        display_name = f"{model_info['display_name']} (lambda={lambda_value})"

        # Cache file for this specific lambda
        cache_file = output_dir / f"embeddings_{model_key}_lambda{lambda_value}.pkl"

        # Process model data (generate embeddings)
        processed_df = process_model_data(
            model, tokenizer, dataset_df,
            model_name=display_name,
            cache_file=str(cache_file),
            batch_size=batch_size,
        )

    # Calculate AQI (after removing hooks)
    results, embeddings_3d, _, _ = analyze_by_axiom(
        processed_df,
        model_name=display_name,
        gamma=GAMMA,
        dim_reduction_method=DIM_REDUCTION,
    )

    # Save metrics summary
    create_metrics_summary(
        results,
        display_name,
        output_dir=str(output_dir),
    )

    # Extract overall scores
    overall = results.get("overall", {})
    result = {
        "model_key": model_key,
        "lambda": lambda_value,
        "aqi_score": overall.get("AQI", 0.0),
        "chi_norm": overall.get("CHI_norm", 0.0),
        "xb_norm": overall.get("XB_norm", 0.0),
        "n_samples": len(dataset_df),
    }

    # Save individual result
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# =============================================================================
# PLOTTING
# =============================================================================

def plot_aqi_vs_lambda(
    results: List[Dict],
    model_key: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot AQI vs lambda curve for a single model.

    Args:
        results: List of result dicts with 'lambda' and 'aqi_score' keys
        model_key: Model identifier
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Sort by lambda
    results = sorted(results, key=lambda x: x["lambda"])
    lambdas = [r["lambda"] for r in results]
    aqi_scores = [r["aqi_score"] for r in results]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot AQI vs lambda
    ax.plot(lambdas, aqi_scores, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax.fill_between(lambdas, aqi_scores, alpha=0.3, color='#2ecc71')

    # Mark endpoints
    ax.axhline(y=aqi_scores[0], color='gray', linestyle='--', alpha=0.5, label=f'Base (lambda=0): {aqi_scores[0]:.1f}')
    ax.axhline(y=aqi_scores[-1], color='blue', linestyle='--', alpha=0.5, label=f'Full (lambda=1): {aqi_scores[-1]:.1f}')

    ax.set_xlabel('Lambda (Steering Strength)', fontsize=12)
    ax.set_ylabel('AQI Score', fontsize=12)
    ax.set_title(f'AQI vs Lambda: {model_key}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 100)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Check monotonicity
    is_monotonic = all(aqi_scores[i] <= aqi_scores[i+1] for i in range(len(aqi_scores)-1))
    monotonic_text = "Monotonic Increase" if is_monotonic else "NOT Monotonic"
    color = 'green' if is_monotonic else 'red'
    ax.text(0.95, 0.05, monotonic_text, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def plot_all_models_comparison(
    all_results: Dict[str, List[Dict]],
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot AQI vs lambda curves for all models in a single figure.

    Args:
        all_results: Dict mapping model_key -> list of results
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (model_key, results), color in zip(all_results.items(), colors):
        results = sorted(results, key=lambda x: x["lambda"])
        lambdas = [r["lambda"] for r in results]
        aqi_scores = [r["aqi_score"] for r in results]

        ax.plot(lambdas, aqi_scores, 'o-', linewidth=2, markersize=6,
                color=color, label=model_key)

    ax.set_xlabel('Lambda (Steering Strength)', fontsize=12)
    ax.set_ylabel('AQI Score', fontsize=12)
    ax.set_title('Same-Architecture Steering Validation', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 100)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def plot_per_axiom_curves(
    output_dir: Path,
    model_key: str,
    lambda_values: List[float],
    output_path: Path,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[plt.Figure]:
    """
    Plot per-axiom AQI curves across lambda values.

    Args:
        output_dir: Directory containing lambda subdirectories
        model_key: Model identifier
        lambda_values: List of lambda values evaluated
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure or None if no data found
    """
    import pandas as pd

    # Collect per-axiom data across lambda values
    axiom_data = {}  # axiom -> list of (lambda, aqi) tuples

    for lambda_val in sorted(lambda_values):
        lambda_dir = output_dir / f"lambda_{lambda_val:.2f}"
        csv_files = list(lambda_dir.glob("*_metrics_summary.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            df = df[df["Category"] != "overall"]

            for _, row in df.iterrows():
                axiom = row["Category"]
                aqi = row["AQI [0-100] (↑)"]
                if axiom not in axiom_data:
                    axiom_data[axiom] = []
                axiom_data[axiom].append((lambda_val, aqi))

    if not axiom_data:
        print(f"No per-axiom data found for {model_key}")
        return None

    # Create subplot grid
    n_axioms = len(axiom_data)
    n_cols = 3
    n_rows = (n_axioms + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_axioms > 1 else [axes]

    for idx, (axiom, data) in enumerate(sorted(axiom_data.items())):
        ax = axes[idx]
        data = sorted(data, key=lambda x: x[0])
        lambdas = [d[0] for d in data]
        aqis = [d[1] for d in data]

        ax.plot(lambdas, aqis, 'o-', linewidth=2, markersize=6, color='#3498db')
        ax.set_title(axiom, fontsize=11, fontweight='bold')
        ax.set_xlabel('Lambda')
        ax.set_ylabel('AQI')
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_axioms, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Per-Axiom AQI vs Lambda: {model_key}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_same_arch_validation(
    model_keys: List[str],
    config: Config,
    samples_per_category: int,
    output_dir: Path,
) -> Dict[str, List[Dict]]:
    """
    Run same-architecture steering validation for specified models.

    Args:
        model_keys: List of model keys to validate
        config: Configuration object
        samples_per_category: Number of samples per axiom category
        output_dir: Output directory

    Returns:
        Dict mapping model_key -> list of results at each lambda
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint manager
    ckpt = CheckpointManager("phase3_validation", output_dir)

    # Check for existing checkpoint
    if ckpt.exists():
        choice = show_checkpoint_menu(ckpt, model_keys)
        if choice == "complete":
            print("Using cached results from completed run.")
            return ckpt.get_results()
        elif choice == "resume":
            pending_models = ckpt.get_pending_models(model_keys)
            all_results = ckpt.get_results()
            print(f"Resuming: {len(pending_models)} models remaining")
            model_keys = pending_models
        else:
            all_results = {}
    else:
        all_results = {}

    set_seed(RANDOM_SEED)

    # Load dataset once (LITMUS - same as Phase 1)
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset_df = load_and_balance_dataset(
        DATASET_NAME,
        samples_per_category=samples_per_category,
        split="train",
    )
    print(f"Loaded {len(dataset_df)} samples")

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
        print(f"Validating: {model_info['display_name']}")
        print(f"Base Model: {model_info['base']}")
        print(f"Lambda values: {config.lambda_values}")
        print(f"{'=' * 60}")

        model = None
        tokenizer = None
        lambda_results = []

        try:
            # Load steering vector
            print("\n[1/3] Loading steering vector...")
            steering_vector = load_steering_vector(model_key, config)

            # Load base model
            print("\n[2/3] Loading base model...")
            model, tokenizer = load_base_model(model_info["base"])
            print_gpu_memory()

            # Get batch size
            batch_size = get_batch_size(model_key, phase=1)
            print(f"Using batch_size={batch_size}")

            # Evaluate at each lambda
            print(f"\n[3/3] Evaluating AQI at {len(config.lambda_values)} lambda values...")

            for lambda_val in config.lambda_values:
                print(f"\n--- Lambda = {lambda_val:.2f} ---")

                lambda_dir = model_output_dir / f"lambda_{lambda_val:.2f}"

                result = evaluate_aqi_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    steering_vector=steering_vector,
                    lambda_value=lambda_val,
                    dataset_df=dataset_df,
                    model_key=model_key,
                    output_dir=lambda_dir,
                    batch_size=batch_size,
                    steering_layers=config.steering_layers,
                    preserve_norm=config.preserve_norm,
                )

                lambda_results.append(result)
                print(f"AQI(lambda={lambda_val}): {result['aqi_score']:.2f}")

                # Cleanup between lambda evaluations
                cleanup_gpu()

            # Store results
            all_results[model_key] = lambda_results

            # Save checkpoint
            ckpt.add_completed_model(model_key, {"lambda_results": lambda_results})

            # Generate per-model plots
            print(f"\nGenerating plots for {model_key}...")

            # AQI vs Lambda curve
            plot_aqi_vs_lambda(
                lambda_results,
                model_key,
                model_output_dir / "aqi_vs_lambda.png",
            )
            plt.close()

            # Per-axiom curves
            plot_per_axiom_curves(
                model_output_dir,
                model_key,
                config.lambda_values,
                model_output_dir / "per_axiom_curves.png",
            )
            plt.close()

            # Save results JSON
            with open(model_output_dir / "aqi_vs_lambda.json", "w") as f:
                json.dump(lambda_results, f, indent=2, default=str)

            # Check monotonicity
            aqi_scores = [r["aqi_score"] for r in sorted(lambda_results, key=lambda x: x["lambda"])]
            is_monotonic = all(aqi_scores[i] <= aqi_scores[i+1] for i in range(len(aqi_scores)-1))
            improvement = aqi_scores[-1] - aqi_scores[0]

            print(f"\nResults for {model_key}:")
            print(f"  AQI(lambda=0): {aqi_scores[0]:.2f}")
            print(f"  AQI(lambda=1): {aqi_scores[-1]:.2f}")
            print(f"  Improvement: {improvement:+.2f}")
            print(f"  Monotonic: {'YES' if is_monotonic else 'NO'}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if model is not None:
                unload_model(model)
            if tokenizer is not None:
                del tokenizer
            cleanup_gpu()
            print_gpu_memory()

    # Mark checkpoint as complete
    ckpt.mark_complete()

    return all_results


def print_summary(results: Dict[str, List[Dict]], output_dir: Path):
    """Print summary of validation results."""
    if not results:
        print("\nNo results.")
        return

    print(f"\n{'=' * 80}")
    print("PHASE 2.2 RESULTS: Same-Architecture Steering Validation")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'AQI(0)':<10} {'AQI(1)':<10} {'Delta':<10} {'Monotonic':<12}")
    print("-" * 80)

    summary_data = []

    for model_key, lambda_results in results.items():
        sorted_results = sorted(lambda_results, key=lambda x: x["lambda"])
        aqi_0 = sorted_results[0]["aqi_score"]
        aqi_1 = sorted_results[-1]["aqi_score"]
        delta = aqi_1 - aqi_0

        aqi_scores = [r["aqi_score"] for r in sorted_results]
        is_monotonic = all(aqi_scores[i] <= aqi_scores[i+1] for i in range(len(aqi_scores)-1))

        print(f"{model_key:<15} {aqi_0:<10.2f} {aqi_1:<10.2f} {delta:+<10.2f} {'YES' if is_monotonic else 'NO':<12}")

        summary_data.append({
            "model_key": model_key,
            "aqi_lambda_0": aqi_0,
            "aqi_lambda_1": aqi_1,
            "delta": delta,
            "is_monotonic": is_monotonic,
            "all_lambdas": {r["lambda"]: r["aqi_score"] for r in sorted_results},
        })

    print("-" * 80)
    print(f"Output: {output_dir}")
    print(f"{'=' * 80}")

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_all_models_comparison(results, output_dir / "all_models_comparison.png")
    plt.close()

    # Save summary JSON
    summary_path = output_dir / "phase3_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "summary": summary_data,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nSummary: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Same-Architecture Steering Validation"
    )
    parser.add_argument(
        "--mode", choices=["sanity", "full", "max"], default="sanity",
        help="Evaluation mode (sanity=100, full=500, max=2000 samples per category)"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to validate (default: recommended models)"
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=None,
        help="Lambda values to test (default: 0.0 0.25 0.5 0.75 1.0)"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Override samples per category"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--steering-type", type=str, default="steering_vector.pth",
        choices=["steering_vector.pth", "steering_vector_svd3.pth"],
        help="Type of steering vector to use"
    )
    parser.add_argument(
        "--steering-layers", nargs="+", type=int, default=None,
        help="Layer indices to steer (default: -5 -4 -3 -2 -1 = last 5 layers). Use negative for relative indexing."
    )
    parser.add_argument(
        "--no-preserve-norm", action="store_true",
        help="Disable norm preservation (not recommended - may cause OOD hidden states)"
    )
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Steer all layers instead of just last 5 (not recommended)"
    )

    args = parser.parse_args()

    require_cuda()

    # Create config
    config = Config()
    config.steering_vector_type = args.steering_type

    if args.lambdas:
        config.lambda_values = sorted(args.lambdas)

    # Steering layers configuration
    if args.all_layers:
        config.steering_layers = None  # Will default to all layers in SteeringHook
        print("WARNING: --all-layers enabled. Steering ALL layers (not recommended).")
    elif args.steering_layers:
        config.steering_layers = args.steering_layers

    # Preserve norm configuration
    config.preserve_norm = not args.no_preserve_norm
    if args.no_preserve_norm:
        print("WARNING: --no-preserve-norm enabled. Hidden states may explode.")

    # Determine samples based on mode
    if args.samples:
        samples = args.samples
    elif args.mode == "sanity":
        samples = SAMPLES_SANITY  # 100 per category → 1,400 total
    elif args.mode == "full":
        samples = SAMPLES_FULL   # 500 per category → 7,000 total
    else:  # max
        samples = SAMPLES_MAX    # 2000 per category → 28,000 total

    # Determine models
    if args.models:
        model_keys = [m for m in args.models if m in MODEL_REGISTRY]
    else:
        model_keys = [m for m in config.recommended_models if m in MODEL_REGISTRY]

    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    if not model_keys:
        print(f"ERROR: No valid models. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # Check steering vectors exist, auto-generate if missing
    missing_sv = []
    for model_key in model_keys:
        sv_path = config.steering_vector_dir / model_key / config.steering_vector_type
        if not sv_path.exists():
            missing_sv.append(model_key)

    if missing_sv:
        print(f"Steering vectors not found for: {missing_sv}")
        print(f"Auto-generating using m03_extract_steering_vectors.py...")

        success = generate_missing_steering_vectors(
            missing_models=missing_sv,
            steering_vector_dir=config.steering_vector_dir,
            mode=args.mode,
        )

        if not success:
            print(f"ERROR: Failed to generate steering vectors for: {missing_sv}")
            sys.exit(1)

        # Re-verify after generation
        still_missing = []
        for model_key in missing_sv:
            sv_path = config.steering_vector_dir / model_key / config.steering_vector_type
            if not sv_path.exists():
                still_missing.append(model_key)

        if still_missing:
            print(f"ERROR: Steering vectors still missing after generation: {still_missing}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Phase 3: Same-Architecture Steering Validation")
    print(f"{'=' * 60}")
    print(f"Mode: {args.mode} | Samples: {samples}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Lambda values: {config.lambda_values}")
    print(f"Steering type: {config.steering_vector_type}")
    print(f"Steering layers: {config.steering_layers if config.steering_layers else 'ALL (not recommended)'}")
    print(f"Preserve norm: {config.preserve_norm}")
    print(f"Models: {model_keys}")
    print(f"Output: {output_dir}")

    results = run_same_arch_validation(model_keys, config, samples, output_dir)
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
