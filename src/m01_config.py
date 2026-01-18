"""
Phase 1 Config: Settings for Multi-Agent Alignment Baseline
Models are defined in src/AQI/05_evaluation/eval_utils/model_loader.py

    python src/m01_config.py
"""

from pathlib import Path

# =============================================================================
# PHASE 1 MODEL KEYS (defined in AQI/05_evaluation/eval_utils/model_loader.py)
# =============================================================================

PHASE1_MODEL_KEYS = [
    "Llama3_8B",
    # "Phi3_Mini",  # Skipped: DynamicCache incompatibility with transformers 4.56+
    "Mistral_7B",
    "Qwen2_7B",
    "Gemma2_9B",
    "Falcon_7B",
    "Zephyr_7B",
    # "Yi_6B",  # Skipped: Disk space error
    # "DeepSeek_7B",  # Skipped: Disk space error
]

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

DATASET_NAME = "hasnat79/litmus"
GAMMA = 0.5
DIM_REDUCTION = "tsne"
RANDOM_SEED = 42
SAMPLES_SANITY = 100
SAMPLES_FULL = 200
BATCH_SIZE = 16  # Default batch size

# Per-model batch sizes (for memory management on A100 80GB)
# With 80GB VRAM, larger batch sizes improve throughput
MODEL_BATCH_SIZES = {
    "Llama3_8B": 16,   # ~16 GB model
    "Phi3_Mini": 16,   # ~8 GB model
    "Mistral_7B": 16,  # ~14 GB model
    "Qwen2_7B": 16,    # ~14 GB model
    "Gemma2_9B": 8,    # ~18 GB model - larger model, conservative batch
    "Falcon_7B": 16,   # ~14 GB model
    "Zephyr_7B": 16,   # ~14 GB model
    "Yi_6B": 16,       # ~12 GB model
    "DeepSeek_7B": 16, # ~14 GB model
}

def get_batch_size(model_key: str) -> int:
    """Get batch size for a model, with fallback to default."""
    return MODEL_BATCH_SIZES.get(model_key, BATCH_SIZE)

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1_baseline_aqi"


if __name__ == "__main__":
    print("=" * 50)
    print("Phase 1: Multi-Architecture AQI Config")
    print("=" * 50)
    print(f"Models: {PHASE1_MODEL_KEYS}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
