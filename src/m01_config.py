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
    "Phi3_Mini",
    "Mistral_7B",
    "Qwen2_7B",
    "Gemma2_9B",
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
BATCH_SIZE = 4

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
