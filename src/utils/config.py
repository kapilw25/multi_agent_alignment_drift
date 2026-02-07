"""
Shared config: Evaluation settings for D-STEER alignment experiments.
Model definitions live in src/utils/model_registry.json (single source of truth).

    python src/utils/config.py
"""

from pathlib import Path

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

DATASET_NAME = "hasnat79/litmus"
GAMMA = 0.5
DIM_REDUCTION = "tsne"
RANDOM_SEED = 42

# LITMUS Dataset Info (from src/AQI/05_evaluation/eval_utils/dataset_info.py):
#   Total samples: ~20,439
#   Min per axiom: ~2,919
#   Structure: 7 axioms × 2 safety_labels × samples_per_category
#   Example: SAMPLES_FULL=500 → 7 × 2 × 500 = 7,000 total samples per inference
SAMPLES_SANITY = 100   # Quick validation: 7 × 2 × 100 = 1,400 samples
SAMPLES_FULL = 500     # Standard run: 7 × 2 × 500 = 7,000 samples
SAMPLES_MAX = 2000     # Near-max utilization: 7 × 2 × 2000 = 28,000 samples (capped by min_per_axiom ~2,919)

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1_baseline_aqi"


if __name__ == "__main__":
    print("=" * 50)
    print("utils/config: Shared Evaluation Settings")
    print("=" * 50)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
