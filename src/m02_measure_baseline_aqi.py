"""
Phase 1: Measure Baseline AQI for 5 Architectures (GPU Only)
Uses src/AQI/05_evaluation/AQI/evaluation.py

    python src/m02_measure_baseline_aqi.py --mode sanity
    python src/m02_measure_baseline_aqi.py --mode full
    python src/m02_measure_baseline_aqi.py --models Llama3_8B Mistral_7B
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch

# =============================================================================
# PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
AQI_DIR = SRC_DIR / "AQI"

# Add paths for importing from AQI suite
sys.path.insert(0, str(AQI_DIR / "05_evaluation"))
sys.path.insert(0, str(AQI_DIR / "0a_AQI_EVAL_utils" / "src"))
sys.path.insert(0, str(AQI_DIR / "0c_utils"))

# Import from AQI suite
from eval_utils import (
    MODELS,
    load_model_for_eval,
    unload_model,
    verify_hf_repos,
)
from aqi.aqi_dealign_xb_chi import (
    set_seed,
    load_and_balance_dataset,
    analyze_by_axiom,
    create_metrics_summary,
    process_model_data,
)

# Import local config
from m01_config import (
    PHASE1_MODEL_KEYS, DATASET_NAME, GAMMA, DIM_REDUCTION,
    RANDOM_SEED, SAMPLES_SANITY, SAMPLES_FULL, BATCH_SIZE, OUTPUT_DIR
)


# =============================================================================
# GPU REQUIREMENT
# =============================================================================

def require_cuda():
    """Require CUDA GPU - exit if not available."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. No M1/CPU fallback.")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_phase1(model_keys, samples_per_category, output_dir):
    """Run AQI evaluation on specified models using eval_utils."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(RANDOM_SEED)

    # Load dataset once
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset_df = load_and_balance_dataset(
        DATASET_NAME,
        samples_per_category=samples_per_category,
        split="train"
    )
    print(f"Loaded {len(dataset_df)} samples")

    all_results = {}

    for model_key in model_keys:
        model_info = MODELS[model_key]
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_info['display_name']}")
        print(f"HF Repo: {model_info['hf_repo']}")
        print(f"{'=' * 60}")

        try:
            # Load model using eval_utils (handles standalone + LoRA)
            model, tokenizer = load_model_for_eval(model_key)

            # Get embeddings
            cache_file = model_output_dir / f"embeddings_{model_key}.pkl"
            processed_df = process_model_data(
                model, tokenizer, dataset_df,
                model_name=model_info["display_name"],
                cache_file=str(cache_file),
                batch_size=BATCH_SIZE
            )

            # Free GPU memory
            unload_model(model)
            del model
            torch.cuda.empty_cache()

            # Calculate AQI
            results, embeddings_3d, _, _ = analyze_by_axiom(
                processed_df,
                model_name=model_info["display_name"],
                gamma=GAMMA,
                dim_reduction_method=DIM_REDUCTION
            )

            # Save metrics summary
            create_metrics_summary(
                results,
                model_info["display_name"],
                output_dir=str(model_output_dir)
            )

            # Store result
            overall = results.get("overall", {})
            all_results[model_key] = {
                "model_key": model_key,
                "hf_repo": model_info["hf_repo"],
                "aqi_score": overall.get("AQI", 0.0),
                "chi_norm": overall.get("CHI_norm", 0.0),
                "xb_norm": overall.get("XB_norm", 0.0),
                "n_samples": len(dataset_df),
            }

            # Save individual result
            with open(model_output_dir / "result.json", "w") as f:
                json.dump(all_results[model_key], f, indent=2, default=str)

            print(f"\nAQI: {overall.get('AQI', 0):.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_results


def print_summary(results, output_dir):
    """Print summary and identify baseline model."""
    if not results:
        print("\nNo results.")
        return

    print(f"\n{'=' * 60}")
    print("PHASE 1 RESULTS: Baseline AQI Scores")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'AQI':<10} {'CHI_norm':<12} {'XB_norm':<10}")
    print("-" * 60)

    sorted_results = sorted(results.values(), key=lambda x: x["aqi_score"], reverse=True)

    for r in sorted_results:
        print(f"{r['model_key']:<15} {r['aqi_score']:<10.2f} {r['chi_norm']:<12.2f} {r['xb_norm']:<10.2f}")

    baseline = sorted_results[0]
    print("-" * 60)
    print(f"BASELINE MODEL: {baseline['model_key']} (AQI={baseline['aqi_score']:.2f})")
    print(f"{'=' * 60}")

    # Save summary
    summary = {
        "baseline_model": baseline["model_key"],
        "baseline_aqi": baseline["aqi_score"],
        "all_scores": {r["model_key"]: r["aqi_score"] for r in sorted_results},
        "timestamp": datetime.now().isoformat()
    }

    summary_path = Path(output_dir) / "phase1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Measure Baseline AQI")
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    require_cuda()

    samples = args.samples or (SAMPLES_SANITY if args.mode == "sanity" else SAMPLES_FULL)
    model_keys = args.models if args.models else PHASE1_MODEL_KEYS
    model_keys = [m for m in model_keys if m in MODELS]
    output_dir = args.output or str(OUTPUT_DIR)

    if not model_keys:
        print(f"ERROR: No valid models. Available: {list(MODELS.keys())}")
        sys.exit(1)

    # Verify HF repos exist
    model_keys = verify_hf_repos(model_keys, interactive=True)
    if not model_keys:
        print("No valid models after verification.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Phase 1: Multi-Architecture AQI Baseline (GPU Only)")
    print(f"{'=' * 60}")
    print(f"Mode: {args.mode} | Samples: {samples}")
    print(f"Models: {model_keys}")
    print(f"Output: {output_dir}")

    results = run_phase1(model_keys, samples, output_dir)
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
