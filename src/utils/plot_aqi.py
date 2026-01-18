"""
Bar plots for AQI scores across LLMs.

    python src/utils/plot_aqi.py --output_dir outputs/phase1_baseline_aqi
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_aqi_results(output_dir: str) -> Dict[str, dict]:
    """Load AQI results from individual model result.json files."""
    output_dir = Path(output_dir)
    results = {}

    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            result_file = model_dir / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    results[data["model_key"]] = data

    return results


def plot_aqi_bar(
    results: Dict[str, dict],
    output_path: Optional[str] = None,
    title: str = "AQI Scores by Model",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create bar plot of AQI scores across models.

    Args:
        results: Dict mapping model_key to result dict with 'aqi_score'
        output_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    # Sort by AQI score descending
    sorted_models = sorted(results.items(), key=lambda x: x[1]["aqi_score"], reverse=True)

    models = [m[0] for m in sorted_models]
    scores = [m[1]["aqi_score"] for m in sorted_models]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars - highlight best model
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(models))]

    bars = ax.bar(models, scores, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("AQI Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)

    # Add baseline annotation
    if sorted_models:
        best_model = sorted_models[0][0]
        best_score = sorted_models[0][1]["aqi_score"]
        ax.axhline(y=best_score, color="green", linestyle="--", alpha=0.5, label=f"Baseline: {best_model}")
        ax.legend(loc="upper right")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_aqi_comparison(
    results: Dict[str, dict],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Create comparison plot showing AQI, CHI_norm, and XB_norm.

    Args:
        results: Dict mapping model_key to result dict
        output_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    # Sort by AQI score descending
    sorted_models = sorted(results.items(), key=lambda x: x[1]["aqi_score"], reverse=True)

    models = [m[0] for m in sorted_models]
    aqi_scores = [m[1]["aqi_score"] for m in sorted_models]
    chi_norms = [m[1].get("chi_norm", 0) for m in sorted_models]
    xb_norms = [m[1].get("xb_norm", 0) for m in sorted_models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, aqi_scores, width, label="AQI", color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, chi_norms, width, label="CHI_norm", color="#3498db", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, xb_norms, width, label="XB_norm", color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("AQI Components Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_axiom_breakdown(
    output_dir: str,
    model_keys: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Create heatmap of AQI scores per axiom for each model.

    Args:
        output_dir: Directory containing model subdirectories
        model_keys: List of model keys to include (None for all)
        output_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    output_dir = Path(output_dir)

    # Collect per-axiom data
    data = {}
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            model_key = model_dir.name
            if model_keys and model_key not in model_keys:
                continue

            # Find metrics CSV
            csv_files = list(model_dir.glob("*_metrics_summary.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                # Exclude 'overall' row
                df = df[df["Category"] != "overall"]
                data[model_key] = dict(zip(df["Category"], df["AQI [0-100] (↑)"]))

    if not data:
        print("No axiom data found")
        return None

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort columns by mean AQI
    col_order = df.mean().sort_values(ascending=False).index
    df = df[col_order]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    # Labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)

    # Add value annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("AQI by Axiom and Model", fontsize=14, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("AQI Score", fontsize=11)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def create_all_plots(output_dir: str) -> Dict[str, str]:
    """
    Create all standard AQI plots and save to output directory.

    Args:
        output_dir: Directory containing model results

    Returns:
        Dict mapping plot name to file path
    """
    output_dir = Path(output_dir)
    results = load_aqi_results(output_dir)

    if not results:
        print("No results found")
        return {}

    paths = {}

    # 1. AQI Bar Plot
    path = output_dir / "aqi_bar_plot.png"
    plot_aqi_bar(results, output_path=str(path), title="Phase 1: Baseline AQI Scores")
    paths["aqi_bar"] = str(path)
    plt.close()

    # 2. AQI Components Comparison
    path = output_dir / "aqi_comparison_plot.png"
    plot_aqi_comparison(results, output_path=str(path))
    paths["aqi_comparison"] = str(path)
    plt.close()

    # 3. Axiom Breakdown Heatmap
    path = output_dir / "aqi_axiom_heatmap.png"
    fig = plot_axiom_breakdown(output_dir, output_path=str(path))
    if fig:
        paths["axiom_heatmap"] = str(path)
        plt.close()

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate AQI bar plots")
    parser.add_argument("--output_dir", type=str, default="outputs/phase1_baseline_aqi")
    args = parser.parse_args()

    paths = create_all_plots(args.output_dir)

    print("\nGenerated plots:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
