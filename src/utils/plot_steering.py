"""
Bar plots and visualizations for steering vectors across LLMs.

    python src/utils/plot_steering.py --output_dir outputs/phase2_steering_vectors
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_steering_results(output_dir: str) -> Dict[str, dict]:
    """Load steering vector results from individual model metadata.json files."""
    output_dir = Path(output_dir)
    results = {}

    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            meta_file = model_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    data = json.load(f)
                    results[data["model_key"]] = data

    return results


def plot_cosine_similarity(
    chosen_similarity: torch.Tensor,
    rejected_similarity: torch.Tensor,
    output_path: Optional[str] = None,
    title: str = "Cosine Similarity (Instruct vs Base) per Layer",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot cosine similarity across layers.

    Args:
        chosen_similarity: Tensor of cosine similarities for chosen responses
        rejected_similarity: Tensor of cosine similarities for rejected responses
        output_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(chosen_similarity, torch.Tensor):
        chosen_similarity = chosen_similarity.numpy()
    if isinstance(rejected_similarity, torch.Tensor):
        rejected_similarity = rejected_similarity.numpy()

    ax.plot(chosen_similarity, label='Chosen', color='blue', marker='o')
    ax.plot(rejected_similarity, label='Rejected', color='red', marker='x')
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_steering_vector_norms(
    steering_vector: torch.Tensor,
    output_path: Optional[str] = None,
    title: str = "Steering Vector Magnitude per Layer",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot steering vector magnitude per layer.

    Args:
        steering_vector: Tensor of shape [num_layers, hidden_dim]
        output_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if isinstance(steering_vector, torch.Tensor):
        norms = torch.norm(steering_vector, dim=1).numpy()
    else:
        norms = np.linalg.norm(steering_vector, axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(norms)), norms, color='green', alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_cosine_similarity_comparison(
    results: Dict[str, dict],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Create bar plot comparing mean cosine similarity across models.

    Args:
        results: Dict mapping model_key to metadata dict
        output_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    # Sort by chosen cosine similarity
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get("mean_cosine_sim_chosen", 0),
        reverse=True
    )

    models = [m[0] for m in sorted_models]
    chosen_sims = [m[1].get("mean_cosine_sim_chosen", 0) for m in sorted_models]
    rejected_sims = [m[1].get("mean_cosine_sim_rejected", 0) for m in sorted_models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width/2, chosen_sims, width, label="Chosen", color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, rejected_sims, width, label="Rejected", color="#e74c3c", edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars1, chosen_sims):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, rejected_sims):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title("Mean Cosine Similarity (Instruct vs Base)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_model_dimensions(
    results: Dict[str, dict],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Create bar plot comparing model dimensions (layers, hidden_dim).

    Args:
        results: Dict mapping model_key to metadata dict
        output_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    models = list(results.keys())
    num_layers = [results[m].get("num_layers", 0) for m in models]
    hidden_dims = [results[m].get("hidden_dim", 0) / 100 for m in models]  # Scale for visibility

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=figsize)

    bars1 = ax1.bar(x - width/2, num_layers, width, label="Num Layers", color="#3498db", edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Number of Layers", fontsize=12, color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, [results[m].get("hidden_dim", 0) for m in models], width,
                    label="Hidden Dim", color="#9b59b6", edgecolor="black", linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("Hidden Dimension", fontsize=12, color="#9b59b6")
    ax2.tick_params(axis="y", labelcolor="#9b59b6")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.set_title("Model Architecture Comparison", fontsize=14, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_steering_norms_comparison(
    output_dir: str,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot steering vector norms across all models in a single figure.

    Args:
        output_dir: Directory containing model subdirectories
        output_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    output_dir = Path(output_dir)

    # Load steering vectors for each model
    model_norms = {}
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            sv_file = model_dir / "steering_vector.pth"
            if sv_file.exists():
                sv = torch.load(sv_file, map_location="cpu")
                norms = torch.norm(sv, dim=1).numpy()
                model_norms[model_dir.name] = norms

    if not model_norms:
        print("No steering vectors found")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_norms)))

    for (model_key, norms), color in zip(model_norms.items(), colors):
        ax.plot(norms, label=model_key, color=color, marker='o', markersize=3, linewidth=1.5)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title("Steering Vector Norms Across Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def create_all_plots(output_dir: str) -> Dict[str, str]:
    """
    Create all standard steering vector plots and save to output directory.

    Args:
        output_dir: Directory containing model results

    Returns:
        Dict mapping plot name to file path
    """
    output_dir = Path(output_dir)
    results = load_steering_results(output_dir)

    if not results:
        print("No results found")
        return {}

    paths = {}

    # 1. Cosine Similarity Comparison
    path = output_dir / "cosine_similarity_comparison.png"
    plot_cosine_similarity_comparison(results, output_path=str(path))
    paths["cosine_similarity_comparison"] = str(path)
    plt.close()

    # 2. Model Dimensions Comparison
    path = output_dir / "model_dimensions.png"
    plot_model_dimensions(results, output_path=str(path))
    paths["model_dimensions"] = str(path)
    plt.close()

    # 3. Steering Norms Comparison (all models in one plot)
    path = output_dir / "steering_norms_comparison.png"
    fig = plot_steering_norms_comparison(output_dir, output_path=str(path))
    if fig:
        paths["steering_norms_comparison"] = str(path)
        plt.close()

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate steering vector plots")
    parser.add_argument("--output_dir", type=str, default="outputs/phase2_steering_vectors")
    args = parser.parse_args()

    paths = create_all_plots(args.output_dir)

    print("\nGenerated plots:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
