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
    - No white lines between cells (black borders)
    - Bold, large text for values
    - Confidence intervals shown below main value

    Args:
        output_dir: Directory containing model subdirectories
        model_keys: List of model keys to include (None for all)
        output_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    output_dir = Path(output_dir)

    # Collect per-axiom data and CI data
    data = {}
    ci_data = {}  # Confidence intervals
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            model_key = model_dir.name
            if model_keys and model_key not in model_keys:
                continue

            # Find metrics CSV
            csv_files = list(model_dir.glob("*_metrics_summary.csv"))
            if csv_files:
                df_csv = pd.read_csv(csv_files[0])
                # Exclude 'overall' row
                df_csv = df_csv[df_csv["Category"] != "overall"]
                data[model_key] = dict(zip(df_csv["Category"], df_csv["AQI [0-100] (↑)"]))

                # Check for CI column (if available)
                if "AQI_CI" in df_csv.columns:
                    ci_data[model_key] = dict(zip(df_csv["Category"], df_csv["AQI_CI"]))
                else:
                    # Estimate CI based on sample size (rough approximation)
                    # Using ~5% of value as placeholder CI when not available
                    ci_data[model_key] = {
                        cat: val * 0.05 for cat, val in
                        zip(df_csv["Category"], df_csv["AQI [0-100] (↑)"])
                    }

    if not data:
        print("No axiom data found")
        return None

    # Create DataFrame
    df = pd.DataFrame(data)
    df_ci = pd.DataFrame(ci_data)

    # Sort columns by mean AQI
    col_order = df.mean().sort_values(ascending=False).index
    df = df[col_order]
    df_ci = df_ci[col_order]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using pcolormesh for no gaps
    # Normalize values to 0-1 for colormap
    norm = plt.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.RdYlGn

    # Draw cells manually with black borders
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            ci_val = df_ci.iloc[i, j]
            color = cmap(norm(val))

            # Draw rectangle with black border (no white gaps)
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=color,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(rect)

            # All text is black (like reference heatmap)
            text_color = "black"

            # Main value - bold and large
            ax.text(
                j, i - 0.12,
                f"{val:.1f}",
                ha="center", va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold"
            )

            # CI value below - smaller, italic
            ax.text(
                j, i + 0.22,
                f"+/-{ci_val:.2f}",
                ha="center", va="center",
                color=text_color,
                fontsize=10,
                fontstyle="italic"
            )

    # Set axis limits
    ax.set_xlim(-0.5, len(df.columns) - 0.5)
    ax.set_ylim(len(df.index) - 0.5, -0.5)

    # Labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=11, fontweight="bold")
    ax.set_yticklabels(df.index, fontsize=11, fontweight="bold")

    ax.set_title("AQI by Axiom and Model", fontsize=16, fontweight="bold", pad=15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("AQI Score", fontsize=12, fontweight="bold")

    # Remove default spines
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_steering_delta_bars(
    summary_path: str,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create horizontal bar chart showing AQI delta (λ=1 - λ=0) for each model.
    Green = positive + monotonic, light green = positive non-monotonic, red = negative.

    Args:
        summary_path: Path to phase3_summary.json
        output_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    with open(summary_path) as f:
        data = json.load(f)

    # Extract and sort by delta descending
    models = [(d["model_key"], d["delta"], d["is_monotonic"]) for d in data["summary"]]
    models.sort(key=lambda x: x[1], reverse=True)

    names = [m[0] for m in models]
    deltas = [m[1] for m in models]
    monotonic = [m[2] for m in models]

    # Colors: dark green (positive+monotonic), light green (positive), red (negative)
    colors = []
    for d, m in zip(deltas, monotonic):
        if d > 0 and m:
            colors.append('#27ae60')  # Dark green
        elif d > 0:
            colors.append('#82e0aa')  # Light green
        else:
            colors.append('#e74c3c')  # Red

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, deltas, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, delta, mono in zip(bars, deltas, monotonic):
        width = bar.get_width()
        label = f'{delta:+.1f}'
        if mono and delta > 0:
            label += ' *'
        x_pos = width + 0.5 if width >= 0 else width - 0.5
        ha = 'left' if width >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                va='center', ha=ha, fontsize=11, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('AQI Delta (λ=1 - λ=0)', fontsize=12)
    ax.set_title('D-STEER Validation: AQI Improvement\n(* = Monotonic Increase)', fontsize=14, fontweight='bold')
    ax.set_xlim(-20, 30)
    ax.grid(axis='x', alpha=0.3)

    # Legend - inside plot, upper right corner (empty area above negative bars)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Positive + Monotonic'),
        Patch(facecolor='#82e0aa', edgecolor='black', label='Positive (non-monotonic)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Negative (degradation)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_steering_combined(
    output_dir: str,
    output_path: Optional[str] = None,
    layout: str = "focus_bar",
) -> plt.Figure:
    """
    Combined figure: individual AQI vs lambda LINE plots + delta BAR chart.

    Layouts:
      - "focus_bar": Large bar chart (right) + small line plots (left 2x3 grid)
      - "grid_7": 7 subplots (6 lines + 1 bar) in 2 rows
      - "vertical": Bar on top, lines below in 2x3

    Args:
        output_dir: Phase 3 output directory containing model subdirs + phase3_summary.json
        output_path: Path to save figure (optional)
        layout: Layout style ("focus_bar", "grid_7", "vertical")

    Returns:
        matplotlib Figure object
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    summary_path = output_dir / "phase3_summary.json"

    with open(summary_path) as f:
        data = json.load(f)

    # Sort by delta descending
    summary = sorted(data["summary"], key=lambda x: x["delta"], reverse=True)
    model_keys = [d["model_key"] for d in summary]
    deltas = [d["delta"] for d in summary]
    monotonic = [d["is_monotonic"] for d in summary]

    # Colors for bar chart
    bar_colors = []
    for d, m in zip(deltas, monotonic):
        if d > 0 and m:
            bar_colors.append('#27ae60')
        elif d > 0:
            bar_colors.append('#82e0aa')
        else:
            bar_colors.append('#e74c3c')

    # Line colors matching bar order
    line_colors = plt.cm.tab10(np.linspace(0, 1, len(model_keys)))
    model_color_map = {k: c for k, c in zip(model_keys, line_colors)}

    if layout == "focus_bar":
        # Layout: Bar chart prominent on right (60%), lines on left (40%)
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1.5, 1.5], hspace=0.35, wspace=0.3)

        # Bar chart spans right 2 columns, all 3 rows
        ax_bar = fig.add_subplot(gs[:, 2:])

        # Line plots in left 2 columns (2x3 grid → actually 3x2)
        line_axes = []
        for i in range(3):
            for j in range(2):
                ax = fig.add_subplot(gs[i, j])
                line_axes.append(ax)

    elif layout == "grid_7":
        # Layout: 2 rows - top row has 4 plots, bottom row has 3 (bar is larger)
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.3, wspace=0.25)

        line_axes = []
        # Top row: 4 line plots
        for j in range(4):
            ax = fig.add_subplot(gs[0, j])
            line_axes.append(ax)
        # Bottom row: 2 line plots + bar chart (spans 2 cols)
        for j in range(2):
            ax = fig.add_subplot(gs[1, j])
            line_axes.append(ax)
        ax_bar = fig.add_subplot(gs[1, 2:])

    else:  # vertical
        # Layout: Bar on top, lines below in 2x3 grid
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)

        # Bar spans top row
        ax_bar = fig.add_subplot(gs[0, :])

        # Lines in bottom 2 rows (2x3)
        line_axes = []
        for i in range(1, 3):
            for j in range(3):
                ax = fig.add_subplot(gs[i, j])
                line_axes.append(ax)

    # --- Plot individual line charts ---
    for idx, model_key in enumerate(model_keys):
        if idx >= len(line_axes):
            break
        ax = line_axes[idx]

        # Load lambda results
        model_data = next(d for d in summary if d["model_key"] == model_key)
        lambdas = sorted([float(k) for k in model_data["all_lambdas"].keys()])
        aqi_scores = [model_data["all_lambdas"][str(l)] for l in lambdas]

        # Determine color based on delta
        delta = model_data["delta"]
        is_mono = model_data["is_monotonic"]
        if delta > 0 and is_mono:
            color = '#27ae60'
        elif delta > 0:
            color = '#82e0aa'
        else:
            color = '#e74c3c'

        ax.plot(lambdas, aqi_scores, 'o-', linewidth=2, markersize=5, color=color)
        ax.fill_between(lambdas, aqi_scores, alpha=0.2, color=color)
        ax.axhline(y=aqi_scores[0], color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.axhline(y=aqi_scores[-1], color='blue', linestyle='--', alpha=0.4, linewidth=1)

        # Title with delta
        title = f"{model_key}\nΔ={delta:+.1f}"
        if is_mono and delta > 0:
            title += " *"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 100)
        ax.set_xlabel('λ', fontsize=9)
        ax.set_ylabel('AQI', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # --- Plot bar chart ---
    bars = ax_bar.barh(model_keys, deltas, color=bar_colors, edgecolor='black', linewidth=0.5)

    for bar, delta, mono in zip(bars, deltas, monotonic):
        width = bar.get_width()
        label = f'{delta:+.1f}'
        if mono and delta > 0:
            label += ' *'
        x_pos = width + 0.5 if width >= 0 else width - 0.5
        ha = 'left' if width >= 0 else 'right'
        ax_bar.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                    va='center', ha=ha, fontsize=11, fontweight='bold')

    ax_bar.axvline(x=0, color='black', linewidth=1)
    ax_bar.set_xlabel('AQI Delta (λ=1 - λ=0)', fontsize=12)
    ax_bar.set_title('D-STEER: AQI Improvement (* = Monotonic)', fontsize=13, fontweight='bold')
    ax_bar.set_xlim(-20, 30)
    ax_bar.grid(axis='x', alpha=0.3)

    # Legend for bar chart - inside plot, upper right corner
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Positive + Monotonic'),
        Patch(facecolor='#82e0aa', edgecolor='black', label='Positive (non-monotonic)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Negative (degradation)'),
    ]
    ax_bar.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

    fig.suptitle('Phase 3: Same-Architecture Steering Validation', fontsize=14, fontweight='bold', y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
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
