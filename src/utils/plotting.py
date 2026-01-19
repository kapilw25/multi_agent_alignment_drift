"""
General plotting utilities shared across evaluation phases.
Provides common plot generation and saving functions.

TODO: Implement when needed for unified plotting
"""

from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    plt = None  # type: ignore
    Figure = Any  # type: ignore


def save_figure_dual_format(
    fig: "Figure",
    path: Path,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """
    Save figure in both PDF and PNG formats.

    Args:
        fig: Matplotlib figure to save
        path: Base path (without extension)
        dpi: DPI for PNG output

    Returns:
        Tuple of (pdf_path, png_path)
    """
    raise NotImplementedError("TODO: Implement save_figure_dual_format")


def generate_comparison_plots(
    models: List[str],
    scores: Dict[str, Dict[str, float]],
    output_dir: Path,
    plot_type: str = "bar",
    title: Optional[str] = None,
) -> None:
    """
    Generate comparison plots across models.

    Args:
        models: List of model keys
        scores: Dict mapping model_key -> metric_name -> score
        output_dir: Directory to save plots
        plot_type: Type of plot ("bar", "line", "heatmap")
        title: Optional plot title
    """
    raise NotImplementedError("TODO: Implement generate_comparison_plots")


def get_model_colors(model_keys: List[str]) -> Dict[str, str]:
    """
    Get consistent color mapping for models.

    Args:
        model_keys: List of model keys

    Returns:
        Dict mapping model_key -> hex color string
    """
    raise NotImplementedError("TODO: Implement get_model_colors")
