"""
Interactive CLI menu utilities for evaluation scripts.
Provides user-friendly menus for mode selection, checkpoint handling, etc.

TODO: Implement when needed for interactive runs
"""

from typing import Tuple, Optional
from pathlib import Path


def show_cached_data_menu(
    checkpoint_dir: Path,
    results_dir: Path,
    eval_type: str,
    model_key: Optional[str] = None,
) -> str:
    """
    Show menu for handling existing cached/checkpoint data.

    Args:
        checkpoint_dir: Directory containing checkpoints
        results_dir: Directory containing results
        eval_type: Type of evaluation (e.g., "aqi", "steering")
        model_key: Optional specific model key

    Returns:
        User selection: "resume", "fresh", or "skip"
    """
    raise NotImplementedError("TODO: Implement show_cached_data_menu")


def show_mode_selection_menu(
    eval_name: str,
    sanity_desc: str,
    full_desc: str,
    max_desc: str,
) -> Tuple[str, int]:
    """
    Show menu for selecting evaluation mode and sample count.

    Args:
        eval_name: Name of the evaluation
        sanity_desc: Description for sanity mode
        full_desc: Description for full mode
        max_desc: Description for max mode

    Returns:
        Tuple of (mode, samples) where mode is "sanity"/"full"/"max"
    """
    raise NotImplementedError("TODO: Implement show_mode_selection_menu")


def show_checkpoint_resume_menu(
    model_key: str,
    n_responses: int,
    eval_type: str,
) -> str:
    """
    Show menu for resuming from checkpoint.

    Args:
        model_key: The model key with checkpoint
        n_responses: Number of responses already generated
        eval_type: Type of evaluation

    Returns:
        User selection: "resume", "restart", or "skip"
    """
    raise NotImplementedError("TODO: Implement show_checkpoint_resume_menu")
