"""
Checkpointing utilities for long-running evaluations.
Allows resuming from crashes without losing progress.

    from utils.checkpoint import CheckpointManager

    ckpt = CheckpointManager("phase2", output_dir)
    if ckpt.exists():
        completed = ckpt.load()
    ckpt.save({"model": "Llama3_8B", "status": "done"})
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class CheckpointManager:
    """
    Manages checkpoints for multi-model evaluation runs.

    Usage:
        ckpt = CheckpointManager("phase2_steering", output_dir)

        # Check for existing checkpoint
        if ckpt.exists():
            state = ckpt.load()
            completed_models = state.get("completed_models", [])

        # Save progress after each model
        ckpt.save({
            "completed_models": ["Llama3_8B", "Mistral_7B"],
            "results": {...},
            "current_model": None
        })

        # Mark complete and cleanup
        ckpt.mark_complete()
    """

    def __init__(self, name: str, output_dir: Path, suffix: str = "_checkpoint.json"):
        """
        Initialize checkpoint manager.

        Args:
            name: Checkpoint name (e.g., "phase1_aqi", "phase2_steering")
            output_dir: Directory to store checkpoint file
            suffix: File suffix for checkpoint files
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / f"{name}{suffix}"

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_path.exists()

    def load(self) -> Dict[str, Any]:
        """
        Load checkpoint state.

        Returns:
            Dict with checkpoint state, or empty dict if not found
        """
        if not self.exists():
            return {}

        try:
            with open(self.checkpoint_path, "r") as f:
                state = json.load(f)
            print(f"Loaded checkpoint: {self.checkpoint_path}")
            return state
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return {}

    def save(self, state: Dict[str, Any]) -> None:
        """
        Save checkpoint state.

        Args:
            state: Dict with current state to save
        """
        state["_checkpoint_time"] = datetime.now().isoformat()
        state["_checkpoint_name"] = self.name

        with open(self.checkpoint_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        print(f"Checkpoint saved: {self.checkpoint_path}")

    def update(self, **kwargs) -> None:
        """
        Update existing checkpoint with new values.

        Args:
            **kwargs: Key-value pairs to update
        """
        state = self.load()
        state.update(kwargs)
        self.save(state)

    def add_completed_model(self, model_key: str, result: Dict = None) -> None:
        """
        Add a model to the completed list.

        Args:
            model_key: Model identifier
            result: Optional result dict for this model
        """
        state = self.load()

        completed = state.get("completed_models", [])
        if model_key not in completed:
            completed.append(model_key)

        results = state.get("results", {})
        if result:
            results[model_key] = result

        state["completed_models"] = completed
        state["results"] = results
        state["current_model"] = None

        self.save(state)

    def set_current_model(self, model_key: str) -> None:
        """Mark which model is currently being processed."""
        self.update(current_model=model_key)

    def get_completed_models(self) -> List[str]:
        """Get list of completed model keys."""
        state = self.load()
        return state.get("completed_models", [])

    def get_pending_models(self, all_models: List[str]) -> List[str]:
        """
        Get list of models that haven't been completed yet.

        Args:
            all_models: Full list of model keys to process

        Returns:
            List of model keys not yet completed
        """
        completed = set(self.get_completed_models())
        return [m for m in all_models if m not in completed]

    def is_complete(self) -> bool:
        """Check if checkpoint is marked as complete."""
        state = self.load()
        return state.get("_complete", False)

    def mark_complete(self) -> None:
        """Mark the checkpoint as complete (all models done)."""
        self.update(_complete=True)

    def delete(self) -> None:
        """Delete the checkpoint file."""
        if self.exists():
            self.checkpoint_path.unlink()
            print(f"Deleted checkpoint: {self.checkpoint_path}")

    def get_results(self) -> Dict[str, Any]:
        """Get the results dict from checkpoint."""
        state = self.load()
        return state.get("results", {})


def show_checkpoint_menu(ckpt: CheckpointManager, all_models: List[str]) -> str:
    """
    Show interactive menu for checkpoint handling.

    Args:
        ckpt: CheckpointManager instance
        all_models: Full list of models to process

    Returns:
        "resume" - Continue from checkpoint (run pending models)
        "restart" - Delete checkpoint and start fresh
        "complete" - All requested models already done, use cached
        "skip" - No checkpoint exists
    """
    if not ckpt.exists():
        return "skip"

    state = ckpt.load()
    completed = set(state.get("completed_models", []))
    current = state.get("current_model")
    is_done = state.get("_complete", False)

    # Check which requested models are already done vs need to run
    requested = set(all_models)
    already_done = requested & completed
    need_to_run = requested - completed

    print(f"\n{'=' * 60}")
    print("CHECKPOINT FOUND")
    print(f"{'=' * 60}")
    print(f"Previously completed: {len(completed)}")
    if completed:
        print(f"  {', '.join(sorted(completed))}")
    print(f"Requested models: {len(requested)}")
    print(f"  {', '.join(sorted(requested))}")
    if need_to_run:
        print(f"Need to run: {len(need_to_run)}")
        print(f"  {', '.join(sorted(need_to_run))}")
    if current:
        print(f"In progress: {current}")
    print(f"{'=' * 60}")

    # Case 1: All requested models already completed
    if not need_to_run:
        print("\nAll requested models already in cache.")
        print("[1] Use cached results (regenerate plots only)")
        print("[2] Start fresh (delete checkpoint)")
        choice = input("\nChoice [1/2]: ").strip()
        if choice == "2":
            ckpt.delete()
            return "restart"
        return "complete"

    # Case 2: Some new models to run (preserve existing + add new)
    if already_done:
        print(f"\n{len(already_done)} model(s) cached, {len(need_to_run)} new to run.")
        print("[1] Add new models (preserve existing results)")
        print("[2] Start fresh (delete checkpoint)")
        choice = input("\nChoice [1/2]: ").strip()
        if choice == "2":
            ckpt.delete()
            return "restart"
        # Unmark complete so new models can be added
        if is_done:
            ckpt.update(_complete=False)
        return "resume"

    # Case 3: None of the requested models are done yet
    print("\n[1] Resume from checkpoint")
    print("[2] Start fresh (delete checkpoint)")
    choice = input("\nChoice [1/2]: ").strip()

    if choice == "2":
        ckpt.delete()
        return "restart"

    return "resume"
