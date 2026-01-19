# Utils package
"""
Utility modules for multi-agent alignment evaluation.

Available modules:
    - checkpoint: CheckpointManager for resumable runs
    - cache: TensorCache, EmbeddingCache for caching
    - plot_aqi: AQI score bar plots
    - plot_steering: Steering vector plots
    - model_loader: Model loading utilities (TODO)
    - generation: Batch text generation (TODO)
    - cli_menus: Interactive CLI menus (TODO)
    - response_validator: Output validation (TODO)
    - plotting: General plotting utils (TODO)
    - dataset_info: Dataset statistics (TODO)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"
_MODEL_REGISTRY = None


def load_model_registry():
    """Load model registry from JSON file."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        with open(_REGISTRY_PATH) as f:
            data = json.load(f)
        _MODEL_REGISTRY = data["models"]
    return _MODEL_REGISTRY


def get_model_info(model_key: str) -> dict:
    """Get model info by key."""
    registry = load_model_registry()
    if model_key not in registry:
        raise KeyError(f"Model '{model_key}' not in registry. Available: {list(registry.keys())}")
    return registry[model_key]


def get_all_model_keys() -> list:
    """Get all available model keys."""
    return list(load_model_registry().keys())


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_training_logger(run_name: str, project_root: Path):
    """
    Setup logger that writes to both console and file.

    Args:
        run_name: Name for the log file
        project_root: Project root directory

    Returns:
        (log_file, log_filename, original_stdout, original_stderr)
    """
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{run_name}_{timestamp}.log"

    log_file = open(log_filename, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Tee class to write to both console and file
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print(f"Logging to: {log_filename}")
    return log_file, log_filename, original_stdout, original_stderr


def restore_logging(log_file, original_stdout, original_stderr):
    """Restore original stdout/stderr after logging."""
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    if log_file:
        log_file.close()


def filter_model_keys(requested_keys, models_dict, default_keys):
    """
    Filter model keys based on user request.

    Args:
        requested_keys: User-specified model keys (or None for all)
        models_dict: Dict of available models
        default_keys: Default list of model keys

    Returns:
        List of valid model keys
    """
    if requested_keys:
        return [k for k in requested_keys if k in models_dict]
    return [k for k in default_keys if k in models_dict]
