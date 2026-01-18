# Utils package
import json
from pathlib import Path

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
