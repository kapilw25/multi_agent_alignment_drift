"""
Model loading utilities for multi-architecture LLM evaluation.
Provides unified interface for loading Base/Instruct model pairs.

TODO: Implement when needed for Phase 3+
"""

from typing import Dict, Tuple, List, Optional, Any

# Placeholder - model configurations dict
MODELS: Dict[str, Dict[str, Any]] = {}


def load_model_for_eval(model_key: str) -> Tuple[Any, Any]:
    """
    Load model and tokenizer for evaluation.

    Args:
        model_key: Key from MODELS dict (e.g., "Mistral_7B")

    Returns:
        Tuple of (model, tokenizer)
    """
    raise NotImplementedError("TODO: Implement load_model_for_eval")


def unload_model(model: Any) -> None:
    """
    Unload model from GPU memory.

    Args:
        model: The model to unload
    """
    raise NotImplementedError("TODO: Implement unload_model")


def cleanup_gpu() -> None:
    """
    Clean up GPU memory (torch.cuda.empty_cache, gc.collect).
    """
    raise NotImplementedError("TODO: Implement cleanup_gpu")


def verify_hf_repos(model_keys: List[str], interactive: bool = True) -> bool:
    """
    Verify HuggingFace repos exist and are accessible.

    Args:
        model_keys: List of model keys to verify
        interactive: Whether to prompt user on failure

    Returns:
        True if all repos accessible, False otherwise
    """
    raise NotImplementedError("TODO: Implement verify_hf_repos")
