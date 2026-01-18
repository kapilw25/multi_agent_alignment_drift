"""
Caching utilities for tensors and embeddings.
Avoids recomputation on re-runs.

    from utils.cache import TensorCache

    cache = TensorCache("hidden_states", output_dir / "Llama3_8B")

    # Check and load if exists
    if cache.exists("h_base_chosen"):
        tensor = cache.load("h_base_chosen")
    else:
        tensor = compute_expensive_tensor(...)
        cache.save("h_base_chosen", tensor)
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

import torch


class TensorCache:
    """
    Cache manager for PyTorch tensors.

    Supports saving/loading individual tensors or batches of tensors.
    Uses .pt format for tensors and .pkl for metadata.

    Usage:
        cache = TensorCache("steering", output_dir / "Llama3_8B")

        # Save tensors
        cache.save("h_base_chosen", tensor)
        cache.save_batch({
            "h_base_chosen": t1,
            "h_base_rejected": t2,
            "h_instruct_chosen": t3,
            "h_instruct_rejected": t4,
        })

        # Load tensors
        if cache.exists("h_base_chosen"):
            tensor = cache.load("h_base_chosen")

        # Load all cached tensors
        all_tensors = cache.load_all()
    """

    def __init__(self, name: str, cache_dir: Path):
        """
        Initialize tensor cache.

        Args:
            name: Cache name prefix (e.g., "hidden_states", "steering")
            cache_dir: Directory to store cache files
        """
        self.name = name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / f"{name}_cache_meta.json"

    def _get_path(self, key: str) -> Path:
        """Get path for a tensor key."""
        return self.cache_dir / f"{self.name}_{key}.pt"

    def exists(self, key: str) -> bool:
        """Check if a cached tensor exists."""
        return self._get_path(key).exists()

    def exists_all(self, keys: List[str]) -> bool:
        """Check if all specified keys exist in cache."""
        return all(self.exists(k) for k in keys)

    def save(self, key: str, tensor: torch.Tensor, metadata: Dict = None) -> Path:
        """
        Save a tensor to cache.

        Args:
            key: Identifier for this tensor
            tensor: PyTorch tensor to save
            metadata: Optional metadata dict

        Returns:
            Path to saved file
        """
        path = self._get_path(key)
        torch.save(tensor, path)
        print(f"  Cached: {path.name} {list(tensor.shape)}")
        return path

    def load(self, key: str, device: str = "cpu") -> torch.Tensor:
        """
        Load a tensor from cache.

        Args:
            key: Identifier for the tensor
            device: Device to load tensor to

        Returns:
            Loaded tensor
        """
        path = self._get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Cache not found: {path}")

        tensor = torch.load(path, map_location=device)
        print(f"  Loaded from cache: {path.name} {list(tensor.shape)}")
        return tensor

    def save_batch(self, tensors: Dict[str, torch.Tensor]) -> None:
        """
        Save multiple tensors at once.

        Args:
            tensors: Dict mapping key -> tensor
        """
        for key, tensor in tensors.items():
            self.save(key, tensor)

    def load_batch(self, keys: List[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Load multiple tensors at once.

        Args:
            keys: List of tensor keys to load
            device: Device to load tensors to

        Returns:
            Dict mapping key -> tensor
        """
        return {key: self.load(key, device) for key in keys}

    def load_all(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Load all cached tensors.

        Args:
            device: Device to load tensors to

        Returns:
            Dict mapping key -> tensor
        """
        tensors = {}
        for path in self.cache_dir.glob(f"{self.name}_*.pt"):
            # Extract key from filename
            key = path.stem.replace(f"{self.name}_", "")
            tensors[key] = torch.load(path, map_location=device)
        return tensors

    def delete(self, key: str) -> None:
        """Delete a cached tensor."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            print(f"  Deleted cache: {path.name}")

    def clear(self) -> None:
        """Delete all cached tensors for this cache name."""
        for path in self.cache_dir.glob(f"{self.name}_*.pt"):
            path.unlink()
            print(f"  Deleted: {path.name}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached tensors."""
        info = {
            "cache_dir": str(self.cache_dir),
            "name": self.name,
            "tensors": {}
        }

        for path in self.cache_dir.glob(f"{self.name}_*.pt"):
            key = path.stem.replace(f"{self.name}_", "")
            stat = path.stat()
            info["tensors"][key] = {
                "path": str(path),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

        return info


class EmbeddingCache:
    """
    Cache manager for embeddings (DataFrames with embedding columns).

    Uses pickle format for compatibility with pandas DataFrames.

    Usage:
        cache = EmbeddingCache(output_dir / "Llama3_8B")

        if cache.exists():
            df = cache.load()
        else:
            df = compute_embeddings(...)
            cache.save(df)
    """

    def __init__(self, cache_dir: Path, filename: str = "embeddings.pkl"):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cache file
            filename: Name of the cache file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / filename

    def exists(self) -> bool:
        """Check if cache exists."""
        return self.cache_path.exists()

    def save(self, data: Any) -> Path:
        """
        Save data to cache.

        Args:
            data: Data to cache (DataFrame, dict, etc.)

        Returns:
            Path to saved file
        """
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Cached: {self.cache_path}")
        return self.cache_path

    def load(self) -> Any:
        """
        Load data from cache.

        Returns:
            Cached data
        """
        if not self.exists():
            raise FileNotFoundError(f"Cache not found: {self.cache_path}")

        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded from cache: {self.cache_path}")
        return data

    def delete(self) -> None:
        """Delete the cache file."""
        if self.exists():
            self.cache_path.unlink()
            print(f"  Deleted cache: {self.cache_path}")


def show_cache_menu(cache_dir: Path, cache_type: str = "tensors") -> str:
    """
    Show interactive menu for cache handling.

    Args:
        cache_dir: Directory containing cache files
        cache_type: Type of cache ("tensors" or "embeddings")

    Returns:
        "use" - Use existing cache
        "recompute" - Delete cache and recompute
        "skip" - No cache exists
    """
    cache_dir = Path(cache_dir)

    if cache_type == "tensors":
        cache_files = list(cache_dir.glob("*_*.pt"))
    else:
        cache_files = list(cache_dir.glob("*.pkl"))

    if not cache_files:
        return "skip"

    total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"CACHE FOUND ({cache_type})")
    print(f"{'=' * 60}")
    print(f"Location: {cache_dir}")
    print(f"Files: {len(cache_files)}")
    print(f"Total size: {total_size:.1f} MB")
    print(f"{'=' * 60}")

    print("\n[1] Use cached data")
    print("[2] Recompute (delete cache)")
    choice = input("\nChoice [1/2]: ").strip()

    if choice == "2":
        for f in cache_files:
            f.unlink()
        print(f"Deleted {len(cache_files)} cache files")
        return "recompute"

    return "use"
