"""
Filesystem helpers for quadratic offline training.

This module centralizes dataset loading, checkpoint saving, and config copying
for the quadratic-game training pipeline.
"""

from pathlib import Path
from typing import Tuple
import shutil

import numpy as np
import torch


def _load_npz_shards(dir_path: Path, prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one split from either a single `.npz` file or multiple numbered shards (DCPA format)."""
    single = dir_path / f"{prefix}.npz"
    if single.exists():
        with np.load(single) as data:
            return data["X"], data["Z"], data["y"]

    shards = sorted(dir_path.glob(f"{prefix}_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No {prefix}.npz or {prefix}_*.npz found in: {dir_path}")

    Xs = []
    Zs = []
    Ys = []
    for shard in shards:
        with np.load(shard) as data:
            Xs.append(data["X"])
            Zs.append(data["Z"])
            Ys.append(data["y"])

    return np.concatenate(Xs, axis=0), np.concatenate(Zs, axis=0), np.concatenate(Ys, axis=0)


def load_dataset_npz(
    base_dir: str | Path,
    *,
    prefix: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/validation arrays from the quadratic dataset directory layout (DCPA format).
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, Z_train, y_train, X_valid, Z_valid, y_valid)
    """
    base_dir = Path(base_dir)
    train_dir = base_dir / "train"
    valid_dir = base_dir / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not valid_dir.exists():
        raise FileNotFoundError(f"Valid dir not found: {valid_dir}")

    X_train, Z_train, y_train = _load_npz_shards(train_dir, prefix)
    X_valid, Z_valid, y_valid = _load_npz_shards(valid_dir, prefix)
    return X_train, Z_train, y_train, X_valid, Z_valid, y_valid


def save_model_weights(model: torch.nn.Module, output_dir: str | Path, filename: str) -> Path:
    """Save a PyTorch state dict into the requested results directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename
    torch.save(model.state_dict(), save_path)
    return save_path


def copy_config_file(config_path: str | Path, output_dir: str | Path, filename: str = "train_config.yaml") -> Path:
    """Copy the training YAML into the results directory for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / filename
    shutil.copyfile(config_path, dst)
    return dst
