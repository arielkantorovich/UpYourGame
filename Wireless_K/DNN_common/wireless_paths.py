"""
Created on : ------

@author: Ariel_Kantorovich
"""

from pathlib import Path
import numpy as np
from typing import Tuple
import torch

def make_dataset_dirs(base_dir: str | Path, *, N: int, K: int) -> tuple[Path, Path]:
    """
    Create:
      base_dir / f"N{N}_K{K}_L{L}" / train
      base_dir / f"N{N}_K{K}_L{L}" / valid
    Returns (train_dir, valid_dir)
    """
    base_dir = Path(base_dir)
    run_dir = base_dir / f"N{N}_K{K}"
    train_dir = run_dir / "train"
    valid_dir = run_dir / "valid"

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    return train_dir, valid_dir


def save_split_npz(out_dir: Path, *, X: np.ndarray, z: np.ndarray, y: np.ndarray, prefix: str) -> None:
    """
    Save a split into one compressed file (easy to load later).
    """
    out_path = out_dir / f"{prefix}.npz"
    np.savez_compressed(out_path, X=X, z=z, y=y)



def load_dataset_npz(
    base_dir: str | Path,
    *,
    prefix: str = "data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train and validation datasets saved as NPZ files.

    Expected structure:
      base_dir / f"N{N}_K{K}_L{L}" / train / {prefix}.npz
      base_dir / f"N{N}_K{K}_L{L}" / valid / {prefix}.npz

    Returns
    -------
    X_train, y_train, X_valid, y_valid
    """

    base_dir = Path(base_dir)

    train_path = base_dir / "train" / f"{prefix}.npz"
    valid_path = base_dir / "valid" / f"{prefix}.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Valid file not found: {valid_path}")

    with np.load(train_path) as train_data:
        X_train = train_data["X"]
        Z_train = train_data["z"]
        y_train = train_data["y"]

    with np.load(valid_path) as valid_data:
        X_valid = valid_data["X"]
        Z_valid = valid_data["z"]
        y_valid = valid_data["y"]

    return X_train, Z_train, y_train, X_valid, Z_valid, y_valid

def save_model_weights(
    model: torch.nn.Module,
    output_dir: str,
    filename: str,
) -> Path:
    """
    Save model weights to output_dir/filename.

    Returns the full path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename

    torch.save(model.state_dict(), save_path)

    return save_path

def load_model_weights(
    model: torch.nn.Module,
    weights_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load model weights into an existing model instance.
    """
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

# ---- Usage example ----
if __name__ == "__main__":
    N, K, L = 5, 14, 200
    train_dir, valid_dir = make_dataset_dirs(r"/Wireless_K/Training_Data", N=N, K=K, L=L)

    # fake example data
    X_train = np.random.randn(1000, N, K).astype(np.float32)
    Z_train = np.random.randn(1000, N, K).astype(np.float32)
    y_train = np.random.randn(1000, N, K).astype(np.float32)

    X_valid = np.random.randn(200, N, K).astype(np.float32)
    Z_valid = np.random.randn(200, N, K).astype(np.float32)
    y_valid = np.random.randn(200, N, K).astype(np.float32)

    save_split_npz(train_dir, X=X_train, z=Z_train, y=y_train, prefix="data")
    save_split_npz(valid_dir, X=X_valid, z=Z_valid, y=y_valid, prefix="data")

    print("Saved to:")
    print(" train:", train_dir)
    print(" valid:", valid_dir)

    X_tr, Z_tr, y_tr, X_va, Z_va, y_va = load_dataset_npz(
        r"/Wireless_K/Training_Data",
        N=N, K=K, L=L
    )

    print("Train X:", X_tr.shape)
    print("Train y:", y_tr.shape)
    print("Valid X:", X_va.shape)
    print("Valid y:", y_va.shape)
