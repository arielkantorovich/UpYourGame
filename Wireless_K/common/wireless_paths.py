"""
Created on : ------

@author: Ariel_Kantorovich
"""

from pathlib import Path
import numpy as np
from typing import Tuple

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


def save_split_npz(out_dir: Path, *, X: np.ndarray, y: np.ndarray, prefix: str) -> None:
    """
    Save a split into one compressed file (easy to load later).
    """
    out_path = out_dir / f"{prefix}.npz"
    np.savez_compressed(out_path, X=X, y=y)



def load_dataset_npz(
    base_dir: str | Path,
    *,
    N: int,
    K: int,
    L: int,
    prefix: str = "data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    run_dir = base_dir / f"N{N}_K{K}_L{L}"

    train_path = run_dir / "train" / f"{prefix}.npz"
    valid_path = run_dir / "valid" / f"{prefix}.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Valid file not found: {valid_path}")

    with np.load(train_path) as train_data:
        X_train = train_data["X"]
        y_train = train_data["y"]

    with np.load(valid_path) as valid_data:
        X_valid = valid_data["X"]
        y_valid = valid_data["y"]

    return X_train, y_train, X_valid, y_valid


# ---- Usage example ----
if __name__ == "__main__":
    N, K, L = 5, 14, 200
    train_dir, valid_dir = make_dataset_dirs(r"C:\Users\arielka\PycharmProjects\Thesis_Disrbuted_Algorithms\Wireless_K\Training_Data", N=N, K=K, L=L)

    # fake example data
    X_train = np.random.randn(1000, N, K).astype(np.float32)
    y_train = np.random.randn(1000, N, K).astype(np.float32)

    X_valid = np.random.randn(200, N, K).astype(np.float32)
    y_valid = np.random.randn(200, N, K).astype(np.float32)

    save_split_npz(train_dir, X=X_train, y=y_train, prefix="data")
    save_split_npz(valid_dir, X=X_valid, y=y_valid, prefix="data")

    print("Saved to:")
    print(" train:", train_dir)
    print(" valid:", valid_dir)

    X_tr, y_tr, X_va, y_va = load_dataset_npz(
        r"C:\Users\arielka\PycharmProjects\Thesis_Disrbuted_Algorithms\Wireless_K\Training_Data",
        N=N, K=K, L=L
    )

    print("Train X:", X_tr.shape)
    print("Train y:", y_tr.shape)
    print("Valid X:", X_va.shape)
    print("Valid y:", y_va.shape)
