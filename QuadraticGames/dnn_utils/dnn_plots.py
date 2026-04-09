"""
Plotting helpers for quadratic offline training outputs.

This module saves the learning-curve figure and the validation scatter plot
produced after the offline training run.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

def plot_loss(
    output_dir: str,
    num_epochs: int,
    train_list: list,
    valid_list: list,
    filename: str = "loss_curve.jpg",
):
    """
    Save training and validation loss to an image.

    :param output_dir: directory to save the figure
    :param num_epochs: number of epochs
    :param train_list: training loss per epoch
    :param valid_list: validation loss per epoch
    :param filename: output image name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_list, label="train")
    plt.plot(epochs, valid_list, label="valid")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path



def save_validation_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    filename: str = "val_scatter.png",
    jump: int = 600,
    max_points: Optional[int] = None,
) -> Path:
    """
    Save a scatter plot of flattened predictions against targets.

    X-axis: targets
    Y-axis: predictions
    Also draws y=x reference line.

    :param jump: take every jump-th point (simple downsampling)
    :param max_points: optional cap on number of points after downsampling
    """
    all_preds = predictions.ravel()
    all_targets = targets.ravel()

    # Downsample for readability/speed – use the shorter array length
    n = min(len(all_preds), len(all_targets))
    all_preds = all_preds[:n]
    all_targets = all_targets[:n]
    idx = np.arange(0, n, max(1, jump))
    if max_points is not None and len(idx) > max_points:
        idx = idx[:max_points]

    xs = all_targets[idx]
    ys = all_preds[idx]

    # Prepare output path
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, label="Predictions", marker="o", alpha=0.6)
    plt.scatter(xs, xs, label="y=x (Target)", marker="x", alpha=0.6)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Validation: Predictions vs Targets")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path
