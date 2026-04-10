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


def save_qnn_bn_scatter(
    predictions: np.ndarray,
    true_params: np.ndarray,
    output_dir: str,
    filename: str = "qnn_bn_scatter.png",
    jump: int = 100,
    max_points: Optional[int] = None,
) -> Path:
    """
    Save side-by-side scatter plots of predicted vs true q_nn and b_n.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions with shape (batch, 2) where [:, 0] is q_nn and [:, 1] is b_n.
    true_params : np.ndarray
        True parameters with shape (batch, 2) where [:, 0] is q_nn and [:, 1] is b_n.
    output_dir : str
        Directory to save the figure.
    filename : str
        Output image name.
    jump : int
        Take every jump-th point for downsampling.
    max_points : Optional[int]
        Optional cap on number of points after downsampling.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    # Extract q_nn and b_n
    pred_qnn = predictions[:, 0]
    pred_bn = predictions[:, 1]
    true_qnn = true_params[:, 0]
    true_bn = true_params[:, 1]

    # Downsample
    n = min(len(pred_qnn), len(true_qnn))
    idx = np.arange(0, n, max(1, jump))
    if max_points is not None and len(idx) > max_points:
        idx = idx[:max_points]

    pred_qnn_sample = pred_qnn[idx]
    pred_bn_sample = pred_bn[idx]
    true_qnn_sample = true_qnn[idx]
    true_bn_sample = true_bn[idx]

    # Prepare output path
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot q_nn
    ax1.scatter(true_qnn_sample, pred_qnn_sample, label="Predictions", marker="o", alpha=0.6, s=30)
    ax1.plot([true_qnn_sample.min(), true_qnn_sample.max()], 
             [true_qnn_sample.min(), true_qnn_sample.max()], 
             'r--', label="y=x (Perfect)", linewidth=2)
    ax1.set_xlabel("True $q_{nn}$", fontsize=12)
    ax1.set_ylabel("Predicted $q_{nn}$", fontsize=12)
    ax1.set_title("$q_{nn}$ Predictions vs True", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot b_n
    ax2.scatter(true_bn_sample, pred_bn_sample, label="Predictions", marker="o", alpha=0.6, s=30, color='orange')
    ax2.plot([true_bn_sample.min(), true_bn_sample.max()], 
             [true_bn_sample.min(), true_bn_sample.max()], 
             'r--', label="y=x (Perfect)", linewidth=2)
    ax2.set_xlabel("True $b_n$", fontsize=12)
    ax2.set_ylabel("Predicted $b_n$", fontsize=12)
    ax2.set_title("$b_n$ Predictions vs True", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path
