"""
Created on : ------

@author: Ariel_Kantorovich
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from typing import Callable, Optional, Tuple

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



@torch.no_grad()
def save_validation_scatter(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    predicted_prior_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    output_dir: str,
    filename: str = "val_scatter.png",
    jump: int = 600,
    max_points: Optional[int] = None,
) -> Path:
    """
    Runs model on val_loader, collects predictions/targets, and saves a scatter plot.

    X-axis: targets
    Y-axis: predictions
    Also draws y=x reference line.

    :param predicted_prior_fn: function(outputs, inputs) -> predicted_prior
    :param jump: take every jump-th point (simple downsampling)
    :param max_points: optional cap on number of points after downsampling
    """
    model.eval()

    preds = []
    targs = []

    for inputs_pre, inputs, targets in val_loader:
        inputs_pre = inputs_pre.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs_pre)
        predicted_prior = predicted_prior_fn(outputs, inputs)

        preds.append(predicted_prior.detach().cpu().numpy())
        targs.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(preds, axis=0).ravel()
    all_targets = np.concatenate(targs, axis=0).ravel()

    # Downsample for readability/speed
    idx = np.arange(0, len(all_preds), max(1, jump))
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