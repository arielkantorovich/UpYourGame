"""
Training utilities for the quadratic neural network.

This module groups optimizer and scheduler builders, epoch loops, full model
fitting, and inference helpers used by the offline trainer.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .quadratic_nn import Quadratic_NN
from .train_data_struct import (
    CosineSchedulerConfig,
    SchedulerCfg,
    StepSchedulerConfig,
    TrainConfig,
)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    """Create the optimizer requested by the training config."""
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def build_loss(cfg: TrainConfig) -> nn.Module:
    """Create the loss function requested by the training config."""
    if cfg.criterion.lower() == "mse":
        return nn.MSELoss()
    if cfg.criterion.lower() == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unknown criterion {cfg.criterion}")


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: SchedulerCfg,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create the optional learning-rate scheduler from the scheduler config."""
    if cfg is None:
        return None
    if isinstance(cfg, StepSchedulerConfig):
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if isinstance(cfg, CosineSchedulerConfig):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.tmax, eta_min=cfg.eta_min)
    raise ValueError(f"Unknown scheduler config type: {type(cfg).__name__}")


def get_io_dimensions(train_cfg: TrainConfig) -> Tuple[int, int]:
    """Infer network input/output dimensions from the training configuration."""
    input_dim = 2 * train_cfg.T_exploration
    output_dim = train_cfg.output_dim
    return input_dim, output_dim


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> float:
    """Train the model for one epoch and return the mean batch loss."""
    model.train()
    total_loss = 0.0
    total_n = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_n += batch_size

    return total_loss / max(total_n, 1)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model for one validation epoch and return the mean loss."""
    model.eval()
    total_loss = 0.0
    total_n = 0

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_n += batch_size

    return total_loss / max(total_n, 1)


def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    num_epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """Run the full training loop and return train/validation loss histories."""
    train_list: List[float] = []
    valid_list: List[float] = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
        )
        valid_loss = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        train_list.append(train_loss)
        valid_list.append(valid_loss)

        lr_value = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Learning Rate:{lr_value} - "
            f"Train Loss: {train_loss:.5f}, Val Loss: {valid_loss:.5f}"
        )

    return train_list, valid_list


@torch.no_grad()
def predict_dataset(model: torch.nn.Module, data_loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run batched inference over a dataset and collect predictions and targets."""
    model.eval()
    predictions = []
    targets = []

    for inputs, batch_targets in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.detach().cpu().numpy())
        targets.append(batch_targets.detach().cpu().numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


@torch.no_grad()
def quadratic_nn_predict(
    input_path: str,
    *,
    inputs: torch.Tensor,
    cfg,
    device: Optional[str] = None,
) -> np.ndarray:
    """Load saved weights and predict quadratic parameters for one input batch."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    input_dim, output_dim = get_io_dimensions(cfg)
    model = Quadratic_NN(input_size=input_dim, output_size=output_dim).to(dev)

    weights_path = Path(input_path) / "model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.eval()

    outputs = model(inputs.to(dev))
    return outputs.detach().cpu().numpy()
