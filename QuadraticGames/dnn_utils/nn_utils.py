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
    if cfg.criterion.lower() == "dcpa":
        return DCPALoss()
    raise ValueError(f"Unknown criterion {cfg.criterion}")


class DCPALoss(nn.Module):
    """
    DCPA loss function for quadratic games.
    
    This loss computes the gradient approximation using the loss path Z
    and compares it with the optimal gradient labels y.
    
    Gradient approximation: R_n/x_n - b_n - 0.5*q_nn*x_n
    where:
    - R_n, x_n come from Z (loss path)
    - q_nn, b_n come from model predictions
    """
    def __init__(self):
        super(DCPALoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, Z_path, targets):
        """
        Compute DCPA loss over ALL timesteps of the loss path.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [q_nn, b_n] with shape (batch, 2)
        Z_path : torch.Tensor
            Loss path trajectory [x_n(0), R_n(0), ..., x_n(T-1), R_n(T-1)]
            with shape (batch, 2*T_loss)
        targets : torch.Tensor
            Optimal gradient labels at all timesteps with shape (batch, T_loss)
            
        Returns
        -------
        torch.Tensor
            MSE loss between approximated gradient and target gradient
        """
        # Extract q_nn and b_n from predictions, broadcast for all timesteps
        q_nn = predictions[:, 0:1]  # (batch, 1)
        b_n = predictions[:, 1:2]   # (batch, 1)
        
        # Extract x_n(t) and R_n(t) at ALL timesteps from Z_path
        x_all = Z_path[:, 0::2]  # (batch, T_loss) — all x_n values
        R_all = Z_path[:, 1::2]  # (batch, T_loss) — all R_n values
        
        # Compute gradient approximation at every timestep:
        # R_n(t)/x_n(t) - b_n - 0.5*q_nn*x_n(t)
        eps = 1e-8
        gradient_approx = (R_all / (x_all + eps)) - b_n - 0.5 * q_nn * x_all
        
        # MSE across all timesteps
        loss = self.mse(gradient_approx, targets)
        return loss


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
    use_dcpa: bool = False,
) -> float:
    """Train the model for one epoch and return the mean batch loss."""
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch_data in train_loader:
        if use_dcpa:
            # DCPA format: (X, Z, y)
            inputs, Z_path, targets = batch_data
            inputs = inputs.to(device)
            Z_path = Z_path.to(device)
            targets = targets.to(device)
        else:
            # Standard format: (X, y)
            inputs, targets = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        
        if use_dcpa:
            loss = criterion(outputs, Z_path, targets)
        else:
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
    use_dcpa: bool = False,
) -> float:
    """Evaluate the model for one validation epoch and return the mean loss."""
    model.eval()
    total_loss = 0.0
    total_n = 0

    for batch_data in val_loader:
        if use_dcpa:
            # DCPA format: (X, Z, y)
            inputs, Z_path, targets = batch_data
            inputs = inputs.to(device)
            Z_path = Z_path.to(device)
            targets = targets.to(device)
        else:
            # Standard format: (X, y)
            inputs, targets = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
        outputs = model(inputs)
        
        if use_dcpa:
            loss = criterion(outputs, Z_path, targets)
        else:
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
    use_dcpa: bool = False,
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
            use_dcpa=use_dcpa,
        )
        valid_loss = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_dcpa=use_dcpa,
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
def predict_dataset(
    model: torch.nn.Module, 
    data_loader, 
    device: torch.device,
    use_dcpa: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run batched inference over a dataset and collect predictions and targets.
    
    For DCPA mode, this computes the gradient approximation using the loss path Z,
    not just the raw parameter predictions [q_nn, b_n].
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network.
    data_loader : DataLoader
        Validation or test data loader.
    device : torch.device
        Device to run inference on.
    use_dcpa : bool
        If True, computes gradient approximation from parameters and loss path.
        If False, returns raw model outputs.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (predictions, targets) where:
        - predictions: gradient approximations (if DCPA) or raw outputs (if not DCPA)
        - targets: true gradient labels
    """
    model.eval()
    predictions = []
    targets = []

    for batch_data in data_loader:
        if use_dcpa:
            # DCPA format: (X, Z, y)
            inputs, Z_path, batch_targets = batch_data
            inputs = inputs.to(device)
            Z_path = Z_path.to(device)
            
            # Get model predictions: [q_nn, b_n]
            outputs = model(inputs)
            
            # Compute gradient approximation using the SAME formula as DCPALoss
            q_nn = outputs[:, 0:1]  # (batch, 1)
            b_n = outputs[:, 1:2]   # (batch, 1)
            
            # Extract x_n(t) and R_n(t) at ALL timesteps from Z_path
            x_all = Z_path[:, 0::2]  # (batch, T_loss) — all x_n values
            R_all = Z_path[:, 1::2]  # (batch, T_loss) — all R_n values
            
            # Compute gradient approximation at every timestep:
            # R_n(t)/x_n(t) - b_n - 0.5*q_nn*x_n(t)
            eps = 1e-8
            gradient_approx = (R_all / (x_all + eps)) - b_n - 0.5 * q_nn * x_all
            
            predictions.append(gradient_approx.detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())
        else:
            # Standard format: (X, y) - return raw outputs
            inputs, batch_targets = batch_data
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


@torch.no_grad()
def predict_parameters(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> np.ndarray:
    """
    Run batched inference and return raw parameter predictions [q_nn, b_n].
    
    This is used for parameter scatter plots (q_nn vs true q_nn, b_n vs true b_n).
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network.
    data_loader : DataLoader
        Data loader (train/val/test).
    device : torch.device
        Device to run inference on.
        
    Returns
    -------
    np.ndarray
        Parameter predictions with shape (n_samples, 2) where [:, 0] is q_nn and [:, 1] is b_n.
    """
    model.eval()
    predictions = []

    for batch_data in data_loader:
        # Get inputs (X) from either (X, Z, y) or (X, y) format
        inputs = batch_data[0]
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.detach().cpu().numpy())

    return np.concatenate(predictions, axis=0)


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
