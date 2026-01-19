"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, List, Optional
from .Train_DataStruct import *
from .wireless_NN import Wireless_NN
from pathlib import Path

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    """
    :param model:
    :param cfg:
    :return:
    """
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def build_loss(cfg: TrainConfig) -> nn.Module:
    if cfg.criterion == "mse":
        return nn.MSELoss()
    elif cfg.criterion == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown criterion {cfg.criterion}")


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: SchedulerCfg,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Factory that returns:
      - None if cfg.type == NONE
      - StepLR if cfg.type == STEP
      - CosineAnnealingLR if cfg.type == COSINE
    """
    if cfg is None:  # <-- handle None case
        return None

    if cfg.type == SchedulerType.NONE:
        return None

    if cfg.type == SchedulerType.STEP:
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )

    if cfg.type == SchedulerType.COSINE:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.tmax,
            eta_min=cfg.eta_min,
        )

    raise ValueError(f"Unknown scheduler type: {cfg.type}")



def make_predicted_prior_fn(is_alpha_beta: bool, K: int, T: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    inputs:  (B, T*2K)   where each time block is [P(0..K-1), I(0..K-1)]
    outputs:
      - if is_alpha_beta: (B, 2K) -> [alpha(0..K-1), beta(0..K-1)]
      - else:             (B, K)  -> alpha(0..K-1) and beta implicitly 0 (or not used)

    returns predicted_prior: (B, K, T) to match Y
    """
    if is_alpha_beta:
        def fn(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
            B = inputs.shape[0]
            x = inputs.view(B, T, 2 * K)         # (B, T, 2K)
            P = x[:, :, :K]                      # (B, T, K)
            I = x[:, :, K:]                      # (B, T, K)

            alpha = outputs[:, :K]               # (B, K)
            beta  = outputs[:, K:2*K]            # (B, K)

            pred_bt_k = alpha[:, None, :] * P + beta[:, None, :] * I  # (B, T, K)
            return pred_bt_k.permute(0, 2, 1)    # (B, K, T)
        return fn
    else:
        def fn(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
            B = inputs.shape[0]
            x = inputs.view(B, T, 2 * K)         # (B, T, 2K)
            P = x[:, :, :K]                      # (B, T, K)

            alpha = outputs[:, :K]               # (B, K) (or outputs is exactly (B,K))
            pred_bt_k = alpha[:, None, :] * P    # (B, T, K)
            return pred_bt_k.permute(0, 2, 1)    # (B, K, T)
        return fn


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    predicted_prior_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for inputs_pre, inputs, targets in train_loader:
        inputs_pre = inputs_pre.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs_pre)
        predicted_prior = predicted_prior_fn(outputs, inputs)

        loss = criterion(predicted_prior, targets)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    criterion: torch.nn.Module,
    device: torch.device,
    predicted_prior_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for inputs_pre, inputs, targets in val_loader:
        inputs_pre = inputs_pre.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs_pre)
        predicted_prior = predicted_prior_fn(outputs, inputs)

        loss = criterion(predicted_prior, targets)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    num_epochs: int,
    is_alpha_beta: bool,
    K: int,
    T: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[List[float], List[float]]:
    train_list: List[float] = []
    valid_list: List[float] = []

    predicted_prior_fn = make_predicted_prior_fn(is_alpha_beta, K, T)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, predicted_prior_fn
        )
        val_loss = validate_one_epoch(
            model, val_loader, criterion, device, predicted_prior_fn
        )

        # Step scheduler (if exists)
        if scheduler is not None:
            scheduler.step()

        train_list.append(train_loss)
        valid_list.append(val_loss)

        lr_value = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Learning Rate:{lr_value} - "
            f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}"
        )

    return train_list, valid_list


def get_IO_NN(train_cfg: TrainConfig) -> Tuple[int, int]:
    """
    return input and output size depends on configuration
    :param train_cfg:
    :return:
    """
    input_dim = 2 * train_cfg.K * train_cfg.T
    if train_cfg.isAlphaBeta:
        output_dim = train_cfg.K * 2
    else:
        output_dim = train_cfg.K
    return input_dim, output_dim



@torch.no_grad()
def wireless_NN_predict(L: int, N: int,
    input_path: str,
    *,
    cfg_name: str = "train_config.yaml",
    device: Optional[str] = None,
    inputs_pre: torch.Tensor
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load training config + weights from input_path, run inference over T turns, return numpy on CPU.

    Args:
        input_path: directory containing train_config.yaml and weights file
        P: numpy array containing the exploration input needed to build NN inputs_pre (your choice of shape)
        cfg_name: config filename inside input_path
        device: "cuda", "cpu", or None for auto

    Returns:
        alpha_np: np.ndarray on CPU size (L, N, K)
        beta_np: np.ndarray on CPU (None if is_alpha_beta == False) size (L, N, K)
    """
    cfg_path = input_path + "/" + cfg_name
    train_cfg, _ = load_configs_from_yaml(cfg_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build model + load weights
    input_dim, output_dim = get_IO_NN(train_cfg)
    model = Wireless_NN(input_size=input_dim, output_size=output_dim).to(dev)

    weights_path = Path(input_path + "/model.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.eval()

    # Build model input from P
    inputs_pre = inputs_pre.to(dev)

    # Forward
    outputs = model(inputs_pre)

    # Interpret outputs -> alpha/beta
    # Expectation:
    #   if is_alpha_beta: outputs shape (B, 2K) -> [alpha(0..K-1), beta(0..K-1)]
    #   else:             outputs shape (B, K)  -> alpha(0..K-1)
    K = int(train_cfg.K)

    if train_cfg.isAlphaBeta:
        if outputs.shape[-1] != 2 * K:
            raise ValueError(f"Expected model output dim {2*K}, got {outputs.shape[-1]}")
        alpha = outputs[..., :K].view(L, N, K)
        beta = outputs[..., K:2*K].view(L, N, K)
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        return alpha_np, beta_np
    else:
        if outputs.shape[-1] != K:
            raise ValueError(f"Expected model output dim {K}, got {outputs.shape[-1]}")
        alpha_np = outputs.detach().view(L, N, K)
        alpha_np = alpha_np.cpu().numpy()
        return alpha_np, None