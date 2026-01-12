"""
Created on : ------

@author: Ariel_Kantorovich
"""

import torch.nn as nn
import torch.optim as optim
from .Train_DataStruct import *

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
    cfg: SchedulerConfig,
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """
    Factory that returns:
      - None if cfg.type == NONE
      - StepLR if cfg.type == STEP
      - CosineAnnealingLR if cfg.type == COSINE
    """
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
