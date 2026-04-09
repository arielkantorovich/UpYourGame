"""
Configuration and dataset structures for quadratic offline training.

This module defines the CLI parser, YAML-backed training dataclasses, and the
PyTorch dataset wrapper used by the offline trainer.
"""

from dataclasses import dataclass, fields
from enum import Enum
import argparse
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import yaml


class SchedulerType(str, Enum):
    NONE = "none"
    STEP = "step"
    COSINE = "cosine"


@dataclass(slots=True)
class TrainConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    criterion: str = "mse"
    batch_size: int = 128
    epochs: int = 100
    T_exploration: int = 200
    T_loss: int = 200
    output_dim: int = 2
    grad_clip: Optional[float] = None
    scheduler: SchedulerType = SchedulerType.NONE


@dataclass(slots=True)
class StepSchedulerConfig:
    step_size: int = 10
    gamma: float = 0.1


@dataclass(slots=True)
class CosineSchedulerConfig:
    tmax: int = 100
    eta_min: float = 0.0


SchedulerCfg = Union[None, StepSchedulerConfig, CosineSchedulerConfig]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the quadratic offline trainer."""
    parser = argparse.ArgumentParser("quadratic train")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config file")
    parser.add_argument("--input_dir", required=True, type=str, help="Input dataset directory")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for checkpoints and logs")
    return parser


def _filter_known_keys(dc_cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop YAML keys that are not defined on the target dataclass."""
    allowed = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in (data or {}).items() if k in allowed}


def load_configs_from_yaml(cfg_path: str) -> Tuple[TrainConfig, SchedulerCfg]:
    """Load training and scheduler configuration blocks from a YAML file."""
    with open(cfg_path, "r") as handle:
        raw = yaml.safe_load(handle) or {}

    train_raw = raw.get("train", {}) or {}
    if "scheduler" in train_raw:
        train_raw = dict(train_raw)
        train_raw["scheduler"] = SchedulerType(str(train_raw["scheduler"]).lower())

    train_cfg = TrainConfig(**_filter_known_keys(TrainConfig, train_raw))

    if train_cfg.scheduler == SchedulerType.NONE:
        sched_cfg: SchedulerCfg = None
    elif train_cfg.scheduler == SchedulerType.STEP:
        step_raw = raw.get("scheduler_step", {}) or {}
        sched_cfg = StepSchedulerConfig(**_filter_known_keys(StepSchedulerConfig, step_raw))
    elif train_cfg.scheduler == SchedulerType.COSINE:
        cosine_raw = raw.get("scheduler_cosine", {}) or {}
        sched_cfg = CosineSchedulerConfig(**_filter_known_keys(CosineSchedulerConfig, cosine_raw))
    else:
        raise ValueError(f"Unsupported scheduler type: {train_cfg.scheduler}")

    return train_cfg, sched_cfg


class XYDataset(Dataset):
    def __init__(self, X, y, dtype=torch.float32):
        """Wrap paired numpy arrays as a PyTorch dataset."""
        assert len(X) == len(y)
        self.X = torch.as_tensor(X, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return one `(features, target)` sample."""
        return self.X[idx], self.y[idx]


class XZYDataset(Dataset):
    def __init__(self, X, Z, y, dtype=torch.float32):
        """
        Wrap X, Z, y numpy arrays as a PyTorch dataset for DCPA approach.
        
        Parameters
        ----------
        X : np.ndarray
            Exploration features
        Z : np.ndarray
            Loss path trajectory data
        y : np.ndarray
            Optimal gradient labels
        """
        assert len(X) == len(Z) == len(y)
        self.X = torch.as_tensor(X, dtype=dtype)
        self.Z = torch.as_tensor(Z, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return one `(X_features, Z_path, target)` sample."""
        return self.X[idx], self.Z[idx], self.y[idx]
