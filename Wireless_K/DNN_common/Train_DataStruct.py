"""
Created on : ------

@author: Ariel_Kantorovich
"""


from dataclasses import dataclass
from enum import Enum
import argparse
from dataclasses import fields
from typing import Any, Dict, Tuple, Optional, Union
import yaml
import torch
from torch.utils.data import Dataset, DataLoader

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
    K: int = 5
    T: int = 100
    isAlphaBeta: bool = True
    grad_clip: Optional[float] = None

    # scheduler selection lives here
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
    p = argparse.ArgumentParser("wireless k train")
    p.add_argument("--config", required=True, type=str, help="Path to YAML config file")
    p.add_argument("--input_dir", required=True, type=str, help="Input dataset directory")
    p.add_argument("--output_dir", required=True, type=str, help="Output directory (checkpoints/logs)")
    return p



def _filter_known_keys(dc_cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys that exist in the dataclass (ignore unknown YAML keys)."""
    allowed = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in (data or {}).items() if k in allowed}



def load_configs_from_yaml(cfg_path: str) -> Tuple[TrainConfig, SchedulerCfg]:
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # ---- Train ----
    train_raw = raw.get("train", {}) or {}
    if "scheduler" in train_raw:
        train_raw = dict(train_raw)
        train_raw["scheduler"] = SchedulerType(str(train_raw["scheduler"]).lower())

    train_cfg = TrainConfig(**_filter_known_keys(TrainConfig, train_raw))

    # ---- Scheduler (separate sections) ----
    if train_cfg.scheduler == SchedulerType.NONE:
        sched_cfg: SchedulerCfg = None

    elif train_cfg.scheduler == SchedulerType.STEP:
        step_raw = raw.get("scheduler_step", {}) or {}
        sched_cfg = StepSchedulerConfig(**_filter_known_keys(StepSchedulerConfig, step_raw))

    elif train_cfg.scheduler == SchedulerType.COSINE:
        cos_raw = raw.get("scheduler_cosine", {}) or {}
        sched_cfg = CosineSchedulerConfig(**_filter_known_keys(CosineSchedulerConfig, cos_raw))

    else:
        raise ValueError(f"Unsupported scheduler type: {train_cfg.scheduler}")

    return train_cfg, sched_cfg


class XZYDataset(Dataset):
    def __init__(self, X, Z, y, device=None, dtype=torch.float32):
        assert len(X) == len(Z) == len(y)

        self.X = torch.as_tensor(X, dtype=dtype)
        self.Z = torch.as_tensor(Z, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)

        self.device = device  # optional

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        z = self.Z[idx]
        y = self.y[idx]

        # NOTE: do NOT move to device here (best practice)
        return x, z, y




def main():
    args = build_parser().parse_args()

    train_cfg, sched_cfg = load_configs_from_yaml(args.config)

    # Example: print what you loaded
    print("input_dir :", args.input_dir)
    print("output_dir:", args.output_dir)
    print("train_cfg :", train_cfg)
    print("sched_cfg :", sched_cfg)

    # Here you call your training function, e.g.:
    # train(train_cfg=train_cfg, sched_cfg=sched_cfg,
    #       input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
