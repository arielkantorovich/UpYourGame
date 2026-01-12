"""
Created on : ------

@author: Ariel_Kantorovich
"""


from dataclasses import dataclass
from typing import Optional
from enum import Enum
import argparse

@dataclass(slots=True)
class TrainConfig:
    optimizer: str = "adam" # [adam, sgd]
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9          # for SGD
    criterion: str = "mse" # [mse, l1]
    batch_size: int = 128
    epochs: int = 100
    input_dim: int = 5
    output_dim: int = 5
    grad_clip: Optional[float] = None
    scheduler: Optional[str] = None


class SchedulerType(str, Enum):
    NONE = "none"
    STEP = "step"
    COSINE = "cosine"


@dataclass(slots=True)
class SchedulerConfig:
    """
    Scheduler configuration.
    If type == NONE -> no scheduler (factory returns None).
    """
    type: SchedulerType = SchedulerType.NONE

    # StepLR params
    step_size: int = 10
    gamma: float = 0.1

    # CosineAnnealingLR params
    tmax: int = 100
    eta_min: float = 0.0



def scheduler_cfg_from_args(args: argparse.Namespace) -> SchedulerConfig:
    """
    Build SchedulerConfig from argparse args.
    """
    return SchedulerConfig(
        type=SchedulerType(args.scheduler),
        step_size=args.sched_step_size,
        gamma=args.sched_gamma,
        tmax=args.sched_tmax,
        eta_min=args.sched_eta_min,
    )


def add_scheduler_args(p: argparse.ArgumentParser) -> None:
    """
    Add optional scheduler-related CLI args.
    """
    p.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=[e.value for e in SchedulerType],
        help="LR scheduler type (optional). Choices: none|step|cosine",
    )

    # StepLR
    p.add_argument("--sched_step_size", type=int, default=10, help="StepLR: step_size (epochs)")
    p.add_argument("--sched_gamma", type=float, default=0.1, help="StepLR: gamma multiplier")

    # CosineAnnealingLR
    p.add_argument("--sched_tmax", type=int, default=100, help="Cosine: T_max (epochs)")
    p.add_argument("--sched_eta_min", type=float, default=0.0, help="Cosine: eta_min")