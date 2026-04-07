"""
Dataset builder for quadratic-game offline training.

This script generates quadratic games, runs the exploration phase used for
parameter estimation, and saves per-player supervised samples as compressed
`.npz` shards for the offline neural-network trainer.

Each input sample is built as:
    [x_n(0), cost_n(0), x_n(1), cost_n(1), ..., x_n(T-1), cost_n(T-1)]

Saved arrays:
- `X`: exploration features for one player
- `y`: ground-truth labels `[q_nn_true, b_n_true]`

Output layout:
    Training_Data/N{N}/train
    Training_Data/N{N}/valid

Usage examples
--------------
python QuadraticGames/build_data_to_train.py --N 5 --L 4000 --valid_L 800 --T_exploration 200
python QuadraticGames/build_data_to_train.py --base_dir QuadraticGames/Training_Data --N 20 --L 5000 --valid_L 1000 --L_batch 500
"""

import argparse
from pathlib import Path

import numpy as np

from utils.quad_utils import estimate_game_parameters, generate_Q_B


def make_dataset_dirs(base_dir: str | Path, *, N: int) -> tuple[Path, Path]:
    """
    Create the quadratic supervised dataset layout:

    ``Training_Data/N{N}/train``
    ``Training_Data/N{N}/valid``

    Parameters
    ----------
    base_dir : str | Path
        Base directory that will contain the generated dataset folders.
    N : int
        Number of players in the generated games.

    Returns
    -------
    tuple[Path, Path]
        ``(train_dir, valid_dir)`` paths.
    """
    base_dir = Path(base_dir)
    run_dir = base_dir / f"N{N}"
    train_dir = run_dir / "train"
    valid_dir = run_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, valid_dir


def save_split_npz(out_dir: str | Path, *, X: np.ndarray, y: np.ndarray, prefix: str) -> None:
    """
    Save one compressed shard with quadratic supervised samples.

    Parameters
    ----------
    out_dir : str | Path
        Directory where the shard is saved.
    X : np.ndarray
        Input features.
    y : np.ndarray
        Ground-truth labels.
    prefix : str
        File name prefix. The function writes ``{prefix}.npz``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{prefix}.npz", X=X, y=y)


def build_supervised_player_dataset(
    Q: np.ndarray,
    B: np.ndarray,
    T_exploration: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-player supervised samples from one batch of quadratic games.

    Parameters
    ----------
    Q : np.ndarray
        Quadratic matrices with shape ``(L, N, N)``.
    B : np.ndarray
        Linear coefficients with shape ``(L, N, 1)``.
    T_exploration : int
        Number of Gaussian exploration turns used to build each sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y)`` where:

        - ``X`` contains ``[x_n(t), cost_n(t)]`` pairs flattened across time
        - ``y`` contains the true labels ``[q_nn_true, b_n_true]``
    """
    _, _, exploration_x, costs = estimate_game_parameters(
        T_exploration=T_exploration,
        Q=Q,
        B=B,
        return_samples=True,
    )

    T, L, N, _ = exploration_x.shape

    feature_pairs = np.concatenate([exploration_x, costs], axis=-1)   # (T, L, N, 2)
    X = np.transpose(feature_pairs, (1, 2, 0, 3)).reshape(L * N, 2 * T)

    q_diag = np.expand_dims(np.diagonal(Q, axis1=1, axis2=2), axis=-1)
    y = np.concatenate([q_diag, B], axis=-1).reshape(L * N, 2)

    return X.astype(np.float32), y.astype(np.float32)


def generate_and_save_dataset_in_batches(
    *,
    out_dir: str | Path,
    L_total: int,
    L_batch: int,
    N: int,
    T_exploration: int,
    alpha: float,
    beta: float,
    qnn_low: float,
    qnn_high: float,
    delta: float,
    non_symmetric: bool,
    prefix: str = "data",
) -> None:
    """
    Generate the dataset in batches and save it as ``.npz`` shards.

    Parameters
    ----------
    out_dir : str | Path
        Output directory for the current split.
    L_total : int
        Total number of games to generate for the split.
    L_batch : int
        Number of games generated and saved per shard.
    N : int
        Number of players.
    T_exploration : int
        Number of Gaussian exploration turns per game.
    alpha : float
        Distance-decay parameter used in ``generate_Q_B``.
    beta : float
        Sampling range parameter for the true linear coefficient ``B``.
    qnn_low : float
        Lower bound for the diagonal entries of ``Q``.
    qnn_high : float
        Upper bound for the diagonal entries of ``Q``.
    delta : float
        Off-diagonal asymmetry noise magnitude.
    non_symmetric : bool
        If ``True``, generate non-symmetric games.
    prefix : str, optional
        Base name used for shard files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard = 0
    for start in range(0, L_total, L_batch):
        curr_L = min(L_batch, L_total - start)
        Q, B = generate_Q_B(
            N=N,
            L=curr_L,
            alpha=alpha,
            beta=beta,
            low=qnn_low,
            high=qnn_high,
            delta=delta,
            non_symmetric=non_symmetric,
        )
        X, y = build_supervised_player_dataset(
            Q=Q,
            B=B,
            T_exploration=T_exploration,
        )
        save_split_npz(out_dir, X=X, y=y, prefix=f"{prefix}_{shard:05d}")
        shard += 1


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset generation.
    """
    p = argparse.ArgumentParser(description="Build quadratic supervised training data")
    p.add_argument("--base_dir", type=str, default="Training_Data")
    p.add_argument("--L", type=int, default=4000, help="Number of training games")
    p.add_argument("--valid_L", type=int, default=800, help="Number of validation games")
    p.add_argument("--L_batch", type=int, default=500, help="Games per shard")
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--T_exploration", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--qnn_low", type=float, default=1.2)
    p.add_argument("--qnn_high", type=float, default=2.2)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--non_symmetric", action="store_true")
    return p.parse_args()


def main() -> None:
    """
    Entry point for building train and validation quadratic datasets.
    """
    args = parse_args()
    train_dir, valid_dir = make_dataset_dirs(args.base_dir, N=args.N)

    generate_and_save_dataset_in_batches(
        out_dir=train_dir,
        L_total=args.L,
        L_batch=args.L_batch,
        N=args.N,
        T_exploration=args.T_exploration,
        alpha=args.alpha,
        beta=args.beta,
        qnn_low=args.qnn_low,
        qnn_high=args.qnn_high,
        delta=args.delta,
        non_symmetric=args.non_symmetric,
        prefix="data",
    )

    generate_and_save_dataset_in_batches(
        out_dir=valid_dir,
        L_total=args.valid_L,
        L_batch=args.L_batch,
        N=args.N,
        T_exploration=args.T_exploration,
        alpha=args.alpha,
        beta=args.beta,
        qnn_low=args.qnn_low,
        qnn_high=args.qnn_high,
        delta=args.delta,
        non_symmetric=args.non_symmetric,
        prefix="data",
    )

    print("Saved shards to:")
    print(" train:", train_dir)
    print(" valid:", valid_dir)


if __name__ == "__main__":
    main()
