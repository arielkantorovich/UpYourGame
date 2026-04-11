"""
Dataset builder for quadratic-game offline training using DCPA approach.

This script generates quadratic games and creates three components:
- X: exploration features from Gaussian sampling
- Z: loss path from optimal agent trajectory 
- Y: optimal gradient labels

The approach follows the DCPA concept from energy games:
1. X contains exploration data: [x_n(t), cost_n(t)] from N(0,1) sampling
2. Z contains optimal agent trajectory over T_loss steps (x_n, R_n)
3. Y contains the target optimal gradient: Q^T * x - q_nn * x

Saved arrays:
- `X`: exploration features [x_n(t), cost_n(t)] pairs
- `Z`: loss path trajectory [x_n(t), R_n(t)] from optimal agent
- `y`: optimal gradient labels

Output layout:
    Training_Data/N{N}/train
    Training_Data/N{N}/valid

Usage examples
--------------
python QuadraticGames/build_data_to_train.py --N 5 --L 4000 --valid_L 800 --T_exploration 200 --T_loss 200
python QuadraticGames/build_data_to_train.py --base_dir QuadraticGames/Training_Data --N 20 --L 5000 --valid_L 1000 --L_batch 500
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

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


def save_build_args_to_yaml(output_dir: str | Path, args: argparse.Namespace, filename: str = "args_build_data.yaml") -> Path:
    """
    Save all data generation arguments to a YAML file for reproducibility.
    
    Parameters
    ----------
    output_dir : str | Path
        Directory where the YAML file will be saved (usually the N{N} directory).
    args : argparse.Namespace
        Parsed command-line arguments from build_data_to_train.py.
    filename : str
        Name of the output YAML file.
        
    Returns
    -------
    Path
        Path to the saved YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename
    
    # Convert argparse.Namespace to dictionary
    args_dict = vars(args)
    
    # Organize into a structured format
    config = {
        "dataset_info": {
            "N": args_dict["N"],
            "N_subset": args_dict["N_subset"],
            "L_train": args_dict["L"],
            "L_valid": args_dict["valid_L"],
            "L_batch": args_dict["L_batch"],
            "base_dir": str(args_dict["base_dir"]),
        },
        "exploration": {
            "T_exploration": args_dict["T_exploration"],
        },
        "loss_path": {
            "T_loss": args_dict["T_loss"],
            "T_jump": args_dict["T_jump"],
            "loss_lr": args_dict["loss_lr"],
            "loss_action_low": args_dict["loss_action_low"],
            "loss_action_high": args_dict["loss_action_high"],
        },
        "game_parameters": {
            "alpha": args_dict["alpha"],
            "beta": args_dict["beta"],
            "qnn_low": args_dict["qnn_low"],
            "qnn_high": args_dict["qnn_high"],
            "delta": args_dict["delta"],
            "non_symmetric": args_dict["non_symmetric"],
        },
    }
    
    # Save to YAML
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved build arguments to: {save_path}")
    return save_path


def save_split_npz(out_dir: str | Path, *, X: np.ndarray, Z: np.ndarray, y: np.ndarray, params: np.ndarray, prefix: str) -> None:
    """
    Save one compressed shard with quadratic supervised samples (DCPA approach).

    Parameters
    ----------
    out_dir : str | Path
        Directory where the shard is saved.
    X : np.ndarray
        Exploration input features.
    Z : np.ndarray
        Loss path trajectory data.
    y : np.ndarray
        Optimal gradient labels.
    params : np.ndarray
        True parameters [q_nn, b_n] with shape (L*N, 2).
    prefix : str
        File name prefix. The function writes ``{prefix}.npz``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{prefix}.npz", X=X, Z=Z, y=y, params=params)


def run_optimal_agent_trajectory(
    Q: np.ndarray,
    B: np.ndarray,
    T_loss: int,
    T_jump: int = 20,
    lr: float = 0.03,
    action_low: float = -8.0,
    action_high: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run optimal agent using real Q, B parameters and residual gradient.
    
    The agent uses the optimal gradient: local_grad + residual_grad
    where residual_grad = Q^T * x - q_nn * x
    
    Only every ``T_jump``-th step is recorded so that the saved trajectory
    spans diverse action values (early transient through convergence).
    
    Parameters
    ----------
    Q : np.ndarray
        Quadratic matrices with shape ``(L, N, N)``.
    B : np.ndarray
        Linear coefficients with shape ``(L, N, 1)``.
    T_loss : int
        Total number of gradient-ascent steps for the optimal agent.
    T_jump : int
        Record interval — store x, cost, gradient every ``T_jump`` steps.
    lr : float
        Learning rate for gradient ascent.
    action_low : float
        Lower bound for action clipping.
    action_high : float
        Upper bound for action clipping.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(trajectory_x, trajectory_cost, optimal_gradients)`` with shapes:
        - trajectory_x: (T_record, L, N, 1)
        - trajectory_cost: (T_record, L, N, 1)
        - optimal_gradients: (T_record, L, N, 1)
        
        where ``T_record = T_loss // T_jump``.
    """
    L, N, _ = B.shape
    T_record = T_loss // T_jump
    
    # Initialize actions same as inference time (Quadratic_sim.py)
    x_curr = np.random.uniform(0.1, 1.1, size=(L, N, 1))
    
    # Get diagonal and transpose of Q
    diagonals = np.diagonal(Q, axis1=1, axis2=2)
    diagonals = np.expand_dims(diagonals, axis=-1)  # (L, N, 1)
    q_transpose = np.transpose(Q, axes=(0, 2, 1))  # (L, N, N)
    
    # Storage for subsampled trajectory
    trajectory_x = np.zeros((T_record, L, N, 1))
    trajectory_cost = np.zeros((T_record, L, N, 1))
    optimal_gradients = np.zeros((T_record, L, N, 1))
    
    record_idx = 0
    for t in range(T_loss):
        # Calculate quadratic term: Q * x
        quadratic_term = np.matmul(Q, x_curr)  # (L, N, 1)
        
        # Calculate cost: 0.5 * (2*x*Qx - q_nn*x^2) + B*x
        cost = 0.5 * (2 * x_curr * quadratic_term - diagonals * (x_curr ** 2)) + B * x_curr
        
        # Calculate local gradient: Q*x + B
        local_gradient = quadratic_term + B
        
        # Calculate residual gradient using REAL parameters: Q^T * x - q_nn * x
        residual_gradient = np.matmul(q_transpose, x_curr) - diagonals * x_curr
        
        # Total optimal gradient
        total_gradient = local_gradient + residual_gradient
        
        # Record at jump intervals
        if t % T_jump == 0 and record_idx < T_record:
            trajectory_x[record_idx] = x_curr.copy()
            trajectory_cost[record_idx] = cost.copy()
            optimal_gradients[record_idx] = residual_gradient.copy()
            record_idx += 1
        
        # Update action using gradient ascent
        x_curr = x_curr + lr * total_gradient
        x_curr = np.clip(x_curr, a_min=action_low, a_max=action_high)
    
    return trajectory_x, trajectory_cost, optimal_gradients


def build_supervised_player_dataset(
    Q: np.ndarray,
    B: np.ndarray,
    T_exploration: int,
    T_loss: int = 200,
    T_jump: int = 20,
    loss_lr: float = 0.03,
    loss_action_low: float = -8.0,
    loss_action_high: float = 8.0,
    N_subset: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-player supervised samples using DCPA approach.

    Parameters
    ----------
    Q : np.ndarray
        Quadratic matrices with shape ``(L, N, N)``.
    B : np.ndarray
        Linear coefficients with shape ``(L, N, 1)``.
    T_exploration : int
        Number of Gaussian exploration turns used to build X.
    T_loss : int
        Total number of steps for optimal agent trajectory.
    T_jump : int
        Record interval for the loss path — subsample every ``T_jump`` steps.
    loss_lr : float
        Learning rate for the optimal agent in the loss path.
    loss_action_low : float
        Lower bound for action projection in the loss path.
    loss_action_high : float
        Upper bound for action projection in the loss path.
    N_subset : int | None
        Number of players to randomly sample. If None, uses all N players.
        Default is None for backward compatibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(X, Z, y, params)`` where:

        - ``X`` contains exploration features ``[x_n(t), cost_n(t)]`` pairs
        - ``Z`` contains subsampled loss path ``[x_n(t), R_n(t)]`` with ``T_record = T_loss // T_jump`` points
        - ``y`` contains optimal gradient labels at the same subsampled points
        - ``params`` contains true parameters ``[q_nn, b_n]`` with shape ``(L*N, 2)`` or ``(L*N_subset, 2)``
    """
    # Build X: Gaussian exploration data
    _, _, exploration_x, costs = estimate_game_parameters(
        T_exploration=T_exploration,
        Q=Q,
        B=B,
        return_samples=True,
    )

    # Build Z: Loss path from optimal agent trajectory (subsampled)
    T_record = T_loss // T_jump
    trajectory_x, trajectory_cost, optimal_gradients = run_optimal_agent_trajectory(
        Q=Q,
        B=B,
        T_loss=T_loss,
        T_jump=T_jump,
        lr=loss_lr,
        action_low=loss_action_low,
        action_high=loss_action_high,
    )
    
    # Check if we should subsample players
    L, N, _ = B.shape
    if N_subset is not None and N_subset < N:
        # Use helper function to subsample N_subset players randomly
        from utils.quad_utils import build_player_subset_dataset
        
        X, Z, y, params = build_player_subset_dataset(
            exploration_x=exploration_x,
            costs=costs,
            trajectory_x=trajectory_x,
            trajectory_cost=trajectory_cost,
            optimal_gradients=optimal_gradients,
            Q=Q,
            B=B,
            N_subset=N_subset,
            player_idx=None,  # Random sampling
        )
        return X, Z, y, params
    
    # Original logic: use all N players
    T, L, N, _ = exploration_x.shape
    feature_pairs = np.concatenate([exploration_x, costs], axis=-1)   # (T, L, N, 2)
    X = np.transpose(feature_pairs, (1, 2, 0, 3)).reshape(L * N, 2 * T)
    
    # Z contains [x_n(t), R_n(t)] pairs from the subsampled optimal agent trajectory
    loss_path_pairs = np.concatenate([trajectory_x, trajectory_cost], axis=-1)  # (T_record, L, N, 2)
    Z = np.transpose(loss_path_pairs, (1, 2, 0, 3)).reshape(L * N, 2 * T_record)
    
    # Build Y: Optimal gradient labels at subsampled timesteps
    y = np.transpose(optimal_gradients[:, :, :, 0], (1, 2, 0)).reshape(L * N, T_record)  # (L*N, T_record)

    # Extract true parameters: q_nn (diagonal of Q) and b_n (from B)
    q_nn = np.diagonal(Q, axis1=1, axis2=2)  # (L, N)
    b_n = B[:, :, 0]  # (L, N)
    
    # Reshape to (L*N, 2) for [q_nn, b_n] pairs
    params = np.stack([q_nn.ravel(), b_n.ravel()], axis=1)  # (L*N, 2)

    return X.astype(np.float32), Z.astype(np.float32), y.astype(np.float32), params.astype(np.float32)


def generate_and_save_dataset_in_batches(
    *,
    out_dir: str | Path,
    L_total: int,
    L_batch: int,
    N: int,
    T_exploration: int,
    T_loss: int,
    T_jump: int = 20,
    alpha: float,
    beta: float,
    qnn_low: float,
    qnn_high: float,
    delta: float,
    non_symmetric: bool,
    loss_lr: float = 0.03,
    loss_action_low: float = -8.0,
    loss_action_high: float = 8.0,
    N_subset: int | None = None,
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
    T_loss : int
        Total number of steps for optimal agent trajectory.
    T_jump : int
        Record interval — subsample the loss path every ``T_jump`` steps.
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
    loss_lr : float
        Learning rate for the optimal agent in the loss path.
    loss_action_low : float
        Lower bound for action projection in the loss path.
    loss_action_high : float
        Upper bound for action projection in the loss path.
    N_subset : int | None
        Number of players to randomly sample. If None, uses all N players.
        Default is None for backward compatibility.
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
        X, Z, y, params = build_supervised_player_dataset(
            Q=Q,
            B=B,
            T_exploration=T_exploration,
            T_loss=T_loss,
            T_jump=T_jump,
            loss_lr=loss_lr,
            loss_action_low=loss_action_low,
            loss_action_high=loss_action_high,
            N_subset=N_subset,
        )
        save_split_npz(out_dir, X=X, Z=Z, y=y, params=params, prefix=f"{prefix}_{shard:05d}")
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
    p.add_argument("--N_subset", type=int, default=5, help="Number of players to randomly sample (default: 5, use 0 or None for all N players)")
    p.add_argument("--T_exploration", type=int, default=200)
    p.add_argument("--T_loss", type=int, default=200, help="Steps for optimal agent trajectory")
    p.add_argument("--T_jump", type=int, default=20, help="Record interval for loss path subsampling")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--qnn_low", type=float, default=1.2)
    p.add_argument("--qnn_high", type=float, default=2.2)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--non_symmetric", action="store_true")
    p.add_argument("--loss_lr", type=float, default=0.03, help="Learning rate for the optimal agent loss path")
    p.add_argument("--loss_action_low", type=float, default=-8.0, help="Lower bound for action projection in loss path")
    p.add_argument("--loss_action_high", type=float, default=8.0, help="Upper bound for action projection in loss path")
    return p.parse_args()


def main() -> None:
    """
    Entry point for building train and validation quadratic datasets.
    """
    args = parse_args()
    train_dir, valid_dir = make_dataset_dirs(args.base_dir, N=args.N)

    # Save build arguments to YAML in the N{N} directory for reproducibility
    dataset_dir = Path(args.base_dir) / f"N{args.N}"
    save_build_args_to_yaml(dataset_dir, args)
    
    # Handle N_subset: if 0 or >= N, use None (all players)
    N_subset = args.N_subset
    if N_subset == 0 or N_subset >= args.N:
        N_subset = None
    
    # Print info about N_subset usage
    if N_subset is None:
        print(f"Using all N={args.N} players for training data")
    else:
        print(f"Randomly sampling N_subset={N_subset} players from N={args.N} (memory reduction: {100*(1-N_subset/args.N):.1f}%)")

    generate_and_save_dataset_in_batches(
        out_dir=train_dir,
        L_total=args.L,
        L_batch=args.L_batch,
        N=args.N,
        T_exploration=args.T_exploration,
        T_loss=args.T_loss,
        T_jump=args.T_jump,
        alpha=args.alpha,
        beta=args.beta,
        qnn_low=args.qnn_low,
        qnn_high=args.qnn_high,
        delta=args.delta,
        non_symmetric=args.non_symmetric,
        loss_lr=args.loss_lr,
        loss_action_low=args.loss_action_low,
        loss_action_high=args.loss_action_high,
        N_subset=N_subset,
        prefix="data",
    )

    generate_and_save_dataset_in_batches(
        out_dir=valid_dir,
        L_total=args.valid_L,
        L_batch=args.L_batch,
        N=args.N,
        T_exploration=args.T_exploration,
        T_loss=args.T_loss,
        T_jump=args.T_jump,
        alpha=args.alpha,
        beta=args.beta,
        qnn_low=args.qnn_low,
        qnn_high=args.qnn_high,
        delta=args.delta,
        non_symmetric=args.non_symmetric,
        loss_lr=args.loss_lr,
        loss_action_low=args.loss_action_low,
        loss_action_high=args.loss_action_high,
        N_subset=N_subset,
        prefix="data",
    )

    print("Saved shards to:")
    print(" train:", train_dir)
    print(" valid:", valid_dir)
    print(f" config: {dataset_dir / 'args_build_data.yaml'}")


if __name__ == "__main__":
    main()
