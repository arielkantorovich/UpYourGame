"""
Created on : ------

@author: Ariel_Kantorovich
"""

from pathlib import Path

import numpy as np
import torch
from dnn_utils.nn_utils import quadratic_nn_predict
from dnn_utils.train_data_struct import load_configs_from_yaml
from utils.data_structure import GradMode, SimConfig, SimRecord



def generate_Q_B(
    N: int,
    L: int,
    alpha: float = 1,
    beta: float = 1,
    low: float = 1,
    high: float = 2,
    delta: float = 0.001,
    non_symmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the quadratic-game matrices ``Q`` and vectors ``B`` for ``L`` samples.

    Each sample starts from ``N`` random 2D player positions. The pairwise distances
    between players are converted into the interaction matrix ``Q`` through
    ``exp(-alpha * distance)``:

    - A larger ``alpha`` makes the exponential decay faster, so off-diagonal
      interactions between distant players become smaller more quickly.
    - A smaller ``alpha`` makes the decay milder, so distant players keep a
      stronger influence in ``Q``.

    After that, the diagonal of each ``Q`` matrix is replaced with random values in
    ``[low, high]`` so that each player's self-term is sampled independently.

    The linear term ``B`` is sampled uniformly from ``[-beta, beta]``:

    - ``beta`` controls the magnitude of the linear coefficients.
    - A larger ``beta`` allows larger positive or negative entries in ``B``.
    - A smaller ``beta`` keeps the entries of ``B`` closer to zero.

    Parameters
    ----------
    N : int
        Number of players in each generated game instance.
    L : int
        Number of game instances to generate.
    alpha : float, optional
        Distance-decay factor used in ``Q = exp(-alpha * distance)``.
    beta : float, optional
        Symmetric bound used to sample ``B`` from ``[-beta, beta]``.
    low : float, optional
        Lower bound for the random diagonal values of ``Q``.
    high : float, optional
        Upper bound for the random diagonal values of ``Q``.

    If ``non_symmetric`` is enabled, uniform off-diagonal noise sampled from
    ``[-delta, delta]`` is added independently to each entry of ``Q``. The
    diagonal is kept unchanged, so only the cross-player interactions lose
    symmetry.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``Q`` with shape ``(L, N, N)`` and ``B`` with shape ``(L, N, 1)``.
    """
    # Step 1: Generate L different sets of N random points
    points = np.random.rand(L, N, 2)

    # Step 2: Build distance maps D for all trials
    points_expanded = points[:, np.newaxis, :, :]
    differences = points_expanded - points[:, :, np.newaxis, :]
    distances = np.linalg.norm(differences, axis=-1)

    # Step 3: Generate Q = exp(-alpha*D) for all trials
    Q = np.exp(-alpha * distances)

    # Step 4: Randomly sample diagonal values and update Q diagonals
    random_diagonals = np.random.uniform(low, high, size=(L, N))  # Generate random values for each diagonal
    for i in range(L):  # Update each trial's diagonal individually
        np.fill_diagonal(Q[i], random_diagonals[i])

    if non_symmetric:
        asym_noise = np.random.uniform(low=-delta, high=delta, size=(L, N, N))
        diag_mask = np.eye(N, dtype=bool)[None, :, :]
        asym_noise = np.where(diag_mask, 0.0, asym_noise)
        Q = Q + asym_noise

    # Step 5: Generate B
    B = np.random.uniform(low=-beta, high=beta, size=(L, N, 1))
    # Replace Q to concave problem
    Q = -1.0 * Q
    return Q, B


def compute_asymmetry_percentage(Q: np.ndarray) -> float:
    """
    Measure how far ``Q`` is from symmetry as a percentage.

    The metric is the relative Frobenius norm of the antisymmetric part:

    ``Delta = ||Q - Q^T||_F / (2 ||Q||_F) * 100``

    averaged across all ``L`` generated games. A symmetric game gives
    ``Delta = 0``.
    """
    q_transpose = np.transpose(Q, axes=(0, 2, 1))
    numerator = np.linalg.norm(Q - q_transpose, axis=(1, 2))
    denominator = 2.0 * np.linalg.norm(Q, axis=(1, 2))
    delta_per_game = 100.0 * numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(delta_per_game))


def _build_lr_schedule(cfg: SimConfig) -> np.ndarray:
    return cfg.lr * np.ones(cfg.T)


def sample_quadratic_exploration(
    T_exploration: int,
    Q: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample Gaussian exploration actions and the corresponding realized costs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(exploration_x, costs)`` with shapes ``(T, L, N, 1)``.
    """
    exploration_x = np.random.normal(loc=0.0, scale=1.0, size=(T_exploration, *B.shape))
    qx = np.matmul(Q[None, :, :, :], exploration_x)
    diagonals = np.expand_dims(np.diagonal(Q, axis1=1, axis2=2), axis=(0, -1))
    costs = 0.5 * (2 * exploration_x * qx - diagonals * (exploration_x ** 2)) + B[None, :, :, :] * exploration_x
    return exploration_x, costs


def estimate_game_parameters(
    T_exploration: int,
    Q: np.ndarray,
    B: np.ndarray,
    return_samples: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the diagonal terms of ``Q`` and the linear term ``B`` from
    exploration trajectories.

    Each player samples actions i.i.d. from ``N(0, 1)`` for ``T_exploration`` turns.
    For quadratic costs with zero-mean unit-variance actions,

    ``E[cost_n] = 0.5 * q_nn``

    so the diagonal estimate is obtained by doubling the empirical mean cost.

    In addition,

    ``E[cost_n * x_n] = b_n``

    because the mixed quadratic terms have zero expectation under independent
    zero-mean Gaussian exploration. Therefore the linear coefficient estimate is
    the empirical mean of ``cost_n * x_n``.

    Parameters
    ----------
    T_exploration : int
        Number of exploration samples.
    Q : np.ndarray
        Quadratic matrix with shape ``(L, N, N)``.
    B : np.ndarray
        Linear term with shape ``(L, N, 1)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(q_nn_est, B_est)`` with shapes ``(L, N, 1)`` and ``(L, N, 1)``.

    If ``return_samples=True``, also returns ``(exploration_x, costs)`` with
    shapes ``(T_exploration, L, N, 1)``.
    """
    exploration_x, costs = sample_quadratic_exploration(
        T_exploration=T_exploration,
        Q=Q,
        B=B,
    )
    q_nn_est = 2.0 * np.mean(costs, axis=0)
    B_est = np.mean(costs * exploration_x, axis=0)
    if return_samples:
        return q_nn_est, B_est, exploration_x, costs
    return q_nn_est, B_est


def estimate_game_parameters_with_nn(
    weights_dir: str | Path,
    Q: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Estimate quadratic diagonals and linear terms using a trained neural network.

    The function loads `train_config.yaml` from the results folder to recover the
    exploration length used during training, regenerates the exploration features,
    and predicts `[q_nn, b_n]` for each player with the saved `model.pt`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        `(q_nn_est, B_est, T_exploration)` where the estimates have shape
        `(L, N, 1)`.
    """
    weights_dir = Path(weights_dir)
    cfg_path = weights_dir / "train_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {cfg_path}")

    train_cfg, _ = load_configs_from_yaml(str(cfg_path))
    _, _, exploration_x, costs = estimate_game_parameters(
        T_exploration=train_cfg.T_exploration,
        Q=Q,
        B=B,
        return_samples=True,
    )

    T, L, N, _ = exploration_x.shape
    feature_pairs = np.concatenate([costs * exploration_x, costs, exploration_x], axis=-1)
    X = np.transpose(feature_pairs, (1, 2, 0, 3)).reshape(L * N, 3 * T).astype(np.float32)

    predictions = quadratic_nn_predict(
        input_path=str(weights_dir),
        inputs=torch.as_tensor(X, dtype=torch.float32),
        cfg=train_cfg,
    )
    predictions = predictions.reshape(L, N, train_cfg.output_dim)
    if train_cfg.output_dim < 2:
        raise ValueError("Quadratic NN output_dim must be at least 2 to predict q_nn and B.")

    q_nn_est = predictions[:, :, 0:1]
    B_est = predictions[:, :, 1:2]
    return q_nn_est, B_est, train_cfg.T_exploration


def build_player_subset_dataset(
    exploration_x: np.ndarray,
    costs: np.ndarray,
    trajectory_x: np.ndarray,
    trajectory_cost: np.ndarray,
    optimal_gradients: np.ndarray,
    Q: np.ndarray,
    B: np.ndarray,
    N_subset: int,
    player_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training dataset for a random subset of N_subset players.
    
    This function randomly samples N_subset players from the total N players
    and creates training data only for those sampled players, reducing memory
    usage while maintaining training quality.
    
    Parameters
    ----------
    exploration_x : np.ndarray
        Exploration actions with shape ``(T, L, N, 1)``.
    costs : np.ndarray
        Exploration costs with shape ``(T, L, N, 1)``.
    trajectory_x : np.ndarray
        Optimal agent trajectory actions with shape ``(T_record, L, N, 1)``.
    trajectory_cost : np.ndarray
        Optimal agent trajectory costs with shape ``(T_record, L, N, 1)``.
    optimal_gradients : np.ndarray
        Optimal gradients with shape ``(T_record, L, N, 1)``.
    Q : np.ndarray
        Quadratic matrices with shape ``(L, N, N)``.
    B : np.ndarray
        Linear coefficients with shape ``(L, N, 1)``.
    N_subset : int
        Number of players to randomly sample (must be <= N).
    player_idx : np.ndarray | None, optional
        Specific player indices to use. If None, randomly samples N_subset players
        without replacement.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(X, Z, y, params)`` where:
        
        - ``X`` contains exploration features ``[cost_n(t)*x_n(t), cost_n(t), x_n(t)]`` triplets
          with shape ``(L * N_subset, 3 * T)``
        - ``Z`` contains loss path ``[x_n(t), R_n(t)]`` pairs
          with shape ``(L * N_subset, 2 * T_record)``
        - ``y`` contains optimal gradient labels
          with shape ``(L * N_subset, T_record)``
        - ``params`` contains true parameters ``[q_nn, b_n]``
          with shape ``(L * N_subset, 2)``
    
    Raises
    ------
    ValueError
        If N_subset > N or if player_idx is invalid.
    """
    T, L, N, _ = exploration_x.shape
    T_record = trajectory_x.shape[0]
    
    # Validate and generate player indices
    if player_idx is None:
        if N_subset > N:
            raise ValueError(f"N_subset={N_subset} cannot be > N={N}")
        # Random sampling without replacement
        player_idx = np.random.choice(N, size=N_subset, replace=False)
    else:
        player_idx = np.asarray(player_idx, dtype=int)
        if player_idx.ndim != 1:
            raise ValueError("player_idx must be 1D")
        if len(player_idx) != N_subset:
            raise ValueError(f"player_idx length {len(player_idx)} != N_subset {N_subset}")
        if np.any(player_idx < 0) or np.any(player_idx >= N):
            raise ValueError("player_idx contains out-of-range indices")
    
    # Subsample exploration data: (T, L, N, 1) → (T, L, N_subset, 1)
    exploration_x_sub = exploration_x[:, :, player_idx, :]
    costs_sub = costs[:, :, player_idx, :]
    
    # Subsample trajectory data: (T_record, L, N, 1) → (T_record, L, N_subset, 1)
    trajectory_x_sub = trajectory_x[:, :, player_idx, :]
    trajectory_cost_sub = trajectory_cost[:, :, player_idx, :]
    optimal_gradients_sub = optimal_gradients[:, :, player_idx, :]
    
    # Build X: Exploration features [cost_n(t)*x_n(t), cost_n(t), x_n(t)]
    feature_pairs = np.concatenate([costs_sub * exploration_x_sub, costs_sub, exploration_x_sub], axis=-1)  # (T, L, N_subset, 3)
    X = np.transpose(feature_pairs, (1, 2, 0, 3)).reshape(L * N_subset, 3 * T)
    
    # Build Z: Loss path [x_n(t), R_n(t)]
    loss_path_pairs = np.concatenate([trajectory_x_sub, trajectory_cost_sub], axis=-1)  # (T_record, L, N_subset, 2)
    Z = np.transpose(loss_path_pairs, (1, 2, 0, 3)).reshape(L * N_subset, 2 * T_record)
    
    # Build y: Optimal gradient labels
    y = np.transpose(optimal_gradients_sub[:, :, :, 0], (1, 2, 0)).reshape(L * N_subset, T_record)
    
    # Extract parameters for sampled players only
    q_nn = np.diagonal(Q, axis1=1, axis2=2)  # (L, N)
    q_nn_sub = q_nn[:, player_idx]  # (L, N_subset)
    
    b_n = B[:, :, 0]  # (L, N)
    b_n_sub = b_n[:, player_idx]  # (L, N_subset)
    
    # Stack parameters: [q_nn, b_n] pairs
    params = np.stack([q_nn_sub.ravel(), b_n_sub.ravel()], axis=1)  # (L * N_subset, 2)
    
    return X.astype(np.float32), Z.astype(np.float32), y.astype(np.float32), params.astype(np.float32)


def IterationLoop(
    cfg: SimConfig,
    Q: np.ndarray,
    B: np.ndarray,
    x: np.ndarray,
    sim_record: SimRecord,
    grad_mode: GradMode,
    diagonals_est: np.ndarray | None = None,
) -> None:
    """
    Run one quadratic-game iteration loop and write all outputs into ``sim_record``.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.
    Q : np.ndarray
        Quadratic matrix with shape ``(L, N, N)``.
    B : np.ndarray
        Linear term with shape ``(L, N, 1)``.
    x : np.ndarray
        Initial player actions with shape ``(L, N, 1)``.
    sim_record : SimRecord
        Output container. The function updates it in-place.
    grad_mode : GradMode
        Selects how the residual gradient is computed.
    diagonals_est : np.ndarray | None, optional
        Estimated diagonals used only for ``PRIOR_APPROXIMATION``.
    """
    lr = _build_lr_schedule(cfg)
    diagonals = np.diagonal(Q, axis1=1, axis2=2)
    diagonals = np.expand_dims(diagonals, axis=-1)
    q_transpose = np.transpose(Q, axes=(0, 2, 1))
    x_curr = x.copy()

    for t in range(cfg.T):
        quadratic_term = np.matmul(Q, x_curr)
        cost = 0.5 * (2 * x_curr * quadratic_term - diagonals * (x_curr ** 2)) + B * x_curr
        local_gradient = quadratic_term + B

        if grad_mode == GradMode.NAIVE_NASH:
            residual_gradient = np.zeros_like(local_gradient)
        elif grad_mode == GradMode.OPTIMAL:
            residual_gradient = np.matmul(q_transpose, x_curr) - diagonals * x_curr
        elif grad_mode == GradMode.PRIOR_APPROXIMATION:
            if diagonals_est is None:
                raise ValueError("diagonals_est must be provided for PRIOR_APPROXIMATION mode.")
            residual_gradient = quadratic_term - diagonals_est * x_curr
        else:
            raise ValueError(f"Unsupported gradient mode: {grad_mode}")

        total_grad = local_gradient + residual_gradient
        sim_record.cost_record[:, t, :] = cost.squeeze(-1)
        sim_record.grad_record[:, t, :] = total_grad.squeeze(-1)

        x_curr = x_curr + lr[t] * total_grad
        x_curr = np.clip(
            x_curr,
            a_min=cfg.action_project_low,
            a_max=cfg.action_project_high,
        )

    sim_record.sum_cost[:, :] = np.sum(sim_record.cost_record, axis=2)
    sim_record.mean_cost[:] = np.mean(sim_record.sum_cost, axis=0)
    sim_record.mean_grad[:, :] = np.mean(sim_record.grad_record, axis=0)
    sim_record.final_x[:, :, :] = x_curr
