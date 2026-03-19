"""
Created on : ------

@author: Ariel_Kantorovich
"""

import numpy as np
from utils.data_structure import GradMode, SimConfig, SimRecord



def generate_Q_B(
    N: int,
    L: int,
    alpha: float = 1,
    beta: float = 1,
    low: float = 1,
    high: float = 2,
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

    # Step 5: Generate B
    B = np.random.uniform(low=-beta, high=beta, size=(L, N, 1))
    return Q, B


def _build_lr_schedule(cfg: SimConfig) -> np.ndarray:
    return cfg.lr * np.ones(cfg.T)


def estimate_diagonals(
    T_exploration: int,
    Q: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Estimate the diagonal terms of ``Q`` from exploration trajectories.

    Each player samples actions i.i.d. from ``N(0, 1)`` for ``T_exploration`` turns.
    For quadratic costs with zero-mean unit-variance actions,

    ``E[cost_n] = 0.5 * q_nn``

    so the diagonal estimate is obtained by doubling the empirical mean cost.

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
    np.ndarray
        Estimated diagonals with shape ``(L, N, 1)``.
    """
    exploration_x = np.random.normal(loc=0.0, scale=1.0, size=(T_exploration, *B.shape))
    qx = np.matmul(Q[None, :, :, :], exploration_x)
    diagonals = np.expand_dims(np.diagonal(Q, axis1=1, axis2=2), axis=(0, -1))
    costs = 0.5 * (2 * exploration_x * qx - diagonals * (exploration_x ** 2)) + B[None, :, :, :] * exploration_x
    return 2.0 * np.mean(costs, axis=0)


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
    x_curr = x.copy()

    for t in range(cfg.T):
        quadratic_term = np.matmul(Q, x_curr)
        cost = 0.5 * (2 * x_curr * quadratic_term - diagonals * (x_curr ** 2)) + B * x_curr
        local_gradient = quadratic_term + B

        if grad_mode == GradMode.NAIVE_NASH:
            residual_gradient = np.zeros_like(local_gradient)
        elif grad_mode == GradMode.OPTIMAL:
            residual_gradient = quadratic_term - diagonals * x_curr
        elif grad_mode == GradMode.PRIOR_APPROXIMATION:
            if diagonals_est is None:
                raise ValueError("diagonals_est must be provided for PRIOR_APPROXIMATION mode.")
            residual_gradient = quadratic_term - diagonals_est * x_curr
        else:
            raise ValueError(f"Unsupported gradient mode: {grad_mode}")

        total_grad = local_gradient + residual_gradient
        sim_record.cost_record[:, t, :] = cost.squeeze(-1)
        sim_record.grad_record[:, t, :] = total_grad.squeeze(-1)

        x_curr = x_curr - lr[t] * total_grad
        x_curr = np.clip(
            x_curr,
            a_min=cfg.action_project_low,
            a_max=cfg.action_project_high,
        )

    sim_record.sum_cost[:, :] = np.sum(sim_record.cost_record, axis=2)
    sim_record.mean_cost[:] = np.mean(sim_record.sum_cost, axis=0)
    sim_record.mean_grad[:, :] = np.mean(sim_record.grad_record, axis=0)
    sim_record.final_x[:, :, :] = x_curr
