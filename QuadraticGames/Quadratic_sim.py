"""
Simulation entrypoint for the quadratic-game experiments.

This script generates quadratic games, estimates parameters from exploration,
runs the Nash, optimal, and prior-approximation updates, and optionally saves
the comparison plots used in the project figures. If `--weights` is provided,
the prior parameters are estimated with the trained offline neural network
instead of the expectation-based estimator.

Usage example
-------------
python QuadraticGames/Quadratic_sim.py
python QuadraticGames/Quadratic_sim.py --weights QuadraticGames/Training_Data/N20/results
"""

from utils.quad_utils import *
from utils.data_structure import *
from utils.plot_utils import plot_quadratic_mean_cost

def main(cfg: SimConfig) -> None:
    """Run the full quadratic-game simulation for one parsed configuration."""
    Q, B = generate_Q_B(
        N=cfg.N,
        L=cfg.L,
        alpha=cfg.alpha,
        beta=cfg.beta,
        low=cfg.qnn_low,
        high=cfg.qnn_high,
        delta=cfg.delta,
        non_symmetric=cfg.non_symmetric,
    )
    asymmetry_pct = compute_asymmetry_percentage(Q)

    x_init = np.random.uniform(0.1, 1.1, size=(cfg.L, cfg.N, 1))
    ne_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    optimal_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    prior_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    if cfg.weights:
        diagonals_est, B_est, exploration_turns = estimate_game_parameters_with_nn(
            weights_dir=cfg.weights,
            Q=Q,
            B=B,
        )
        print(f"Using NN-based parameter estimation from: {cfg.weights}")
    else:
        diagonals_est, B_est = estimate_game_parameters(
            T_exploration=cfg.T_exploration,
            Q=Q,
            B=B,
        )
        exploration_turns = cfg.T_exploration
        print("Using expectation-based parameter estimation.")

    IterationLoop(
        cfg=cfg,
        Q=Q,
        B=B_est,
        x=x_init,
        sim_record=ne_record,
        grad_mode=GradMode.NAIVE_NASH,
    )

    IterationLoop(
        cfg=cfg,
        Q=Q,
        B=B,
        x=x_init,
        sim_record=optimal_record,
        grad_mode=GradMode.OPTIMAL,
    )

    IterationLoop(
        cfg=cfg,
        Q=Q,
        B=B_est,
        x=x_init,
        sim_record=prior_record,
        grad_mode=GradMode.PRIOR_APPROXIMATION,
        diagonals_est=diagonals_est,
    )

    if cfg.isPlot:
        plot_quadratic_mean_cost(
            ne_mean_cost=ne_record.mean_cost,
            optimal_mean_cost=optimal_record.mean_cost,
            dcpa_mean_cost=prior_record.mean_cost,
            optimal_std_cost=np.std(optimal_record.sum_cost, axis=0),
            ne_std_cost=np.std(ne_record.sum_cost, axis=0),
            dcpa_std_cost=np.std(prior_record.sum_cost, axis=0),
            num_players=cfg.N,
            asymmetry_pct=asymmetry_pct,
            non_symmetric=cfg.non_symmetric,
            plot_std=cfg.plot_std,
            debug=cfg.debug,
            y_min=cfg.y_min,
            y_max=cfg.y_max,
        )

    print(f"Final mean payoff NE: {ne_record.mean_cost[-1]:.6f}")
    print(f"Final mean payoff Optimal: {optimal_record.mean_cost[-1]:.6f}")
    print(f"Final mean payoff DCPA: {prior_record.mean_cost[-1]:.6f}")
    print(f"Exploration turns used for parameter estimation: {exploration_turns}")
    print(f"Delta from symmetry: {asymmetry_pct:.2f}%")
    print("Quadratic simulation finished.")








if '__main__' == __name__:
    cfg = parse_args()
    print("================================ Running Quadratic Simulation ===============================")
    print(cfg)
    main(cfg)
