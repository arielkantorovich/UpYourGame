"""
Created on : ------

@author: Ariel_Kantorovich
"""

from utils.quad_utils import *
from utils.data_structure import *
from utils.plot_utils import plot_quadratic_mean_cost

def main(cfg: SimConfig) -> None:
    Q, B = generate_Q_B(
        N=cfg.N,
        L=cfg.L,
        alpha=cfg.alpha,
        beta=cfg.beta,
        low=cfg.qnn_low,
        high=cfg.qnn_high,
    )

    x_init = np.random.uniform(0.1, 1.1, size=(cfg.L, cfg.N, 1))
    ne_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    optimal_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    prior_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    diagonals_est = estimate_diagonals(
        T_exploration=cfg.T_exploration,
        Q=Q,
        B=B,
    )

    IterationLoop(
        cfg=cfg,
        Q=Q,
        B=B,
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
        B=B,
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
            num_players=cfg.N,
            debug=cfg.debug,
        )

    print(f"Final mean cost NE: {ne_record.mean_cost[-1]:.6f}")
    print(f"Final mean cost Optimal: {optimal_record.mean_cost[-1]:.6f}")
    print(f"Final mean cost DCPA: {prior_record.mean_cost[-1]:.6f}")
    print(f"Exploration turns used for diagonal estimation: {cfg.T_exploration}")
    print("Quadratic simulation finished.")








if '__main__' == __name__:
    cfg = parse_args()
    print("================================ Running Quadratic Simulation ===============================")
    print(cfg)
    main(cfg)
