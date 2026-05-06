import sys
import numpy as np
from dataclasses import dataclass

from .sweep_grid import SweepConfig, sweep_grid
from .boundary import extract_lambda_star
from .plot_phase4 import (
    plot_T_star_heatmap,
    plot_binary_phase_diagram,
    plot_T_star_surface,
)


def default_config():
    return SweepConfig(
        etas=np.linspace(0.001, 0.010, 10),
        lambdas=np.linspace(0.02, 0.20, 20),
        max_episodes=400,
        n_seeds=3,
        equilibrium_window=40,
        equilibrium_tol=0.02,
    )


def test_config():
    return SweepConfig(
        etas=np.linspace(0.001, 0.003, 2),
        lambdas=np.linspace(0.02, 0.04, 2),
        max_episodes=20,
        n_seeds=1,
        equilibrium_window=10,
        equilibrium_tol=0.05,
    )



import sys

def main(mode=None):
    # Priority 1: explicit argument from run_all.py
    if mode == "test":
        print("Running Phase 4 in TEST MODE (fast micro-sweep)")
        cfg = test_config()

    # Priority 2: command-line invocation
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        print("Running Phase 4 in TEST MODE (fast micro-sweep)")
        cfg = test_config()

    # Default: full sweep
    else:
        print("Running Phase 4 in FULL MODE (long sweep)")
        cfg = default_config()

    # Run sweep
    T_star, eq_flag = sweep_grid(cfg)
    lambda_star_curve = extract_lambda_star(cfg, eq_flag)

    # Plots
    plot_T_star_heatmap(cfg, T_star)
    plot_binary_phase_diagram(cfg, eq_flag, lambda_star_curve)
    plot_T_star_surface(cfg, T_star)

    print("Phase 4 equilibrium pipeline complete.")

if __name__ == "__main__":
    main()
