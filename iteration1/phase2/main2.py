# phase2/main2.py

import numpy as np

from iteration1.phase2.dynamics_phase2 import run_phase2, default_params_phase2
from iteration1.phase2.diagnostics_phase2 import save_phase2_results, plot_phase2


def main():
    # inputs to learn over
    inputs = np.linspace(-2*np.pi, 2*np.pi, 12)

    # choose function: sin or cos
    f = np.sin

    params = default_params_phase2()

    # run Phase II
    results = run_phase2(
        inputs=inputs,
        f=f,
        T_steps=600,
        N=64,
        params=params
    )

    # save all results
    save_phase2_results(results, results_dir="results/phase2", tag="sin")

    # plot the first input's dynamics
    plot_phase2(results[0], results_dir="results/phase2", tag="sin_example")

    print("Phase II complete. Results saved to results/phase2/")


if __name__ == "__main__":
    main()
