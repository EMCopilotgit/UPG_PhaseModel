import numpy as np

from iteration2.phase3_2d.state_init_2d import init_random_state_2d
from iteration2.phase3_2d.update_2d import update_n_2d
from iteration2.phase3_2d.defects_2d import defects_summary_2d


def run_phase3_2d(T_steps=500, Nx=64, Ny=64, dt=0.01, kappa=0.2):
    """
    Iteration 2 — Phase III (2D):
    Topological defects on a 2D S^2 field.

    - Evolve n[i,j] on S^2 with diffusive dynamics
    - Compute topological charge density per plaquette
    - Summarize total charge and defect cores
    """
    n = init_random_state_2d(Nx, Ny)
    logs = {"defects": []}

    for t in range(T_steps):
        n = update_n_2d(n, dt=dt, kappa=kappa)
        dsum = defects_summary_2d(n)
        logs["defects"].append(dsum)

    final_state = {"n": n}
    return logs, final_state
