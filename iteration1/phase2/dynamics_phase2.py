# phase2/dynamics_phase2.py

import numpy as np

from iteration1.shared.params import Params
from iteration1.shared.state_init import init_state_for_input
from iteration1.shared.phase_update import update_phase
from iteration1.shared.update_rules import update_state
from iteration1.shared.logging import init_logs, log_step


def default_params_phase2():
    return Params(
        beta=0.1,
        gamma=0.1,     # stronger global error correction than Phase I
        eta_T=0.02,    # enable torsion coupling
        eta_F=0.02,    # enable phase-field coupling
        alpha_th=0.1,
        noise_x=0.0,
        noise_th=0.0,
        w_R=1.0,
        w_T=1.0,
        w_F=1.0
    )


def run_phase2(inputs, f, T_steps=500, N=64, params=None):
    """
    For each input x_in:
        - initialize state near x_in
        - run unified-field dynamics
        - log final MAE and unified energy
    """
    if params is None:
        params = default_params_phase2()

    results = []

    for x_in in inputs:
        target_value = f(x_in)

        # initialize state for this input
        x, theta = init_state_for_input(x_in, N)
        logs = init_logs()

        for t in range(T_steps):
            theta = update_phase(theta, params)
            x, y_em, eps, N_unified, R, T, F, E = update_state(
                x, theta, params, target_value
            )
            log_step(logs, y_em, eps, N_unified, R, T, F)

        # final metrics
        final_mae = abs(logs["y_em"][-1] - target_value)
        final_N = logs["N"][-1]

        results.append({
            "x_in": float(x_in),
            "target": float(target_value),
            "final_y_em": float(logs["y_em"][-1]),
            "final_mae": float(final_mae),
            "final_unified_N": float(final_N),
            "logs": logs
        })

    return results
