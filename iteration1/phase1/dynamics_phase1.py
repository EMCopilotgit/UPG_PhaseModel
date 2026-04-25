import numpy as np

from iteration1.shared.params import Params
from iteration1.shared.state_init import init_random_state
from iteration1.shared.phase_update import update_phase
from iteration1.shared.update_rules import update_state
from iteration1.shared.logging import init_logs, log_step

def default_params_phase1():
    return Params(
        beta=0.1,
        gamma=0.05,
        eta_T=0.0,    # no torsion coupling yet
        eta_F=0.0,    # no phase-field coupling yet
        alpha_th=0.1,
        noise_x=0.0,
        noise_th=0.0,
        w_R=1.0,
        w_T=1.0,
        w_F=1.0
    )

def run_phase1(T_steps=500, N=64, target_value=0.5, params=None):
    if params is None:
        params = default_params_phase1()

    x, theta = init_random_state(N)
    logs = init_logs()

    for t in range(T_steps):
        # phase update (U(1) coupling)
        theta = update_phase(theta, params)

        # state update (unified curvature-only rule in Phase I)
        x, y_em, eps, N_unified, R, T, F, E = update_state(
            x, theta, params, target_value
        )

        log_step(logs, y_em, eps, N_unified, R, T, F)

    final_state = {"x": x, "theta": theta}
    return logs, final_state
