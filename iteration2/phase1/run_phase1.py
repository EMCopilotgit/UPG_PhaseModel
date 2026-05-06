import numpy as np

from iteration2.shared.params import Params
from iteration2.shared.state_init import init_random_state
from iteration2.shared.phase_update import update_phase
from iteration2.shared.update_rules import update_state
from iteration2.shared.logging import init_logs, log_step
from iteration2.shared.diagnostics import diagnostics_summary


def default_params_phase1():
    """
    Phase I parameters for the S^2 vector-phase model.
    This is the simplest stable configuration.
    """
    return Params(
        alpha_n=0.1,     # phase update strength
        beta=0.0,        # curvature coupling into x
        gamma=0.1,       # torsion coupling
        kappa=0.1,       # spatial coupling
        noise_n=0.0,     # no phase noise for stability test
        noise_x=0.0,     # no x noise for stability test
        dt=0.01
    )


def run_phase1(T_steps=500, N=64, target_value=0.5, params=None):
    """
    Iteration 2 — Phase I:
    Stability test of the S^2 vector-phase model.

    - Initialize x-field and S^2 phase field n[i]
    - Update n via renormalized vector-phase rule
    - Update x via unified geometric field
    - Log curvature, torsion, unified magnitude, n-field, x-field
    - Run diagnostics each step (normalization, energy, instability)
    """
    if params is None:
        params = default_params_phase1()

    # --- initialize fields ---
    x, n = init_random_state(N)
    logs = init_logs()
    logs["diagnostics"] = []   # add diagnostics channel

    for t in range(T_steps):

        # --- update phase (S^2 vector) ---
        n = update_phase(n, params)

        # --- update x-field + compute unified geometry ---
        x, y_em, eps, N_unified, curvature, torsion, n_copy = update_state(
            x, n, params, target_value
        )

        # --- log everything ---
        log_step(
            logs,
            y_em,
            eps,
            N_unified,
            curvature,
            torsion,
            n_copy,
            x
        )

        # --- diagnostics ---
        diag = diagnostics_summary(n)
        logs["diagnostics"].append(diag)

    final_state = {"x": x, "n": n}
    return logs, final_state
