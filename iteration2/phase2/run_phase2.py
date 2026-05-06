import numpy as np

from iteration2.shared.params import Params
from iteration2.shared.state_init import init_random_state
from iteration2.shared.phase_update import update_phase
from iteration2.shared.update_rules import update_state
from iteration2.shared.logging import init_logs, log_step
from iteration2.phase2.diagnostics_phase2 import diagnostics_phase2


def default_params_phase2():
    """
    Phase II parameters.
    Slightly stronger geometric coupling than Phase I.
    """
    return Params(
        alpha_n=0.1,
        beta=0.0,
        gamma=0.2,
        kappa=0.2,
        noise_n=0.0,
        noise_x=0.0,
        dt=0.01
    )


def run_phase2(T_steps=500, N=64, target_value=0.5, params=None):
    """
    Iteration 2 — Phase II:
    Unified geometric diagnostics.

    - Update S^2 vector-phase field
    - Compute curvature, torsion, unified magnitude
    - Compute coherence and geometric statistics
    - Log everything
    """
    if params is None:
        params = default_params_phase2()

    x, n = init_random_state(N)
    logs = init_logs()
    logs["geom"] = []   # geometric diagnostics

    for t in range(T_steps):

        # update S^2 phase
        n = update_phase(n, params)

        # update x-field + compute geometry
        x, y_em, eps, N_unified, curvature, torsion, n_copy = update_state(
            x, n, params, target_value
        )

        # log fields
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

        # geometric diagnostics
        geom = diagnostics_phase2(n_copy, curvature, torsion, N_unified)
        logs["geom"].append(geom)

    final_state = {"x": x, "n": n}
    return logs, final_state
