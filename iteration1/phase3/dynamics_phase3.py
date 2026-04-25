import numpy as np

from iteration1.shared.params import Params
from iteration1.shared.state_init import init_state_for_input
from iteration1.shared.phase_update import update_phase
from iteration1.shared.update_rules import update_state
from iteration1.shared.logging import init_logs as base_init_logs
from iteration1.shared.logging import log_step as base_log_step
from iteration1.phase3.defects_phase3 import detect_defects_1d


def default_params_phase3():
    return Params(
        beta=0.0,
        gamma=0.1,
        eta_T=0.0,
        eta_F=0.0,
        alpha_th=0.05,
        omega_th=0.01,
        kappa_th=0.2,
        gamma_th=0.05,
        noise_x=0.01,
        noise_th=0.0,
        w_R=1.0,
        w_T=1.0,
        w_F=1.0,
        mu=0.02,        # double-well strength
        v=1.0,          # preferred |x|
        kappa=0.1,      # spatial coupling
        alpha_x=0.1,   # phase drive into x
        dt=0.01
    )

def init_logs():
    logs = base_init_logs()
    logs.update({
        "R": [],
        "T": [],
        "F": [],
        "N_field": [],
        "R_mean": [],
        "T_mean": [],
        "F_mean": [],
        "defect_positions": [],
        "defect_charges": [],
        "defect_types": [],
    })
    return logs


def log_step(logs, y_em, eps, N_unified, R, T, F, N_field=None):
    # base scalar logs
    base_log_step(logs, y_em, eps, N_unified, R, T, F)

    # full fields
    logs["R"].append(R.copy())
    logs["T"].append(T.copy())
    logs["F"].append(F.copy())

    # means for plotting
    logs["R_mean"].append(float(np.mean(R)))
    logs["T_mean"].append(float(np.mean(T)))
    logs["F_mean"].append(float(np.mean(F)))

    # full unified field
    if N_field is not None:
        logs["N_field"].append(N_field.copy())


def run_single_task(inputs, f, T_steps, N, params):
    """
    Runs one function-learning task (sin or cos) over all inputs.
    Returns list of results.
    """
    results = []

    for x_in in inputs:
        target_value = f(x_in)

        x, theta = init_state_for_input(x_in, N)
        logs = init_logs()

        for t in range(T_steps):

            # --- update x first ---
            x, y_em, eps, N_unified, R, T, F, E, N_field = update_state(
                x, theta, params, target_value
            )

            # --- build local error field ---
            eps_field = np.full_like(theta, eps)

            # --- update phase with revised rule ---
            theta = update_phase(theta, params, eps_field=eps_field)

            # --- log everything ---
            log_step(logs, y_em, eps, N_unified, R, T, F, N_field=N_field)
        
            # defect detection per time step
            pos, ch, ty = detect_defects_1d(N_field)
            logs["defect_positions"].append(pos)
            logs["defect_charges"].append(ch)
            logs["defect_types"].append(ty)

        final_mae = abs(logs["y_em"][-1] - target_value)

        results.append({
            "x_in": float(x_in),
            "target": float(target_value),
            "final_y_em": float(logs["y_em"][-1]),
            "final_mae": float(final_mae),
            "final_unified_N": float(logs["N"][-1]),
            "logs": logs,
            "final_fields": {
                "R": R.copy(),
                "T": T.copy(),
                "F": F.copy(),
                "N": N_field.copy(),
                "x": x.copy(),
            },
            "defects": {
                "positions": logs["defect_positions"],
                "charges": logs["defect_charges"],
                "types": logs["defect_types"],
            },
        })

    return results


def run_phase3(inputs, T_steps=600, N=64, params=None):
    """
    Phase III: Continual learning sequence:
        1. Learn sin(x)
        2. Learn cos(x)
        3. Learn sin(x) again
    """
    if params is None:
        params = default_params_phase3()

    # Task 1: sin(x)
    results_sin1 = run_single_task(inputs, np.sin, T_steps, N, params)

    # Task 2: cos(x)
    results_cos = run_single_task(inputs, np.cos, T_steps, N, params)

    # Task 3: sin(x) again
    results_sin2 = run_single_task(inputs, np.sin, T_steps, N, params)

    final_fields = results_sin2[-1]["final_fields"]

    # choose Phase V source trajectory: last input of last task
    phase5_source = results_sin2[-1]
    logs = phase5_source["logs"]

    defect_positions = logs["defect_positions"]
    defect_charges = logs["defect_charges"]
    defect_types = logs["defect_types"]

    return {
        "sin1": results_sin1,
        "cos": results_cos,
        "sin2": results_sin2,
        "R": final_fields["R"],
        "T": final_fields["T"],
        "F": final_fields["F"],
        "N": final_fields["N"],
        "x": final_fields["x"],
        "defect_positions": defect_positions,
        "defect_charges": defect_charges,
        "defect_types": defect_types,
    }
