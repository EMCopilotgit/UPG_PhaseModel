import numpy as np
from .dynamics_phase4 import (
    inject_noise, apply_drift, localized_perturbation, recovery_step
)
from .diagnostics_phase4 import (
    compute_norm_drift, stability_score, recovery_metric
)
from .plotting_phase4 import Phase4Plotter

def run_phase4(R, T, F, N, x, p):

    R0, T0, F0, N0 = R.copy(), T.copy(), F.copy(), N.copy()
    
    # Initialize perturbation snapshots
    R_before = None
    R_after = None
    T_before = None
    T_after = None
    F_before = None
    F_after = None
    N_before = None
    N_after = None

    R_recovery_traj = []
    in_recovery_window = False        
    history = []

    for t in range(p.steps):

        R, T, F, N = inject_noise(R, T, F, N, p)
        x = apply_drift(x, p)

        if t % 500 == 0 and t > 0:
            R_before = R.copy()
            T_before = T.copy()
            F_before = F.copy()
            N_before = N.copy()

            R, T, F, N = localized_perturbation(R, T, F, N, x, p)

            R_after = R.copy()
            T_after = T.copy()
            F_after = F.copy()
            N_after = N.copy()

            in_recovery_window = True  # start tracking recovery from here

        R, T, F, N = recovery_step(R, T, F, N, p)

        if in_recovery_window:
            R_recovery_traj.append(R.copy())

        history.append(stability_score(R, T, F, N))

    drift = compute_norm_drift(R0, T0, F0, N0, R, T, F, N)
    recovery = recovery_metric(history)
    
    # ---------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------
    plotter = Phase4Plotter("results/phase4")
    plotter.plot_stability_history(history)
    plotter.plot_drift(drift)
    plotter.plot_field_snapshots(R0, T0, F0, N0, R, T, F, N)

    if R_before is not None and R_after is not None:
        plotter.plot_perturbation_response(R_before, R_after, R)
    if R_before is not None and R_after is not None:
        # Extra diagnostics
        plotter.plot_perturbation_footprint(R_before, R_after)

        if len(R_recovery_traj) > 0:
            plotter.plot_recovery_trajectory(R_recovery_traj, R_before)

        if (
            T_before is not None and T_after is not None and
            F_before is not None and F_after is not None and
            N_before is not None and N_after is not None
        ):
            plotter.plot_perturbation_multifield(
                R_before, T_before, F_before, N_before,
                R_after, T_after, F_after, N_after
            )    
    return R, T, F, N, x, drift, recovery, history
