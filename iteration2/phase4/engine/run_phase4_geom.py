import numpy as np

from iteration2.phase3_2d.state_init_2d import init_random_state_2d
from iteration2.phase3_2d.defects_2d import plaquette_charge

from .state_phase4 import init_plasticity_field
from .coupled_update_2d import update_n_2d_coupled
from .plasticity import update_plasticity_global


def run_phase4_geom(
    episodes=20,
    T_steps=200,
    Nx=64,
    Ny=64,
    dt=0.01,
    kappa0=0.2,
    alpha=0.1,
    eta=0.01,
    lam=0.01,
):
    """
    Phase IV — geometric continual learning engine.
    """
    w = init_plasticity_field(Nx, Ny, w0=0.0)
    w_history = [w.copy()]
    logs = {"episodes": []}

    for ep in range(episodes):
        n = init_random_state_2d(Nx, Ny)

        # Phase III evolution with spatially varying kappa
        for t in range(T_steps):
            n = update_n_2d_coupled(n, w, dt=dt, kappa0=kappa0, alpha=alpha)

        # Topological diagnostics
        q_final = plaquette_charge(n)
        Q_total = float(np.sum(q_final))

        # Plasticity update
        w, q = update_plasticity_global(w, n, Q_total, eta=eta, lam=lam)

        w_history.append(w.copy())

        logs["episodes"].append(
            {
                "episode": ep,
                "Q_total": Q_total,
                "q_abs_mean": float(np.mean(np.abs(q_final))),
                "w_mean": float(np.mean(w)),
                "w_std": float(np.std(w)),
            }
        )

        print(
            f"ep {ep+1}/{episodes}: "
            f"Q_total={Q_total:.3f}, "
            f"<|q|>={np.mean(np.abs(q_final)):.4f}, "
            f"<w>={np.mean(w):.4f}, std(w)={np.std(w):.4f}"
        )

    final_state = {"w": w, "n": n}
    return logs, final_state, w_history
