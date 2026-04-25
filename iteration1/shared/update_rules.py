import numpy as np
from .invariants import compute_invariants, neighbors


def update_state(x, theta, params, target_value):
    """
    Update the state x given phase theta and target_value.

    Now includes:
      - double-well potential in x (φ^4-like)
      - spatial Laplacian coupling
      - phase-driven forcing
      - existing learning terms (R, T, F, eps)
    so that N_field / N_unified can become signed, oscillatory,
    and defect-supporting.
    """

    # --- invariants and unified field ---
    R, T, F, E, N_unified, N_field = compute_invariants(
        x, theta,
        w_R=params.w_R,
        w_T=params.w_T,
        w_F=params.w_F,
    )

    # emergent output + error
    y_em = x.mean()
    eps = target_value - y_em

    N = len(x)
    x_new = np.zeros_like(x)

    for i in range(N):
        L, Rn = neighbors(i, N)

        # --- spatial Laplacian (diffusion / coupling) ---
        lap = x[L] - 2.0 * x[i] + x[Rn]

        # --- double-well potential (φ^4) ---
        dw = params.mu * (params.v**2 - x[i]**2) * x[i]

        # --- phase drive ---
        phase_drive = params.alpha_x * np.sin(theta[i])

        # --- learning / invariant-driven terms ---
        learn = (
            params.beta * R[i] +
            params.gamma * eps +
            params.eta_T * T[i] +
            params.eta_F * F[i]
        )

        # --- stable update with explicit timestep ---
        dx = (
            params.kappa * lap +
            dw +
            phase_drive +
            learn
        )

        x_new[i] = x[i] + params.dt * dx

        # --- clamp to avoid runaway divergence ---
        x_new[i] = np.clip(x_new[i], -5.0, 5.0)

    # --- optional noise (applied after the loop) ---
    if params.noise_x != 0.0:
        x_new += params.noise_x * np.random.randn(N)

    return x_new, y_em, eps, N_unified, R, T, F, E, N_field
    
    for i in range(N):
        L, Rn = neighbors(i, N)

        # --- spatial Laplacian (diffusion / coupling) ---
        lap = x[L] - 2.0 * x[i] + x[Rn]

        # --- double-well potential (φ^4) ---
        # pushes x toward ±v, enabling sign changes and kinks
        dw = params.mu * (params.v**2 - x[i]**2) * x[i]

        # --- phase drive into x ---
        # couples unified field to phase, enabling oscillations
        phase_drive = params.alpha_x * np.sin(theta[i])

        # --- learning / invariant-driven terms (existing structure) ---
        learn = (
            params.beta * R[i] +
            params.gamma * eps +
            params.eta_T * T[i] +
            params.eta_F * F[i]
        )

    # Stable update with explicit timestep
    dx = (
        params.kappa * lap
        + params.mu * (params.v**2 - x[i]**2) * x[i]   # φ⁴ force
        + phase_drive
        + learn
    )

    # Apply timestep
    x_new[i] = x[i] + params.dt * dx

    # Optional but highly recommended: clamp to avoid runaway divergence
    x_new[i] = np.clip(x_new[i], -5.0, 5.0)
    
    # optional noise
    if params.noise_x != 0.0:
        x_new += params.noise_x * np.random.randn(N)

    return x_new, y_em, eps, N_unified, R, T, F, E, N_field
