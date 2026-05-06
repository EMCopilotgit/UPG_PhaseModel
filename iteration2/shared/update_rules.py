import numpy as np

def spatial_derivative(v):
    """
    Simple periodic finite difference derivative.
    v: (N, 3)
    """
    return np.roll(v, -1, axis=0) - v

def update_state(x, n, params, target_value):
    """
    Update x-field and compute unified geometric quantities.
    """
    N = len(x)
    dt = params.dt

    # --- unified geometric quantities ---
    dn = spatial_derivative(n)          # (N,3)
    curvature = np.linalg.norm(dn, axis=1)  # scalar curvature-like
    torsion = np.cross(n, dn)           # vector torsion-like

    # unified magnitude
    N_unified = curvature + params.gamma * np.linalg.norm(torsion, axis=1)

    # --- prediction ---
    y_em = float(np.mean(x))
    eps = target_value - y_em

    # --- update x-field ---
    x_new = x + dt * (params.beta * N_unified + eps)
    if params.noise_x > 0:
        x_new += params.noise_x * np.random.randn(N)

    return (
        x_new,
        y_em,
        eps,
        N_unified,
        curvature,
        torsion,
        n.copy()
    )
