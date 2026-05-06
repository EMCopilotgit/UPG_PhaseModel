import numpy as np

def update_phase(n, params, eps_field=None):
    """
    Update the S^2 vector-phase field.
    n[i] is a 3-component unit vector.

    eps_field: scalar error field (N,)
    """
    dt = params.dt

    if eps_field is None:
        eps_field = np.zeros(n.shape[0])

    # local "force" proportional to error
    F = params.alpha_n * eps_field[:, None] * n

    # noise
    if params.noise_n > 0:
        F += params.noise_n * np.random.randn(*n.shape)

    # update
    n_new = n + dt * F

    # renormalize to S^2
    n_new /= np.linalg.norm(n_new, axis=1, keepdims=True)
    return n_new
