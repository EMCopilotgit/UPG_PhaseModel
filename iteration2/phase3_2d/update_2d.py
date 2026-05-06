import numpy as np


def laplacian_2d(n):
    """
    2D discrete Laplacian on a periodic grid for S^2 field n[i,j,3].
    """
    n_ip = np.roll(n, -1, axis=0)
    n_im = np.roll(n,  1, axis=0)
    n_jp = np.roll(n, -1, axis=1)
    n_jm = np.roll(n,  1, axis=1)
    return n_ip + n_im + n_jp + n_jm - 4.0 * n


def update_n_2d(n, dt=0.01, kappa=0.2):
    """
    Simple geometric diffusion on S^2:
        n_t = kappa * Δn
    with renormalization back to S^2.
    """
    lap = laplacian_2d(n)
    n_new = n + dt * kappa * lap

    norms = np.linalg.norm(n_new, axis=2, keepdims=True)
    n_new /= norms
    return n_new
