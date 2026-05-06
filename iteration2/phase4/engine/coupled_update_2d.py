import numpy as np

def laplacian_2d_field(n):
    """
    2D discrete Laplacian on a periodic grid for S^2 field n[i,j,3].
    """
    n_ip = np.roll(n, -1, axis=0)
    n_im = np.roll(n,  1, axis=0)
    n_jp = np.roll(n, -1, axis=1)
    n_jm = np.roll(n,  1, axis=1)
    return n_ip + n_im + n_jp + n_jm - 4.0 * n


def update_n_2d_coupled(n, w, dt=0.01, kappa0=0.2, alpha=0.1):
    """
    Geometric diffusion with spatially varying kappa(x,y):
        kappa_ij = kappa0 + alpha * w_ij
    """
    lap = laplacian_2d_field(n)

    kappa_local = kappa0 + alpha * w
    kappa_local = kappa_local[..., None]

    n_new = n + dt * kappa_local * lap

    norms = np.linalg.norm(n_new, axis=2, keepdims=True)
    n_new /= norms
    return n_new
