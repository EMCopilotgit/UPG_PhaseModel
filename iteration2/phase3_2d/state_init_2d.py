import numpy as np


def init_random_state_2d(Nx, Ny):
    """
    Initialize a 2D S^2 vector field n[i,j] on a torus.
    """
    # Gaussian random, then normalize
    n = np.random.randn(Nx, Ny, 3)
    norms = np.linalg.norm(n, axis=2, keepdims=True)
    n /= norms
    return n
