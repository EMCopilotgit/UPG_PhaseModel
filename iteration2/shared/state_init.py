import numpy as np

def random_unit_vectors(N):
    v = np.random.randn(N, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def init_state_for_input(x_in, N):
    """
    Initialize x-field and S^2 vector-phase field.
    """
    x = np.full(N, float(x_in))
    n = random_unit_vectors(N)
    return x, n

def init_random_state(N):
    """
    Random x-field + random S^2 phase.
    """
    x = np.random.uniform(-1, 1, size=N)
    n = random_unit_vectors(N)
    return x, n
