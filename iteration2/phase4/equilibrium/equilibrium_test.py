import numpy as np

def is_equilibrated(observable, window, tol):
    """
    Relative-variance equilibrium test.
    """
    if len(observable) < window:
        return False

    tail = observable[-window:]
    mean = np.mean(tail)
    std = np.std(tail)

    if abs(mean) < 1e-12:
        return std < tol

    return (std / abs(mean)) < tol


def compute_T_star(observable, window, tol):
    """
    First time equilibrium is reached and stays reached.
    """
    T = len(observable)
    for t in range(window, T + 1):
        if is_equilibrated(observable[:t], window, tol):
            return t - 1
    return None
