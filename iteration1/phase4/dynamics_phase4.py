import numpy as np
from numba import njit
from iteration1.shared.params import Params   # <-- REQUIRED

def inject_noise(R, T, F, N, p):
    amp = p.noise_amp
    R += amp * (np.random.rand(*R.shape) - 0.5)
    T += amp * (np.random.rand(*T.shape) - 0.5)
    F += amp * (np.random.rand(*F.shape) - 0.5)
    N += amp * (np.random.rand(*N.shape) - 0.5)
    return R, T, F, N

def apply_drift(x, p):
    rate = p.drift_rate
    return x + rate * (np.random.rand(*x.shape) - 0.5)

def localized_perturbation(R, T, F, N, x, p):
    c = p.shock_center
    r = p.shock_radius
    s = p.shock_strength

    for i in range(R.shape[0]):
        if abs(x[i] - c) < r:
            R[i] += s * (np.random.rand() - 0.5)
            T[i] += s * (np.random.rand() - 0.5)
            F[i] += s * (np.random.rand() - 0.5)
            N[i] += s * (np.random.rand() - 0.5)
    return R, T, F, N

def recovery_step(R, T, F, N, p):
    dt = p.dt
    R += dt * (-0.1 * R + 0.05 * T)
    T += dt * (-0.1 * T + 0.05 * F)
    F += dt * (-0.1 * F + 0.05 * N)
    N += dt * (-0.1 * N + 0.05 * R)
    return R, T, F, N
