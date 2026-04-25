import numpy as np

def neighbors(i, N):
    return (i - 1) % N, (i + 1) % N

def compute_invariants(x, theta, w_R=1.0, w_T=1.0, w_F=1.0):
    N = len(x)
    R = np.zeros(N)
    T = np.zeros(N)
    F = np.zeros(N)

    for i in range(N):
        L, Rn = neighbors(i, N)

        # Scalar curvature (Laplacian)
        R[i] = x[L] + x[Rn] - 2 * x[i]

        # Torsion proxy (asymmetry)
        T[i] = ((x[Rn] - x[i]) - (x[i] - x[L]))

        # U(1) phase curvature (field strength)
        F[i] = (theta[Rn] - theta[i]) - (theta[i] - theta[L])

    # Unified energy density
    E = w_R * R**2 + w_T * T**2 + w_F * F**2
    N_unified = E.mean()
    N_field = x
    return R, T, F, E, N_unified, N_field
