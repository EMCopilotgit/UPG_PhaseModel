
import numpy as np
from .invariants import neighbors

def update_phase(theta, params, eps_field=None):
    """
    Revised phase update:
      - avoids trivial global synchronization
      - supports traveling / frustrated phase patterns
      - can couple to a local error field if provided
    """

    N = len(theta)
    th_new = np.zeros_like(theta)

    # global drift term (prevents collapse to a static fixed point)
    omega_th = getattr(params, "omega_th", 0.01)

    # frustration / preferred gradient (encourages nonzero phase differences)
    kappa_th = getattr(params, "kappa_th", 0.1)

    # error coupling strength (phase responds to learning signal)
    gamma_th = getattr(params, "gamma_th", 0.0)

    # if no local error field is given, use zero
    if eps_field is None:
        eps_field = np.zeros_like(theta)

    for i in range(N):
        L, Rn = neighbors(i, N)

        # local phase differences
        dL = theta[L] - theta[i]
        dR = theta[Rn] - theta[i]

        # frustrated coupling: prefers a nonzero gradient
        # (e.g. traveling wave / spiral-like structure in 1D)
        coupling = (
            np.sin(dL - kappa_th) +
            np.sin(dR + kappa_th)
        )

        # local error influence
        err_drive = gamma_th * eps_field[i]

        th_new[i] = theta[i] + params.alpha_th * coupling + omega_th + err_drive

    # phase noise to keep things from freezing
    th_new += params.noise_th * np.random.randn(N)

    # keep phases in a reasonable range (optional)
    th_new = (th_new + np.pi) % (2.0 * np.pi) - np.pi

    return th_new

