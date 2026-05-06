import numpy as np

class Params:
    """
    Iteration 2 parameter container.
    Vector-phase model on S^2.
    """
    def __init__(
        self,
        alpha_n=0.1,      # phase update strength
        beta=0.0,         # curvature coupling
        gamma=0.1,        # torsion coupling
        kappa=0.1,        # spatial coupling strength
        noise_n=0.0,      # noise on vector phase
        noise_x=0.01,     # noise on x-field
        dt=0.01           # timestep
    ):
        self.alpha_n = alpha_n
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        self.noise_n = noise_n
        self.noise_x = noise_x
        self.dt = dt
