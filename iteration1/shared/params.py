# phase1/params.py

class Params:
    def __init__(
        self,
        beta=0.1,        # curvature smoothing
        gamma=0.05,      # global error gain
        eta_T=0.02,      # torsion coupling
        eta_F=0.02,      # phase-field coupling
        alpha_th=0.1,    # U(1) phase coupling
        omega_th=0.01,
        kappa_th=0.2,
        gamma_th=0.05,
        noise_x=0.0,     # state noise
        noise_th=0.1,    # phase noise
        w_R=1.0,         # curvature weight
        w_T=1.0,         # torsion weight
        w_F=1.0,          # phase curvature weight
        mu=0.02,      # double-well strength
        v=1.0,       # preferred magnitude (vacuum value)
        kappa=0.1,   # spatial coupling strength
        alpha_x=0.05, # phase drive into x
        dt = 0.1
    ):
        self.beta = beta
        self.gamma = gamma
        self.eta_T = eta_T
        self.eta_F = eta_F
        self.alpha_th = alpha_th
        self.omega_th = omega_th
        self.kappa_th = kappa_th
        self.gamma_th = gamma_th     
        self.noise_x = noise_x
        self.noise_th = noise_th

        # unified energy weights
        self.w_R = w_R
        self.w_T = w_T
        self.w_F = w_F

        self.mu = mu
        self.v = v
        self.kappa = kappa
        self.alpha_x = alpha_x

        self.dt = dt
