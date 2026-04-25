def default_params_phase3():
    return Params(
        beta=0.1,
        gamma=0.25,
        eta_T=0.02,
        eta_F=0.02,
        alpha_th=0.1,
        noise_x=0.0,
        noise_th=0.0,
        w_R=1.0,
        w_T=1.0,
        w_F=1.0,
        mu=0.02,        # double-well strength
        v=1.0,          # target |x| ~ 1
        kappa=0.1,      # moderate spatial coupling
        alpha_x=0.05,   # gentle phase drive
    )
