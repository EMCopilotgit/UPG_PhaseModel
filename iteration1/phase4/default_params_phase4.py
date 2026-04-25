from iteration1.shared.params import Params

def default_params_phase4():
    p = Params()

    # noise + drift
    p.noise_amp = 0.01
    p.drift_rate = 0.001

    # shock parameters
    p.shock_center = 0.0
    p.shock_radius = 0.2
    p.shock_strength = 0.1

    # integration
    p.steps = 2000
    p.dt = 0.01

    return p
