import numpy as np

def extract_lambda_star(cfg, eq_flag):
    etas_eff = []
    lambda_star = []

    for i, eta in enumerate(cfg.etas):
        row = eq_flag[i, :]
        transitions = np.where((row[:-1] == 1) & (row[1:] == 0))[0]

        if len(transitions) > 0:
            k = transitions[0]
            lam_c = 0.5 * (cfg.lambdas[k] + cfg.lambdas[k + 1])
            etas_eff.append(eta)
            lambda_star.append(lam_c)

    return np.array(etas_eff), np.array(lambda_star)
