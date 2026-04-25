def init_logs():
    return {
        "y_em": [],
        "eps": [],
        "N": [],
        "R_mean": [],
        "T_mean": [],
        "F_mean": []
    }

def log_step(logs, y_em, eps, N_unified, R, T, F):
    logs["y_em"].append(y_em)
    logs["eps"].append(eps)
    logs["N"].append(N_unified)
    logs["R_mean"].append(abs(R).mean())
    logs["T_mean"].append(T.mean())
    logs["F_mean"].append(abs(F).mean())
