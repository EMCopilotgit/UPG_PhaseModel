import os
import json
import numpy as np

def save_phase4_results(R, T, F, N, x, drift, recovery, history, results_dir="results/phase4"):
    os.makedirs(results_dir, exist_ok=True)

    # Save fields
    np.savez_compressed(
        os.path.join(results_dir, "final_fields.npz"),
        R=R, T=T, F=F, N=N, x=x
    )

    # Save metrics
    summary = {
        "drift": [float(d) for d in drift],
        "recovery": float(recovery),
        "history": [float(h) for h in history]
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
