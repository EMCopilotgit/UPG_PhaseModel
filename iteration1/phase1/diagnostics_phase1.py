import os
import json
import numpy as np

from iteration1.shared.plotting import plot_time_series

def summarize_phase1(logs):
    final_y = logs["y_em"][-1]
    final_err = logs["eps"][-1]
    final_N = logs["N"][-1]

    summary = {
        "final_y_em": float(final_y),
        "final_error": float(final_err),
        "final_unified_N": float(final_N),
        "mean_error": float(np.mean(np.abs(logs["eps"]))),
        "mean_unified_N": float(np.mean(logs["N"]))
    }
    return summary

def save_phase1_results(logs, summary, results_dir="results/phase1", tag="run1"):
    os.makedirs(results_dir, exist_ok=True)

    # save summary
    summary_path = os.path.join(results_dir, f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # optional: save raw logs
    logs_path = os.path.join(results_dir, f"logs_{tag}.npz")
    np.savez_compressed(logs_path, **logs)

def plot_phase1(logs, title="Phase I Dynamics"):
    plot_time_series(logs, title=title)
