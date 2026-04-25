# phase2/diagnostics_phase2.py

import os
import json
import numpy as np

from iteration1.shared.plotting import plot_time_series


def summarize_phase2_result(result):
    """
    Summarize a single input's learning run.
    """
    return {
        "x_in": result["x_in"],
        "target": result["target"],
        "final_y_em": result["final_y_em"],
        "final_mae": result["final_mae"],
        "final_unified_N": result["final_unified_N"],
        "mean_error": float(np.mean(np.abs(result["logs"]["eps"]))),
        "mean_unified_N": float(np.mean(result["logs"]["N"]))
    }


def save_phase2_results(results, results_dir="results/phase2", tag="run2"):
    os.makedirs(results_dir, exist_ok=True)

    # save summary for all inputs
    summary = [summarize_phase2_result(r) for r in results]
    summary_path = os.path.join(results_dir, f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # save logs for each input
    for i, r in enumerate(results):
        logs_path = os.path.join(results_dir, f"logs_{tag}_{i}.npz")
        np.savez_compressed(logs_path, **r["logs"])


def plot_phase2(result, results_dir="results/phase2", tag="plot"):
    """
    Plot the time series for a single input run.
    """
    os.makedirs(results_dir, exist_ok=True)
    fig = plot_time_series(result["logs"], title=f"Phase II: x_in={result['x_in']}")
    fig.savefig(os.path.join(results_dir, f"{tag}_x{result['x_in']}.png"), dpi=200)
