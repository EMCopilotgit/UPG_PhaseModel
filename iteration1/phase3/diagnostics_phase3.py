# phase3/diagnostics_phase3.py

import os
import json
import numpy as np

from iteration1.shared.plotting import plot_time_series


def summarize_result(r):
    return {
        "x_in": r["x_in"],
        "target": r["target"],
        "final_y_em": r["final_y_em"],
        "final_mae": r["final_mae"],
        "final_unified_N": r["final_unified_N"],
        "mean_error": float(np.mean(np.abs(r["logs"]["eps"]))),
        "mean_unified_N": float(np.mean(r["logs"]["N"]))
    }



def save_phase3_results(all_results, results_dir="results/phase3", tag="phase3"):
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Save per-input logs for each phase
    # ---------------------------------------------------------
    for phase_name in ["sin1", "cos", "sin2"]:
        for idx, r in enumerate(all_results[phase_name]):
            logs = r["logs"]

            # Convert ragged lists to object arrays
            safe_logs = {}
            for k, v in logs.items():
                if isinstance(v, list):
                    safe_logs[k] = np.array(v, dtype=object)
                else:
                    safe_logs[k] = v

            outpath = os.path.join(results_dir, f"{tag}_{phase_name}_{idx}.npz")

            np.savez_compressed(
                outpath,
                x_in=r["x_in"],
                target=r["target"],
                final_y_em=r["final_y_em"],
                final_mae=r["final_mae"],
                final_unified_N=r["final_unified_N"],
                **safe_logs
            )

    # ---------------------------------------------------------
    # 2. Build JSON summary (including defect summary)
    # ---------------------------------------------------------
    summary = {
        "sin1": [summarize_result(r) for r in all_results["sin1"]],
        "cos":  [summarize_result(r) for r in all_results["cos"]],
        "sin2": [summarize_result(r) for r in all_results["sin2"]],
        "defects": {
            phase_name: [
                {
                    "x_in": r["x_in"],
                    "defect_count": sum(len(p) for p in r["logs"]["defect_positions"]),
                    "first_defects": r["logs"]["defect_positions"][0].tolist(),
                    "last_defects":  r["logs"]["defect_positions"][-1].tolist(),
                }
                for r in all_results[phase_name]
            ]
            for phase_name in ["sin1", "cos", "sin2"]
        }
    }

    # ---------------------------------------------------------
    # 3. Save JSON summary
    # ---------------------------------------------------------
    with open(os.path.join(results_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

def plot_phase3_single(result, phase_name, results_dir="results/phase3"):
    os.makedirs(results_dir, exist_ok=True)

    fig = plot_time_series(result["logs"], title=f"Phase III: {phase_name}, x={result['x_in']:.3f}")

    x_str = f"{result['x_in']:.3f}".replace(".", "p").replace("-", "m")
    filename = f"{phase_name}_x{x_str}.png"

    fig.savefig(os.path.join(results_dir, filename), dpi=200)
