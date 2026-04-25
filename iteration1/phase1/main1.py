from iteration1.phase1.dynamics_phase1 import run_phase1, default_params_phase1
from iteration1.phase1.diagnostics_phase1 import summarize_phase1, save_phase1_results, plot_phase1

def main():
    # basic config
    T_steps = 500
    N = 64
    target_value = 0.5  # simple constant target for Phase I
    params = default_params_phase1()

    # run Phase I
    logs, final_state = run_phase1(
        T_steps=T_steps,
        N=N,
        target_value=target_value,
        params=params
    )

    # diagnostics
    summary = summarize_phase1(logs)
    print("Phase I summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # save + plot
    save_phase1_results(logs, summary, results_dir="results/phase1", tag="run1")
    plot_phase1(logs, title="Phase I: Unified Curvature Dynamics")

if __name__ == "__main__":
    main()
