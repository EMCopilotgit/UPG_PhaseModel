import numpy as np
from phase4.run_phase4 import run_phase4
from phase4.default_params_phase4 import default_params_phase4

def main():
    data = np.load("phase3_output.npz")
    R, T, F, N, x = data["R"], data["T"], data["F"], data["N"], data["x"]

    p = default_params_phase4()

    R, T, F, N, x, drift, recovery, history = run_phase4(R, T, F, N, x, p)
    
    from .save_phase4 import save_phase4_results
    save_phase4_results(R, T, F, N, x, drift, recovery, history)
    np.savez("phase4_output.npz",
             R=R, T=T, F=F, N=N, x=x,
             drift=drift, recovery=recovery,
             history=np.array(history))

    print("Phase IV complete.")
    print("Norm drift:", drift)
    print("Recovery metric:", recovery)

if __name__ == "__main__":
    main()
