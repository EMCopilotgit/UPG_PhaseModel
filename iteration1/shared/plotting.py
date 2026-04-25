import matplotlib.pyplot as plt

def plot_time_series(logs, title="Dynamics"):
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(logs["y_em"])
    axs[0].set_ylabel("y_em")

    axs[1].plot(logs["eps"])
    axs[1].set_ylabel("error")

    axs[2].plot(logs["N"])
    axs[2].set_ylabel("unified N")

    axs[3].plot(logs["R_mean"], label="R")
    axs[3].plot(logs["T_mean"], label="T")
    axs[3].plot(logs["F_mean"], label="F")
    axs[3].set_ylabel("geometry")
    axs[3].legend()

    plt.suptitle(title)
    plt.tight_layout()
    return fig
