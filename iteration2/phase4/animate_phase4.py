import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DEFAULT_DIR = os.path.join("iteration2", "results", "phase4")

def animate_w_evolution(w_history, savepath=None, fps=2):
    """
    w_history: list of 2D arrays, one per episode
    """
    if savepath is None:
        savepath = os.path.join(DEFAULT_DIR, "w_evolution.gif")

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(w_history[0], cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(w_history[frame])
        ax.set_title(f"w(x,y) — episode {frame}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(w_history), interval=1000/fps)
    anim.save(savepath, writer=PillowWriter(fps=fps))
    plt.close()
