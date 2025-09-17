import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_histories(A_hist, R_hist, save_every, title="Coupled Dynamics (Neumann)"):
    """Animate activator and inhibitor histories side by side."""
    num_frames = len(A_hist)

    fig, ax = plt.subplots()
    line_A, = ax.plot(A_hist[0], "--", color="red", label="Activator")
    line_R, = ax.plot(R_hist[0], "--", color="blue", label="Inhibitor")

    ax.set_ylim(0, max(1e-6, max(max(A_hist[0]), max(R_hist[0])) * 1.2))
    ax.set_title(title)
    ax.set_xlabel("Space (Cell Index)")
    ax.set_ylabel("Concentration")
    ax.legend(loc="upper right", fontsize=9)

    def update(frame):
        line_A.set_ydata(A_hist[frame])
        line_R.set_ydata(R_hist[frame])
        max_val = max(max(A_hist[frame]), max(R_hist[frame]))
        ax.set_ylim(0, max(1e-6, max_val * 1.2))
        ax.set_title(f"{title}\nStep {frame * save_every}")
        return [line_A, line_R]

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    plt.tight_layout()
    plt.show()
