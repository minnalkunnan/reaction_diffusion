from parameters import params, N, steps, dt, dx, save_every, spike_value, stopping_threshold
from simulation import run_coupled_neumann
from visualize import animate_histories


def main():
    # Run baseline simulation (two activator spikes, Neumann BC)
    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, params, stopping_threshold,
        init_mode="activator_spike",
        activator_type="membrane-tethered",
        spike_value=spike_value,
        save_every=save_every,
    )

    # Visualize results
    animate_histories(A_hist, R_hist, save_every, title="Baseline Simulation (Neumann)")


if __name__ == "__main__":
    main()
