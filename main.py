import argparse
from parameters import params, N, steps, dt, dx, save_every, spike_value, stopping_threshold
from simulation import run_coupled_neumann
from visualize import animate_histories, plot_last_frame
from writing_simulation_results import str2bool, write_simulation_results


def main():
    parser = argparse.ArgumentParser(description="Run coupled Neumann simulation.")
    parser.add_argument("--output", type=str, help="Base name of output file (without extension). "
                                                   "Will save simulation_results/NAME.txt and NAME.png.")
    parser.add_argument("--vis", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to show visualization (True/False). Default = True.")
    args = parser.parse_args()

    #Define initiation mode and activator type
    init_mode = "activator_spike"
    activator_type = "membrane-tethered"

    # Run baseline simulation (two activator spikes, Neumann BC)
    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, params, stopping_threshold,
        init_mode=init_mode,
        activator_type=activator_type,
        spike_value=spike_value,
        save_every=save_every,
    )

    # Run visualization only if requested - looping can be turned on/off
    if args.vis:
        animate_histories(A_hist, R_hist, save_every, title="Baseline Simulation (Neumann)", loop=False)

    # If --output is provided, save results to file + static plot
    if args.output:
        outfile_txt, outfile_png = write_simulation_results(
            args, activator_type, init_mode, spike_value, params, A_hist, R_hist
        )

        # Save static plot
        plot_last_frame(A_hist[-1], R_hist[-1], outfile_png)
        print(f"Final state plot saved to {outfile_png}")

if __name__ == "__main__":
    main()
