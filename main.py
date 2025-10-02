import argparse
import os
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
    parser.add_argument("--movie", action="store_true",
                        help="If set, also save an MP4 movie using the same basename as --output.")
    args = parser.parse_args()

    # If movie is requested and user did not explicitly set --vis, turn off visualization
    if args.movie and args.vis is True and "--vis" not in " ".join(os.sys.argv):
        args.vis = False

    #Define initiation mode and activator type
    init_mode = "activator_spike"
    activator_type = "membrane-tethered"

    # Run baseline simulation (two activator spikes, Neumann BC)
    A_hist, R_hist, final_step = run_coupled_neumann(
        N, steps, dt, dx, params, stopping_threshold,
        init_mode=init_mode,
        activator_type=activator_type,
        spike_value=spike_value,
        save_every=save_every,
    )

    # If --output is provided, save results to file + static plot
    if args.output:
        outfile_txt, outfile_png = write_simulation_results(
            args, activator_type, init_mode, spike_value, params, A_hist, R_hist, final_step
        )

        # Save static plot
        plot_last_frame(A_hist[-1], R_hist[-1], final_step, outfile_png)
        print(f"Final state plot saved to {outfile_png}")

    # Visualization and/or movie saving
    if args.vis or args.movie:
        outdir = "simulation_results"
        movie_path = None

        # If movie flag is set, require --output
        if args.movie:
            if not args.output:
                raise ValueError("--movie requires --output to be specified.")
            os.makedirs(outdir, exist_ok=True)
            movie_path = os.path.join(outdir, args.output + ".mp4")

        animate_histories(A_hist, R_hist, save_every,
                          title="Baseline Simulation (Neumann)",
                          loop=False,
                          savefile=movie_path)

if __name__ == "__main__":
    main()
