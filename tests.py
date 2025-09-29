from parameters import params, N, steps, dt, dx, save_every, spike_value, stopping_threshold
from simulation import run_coupled_neumann
from visualize import animate_histories
import argparse


def test_inhibitor_diffusion_only():
    """
    Pure diffusion test: inhibitor spike with no production/decay.
    Expect flattening over time without vanishing.
    """
    p = params.copy()
    p["inh_prod_rate"] = 0.0
    p["inh_decay_rate"] = 0.0
    p["act_prod_rate"] = 0.0
    p["act_decay_rate"] = 0.0

    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="inhibitor_spike",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: inhibitor diffusion only")
    animate_histories(A_hist, R_hist, save_every, title="Inhibitor diffusion-only (Neumann)")


def test_activator_decay_only():
    """
    Pure decay test: activator spike with no production.
    Expect exponential decay toward zero.
    """
    p = params.copy()
    p["act_prod_rate"] = 0.0
    p["inh_prod_rate"] = 0.0
    p["inh_decay_rate"] = 0.0

    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="activator_spike",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: activator decay only")
    animate_histories(A_hist, R_hist, save_every, title="Activator decay-only (Neumann)")

def test_activator_no_diffusion():
    """
    Pure decay test: activator spike with no production.
    Expect exponential decay toward zero.
    """
    p = params.copy()
    p["act_prod_rate"] = 0.0
    p["act_decay_rate"] = 0.0
    p["inh_prod_rate"] = 0.0
    p["inh_decay_rate"] = 0.0

    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="activator_spike",
        activator_type="membrane-tethered",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: activator no diffusion (membrane)")
    animate_histories(A_hist, R_hist, save_every, title="Activator decay-only (Neumann)")

def test_activator_with_diffusion():
    """
    Pure decay test: activator spike with no production.
    Expect exponential decay toward zero.
    """
    p = params.copy()
    p["act_prod_rate"] = 0.0
    p["inh_prod_rate"] = 0.0
    p["inh_decay_rate"] = 0.0
    p["act_decay_rate"] = 0.0

    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="activator_spike",
        activator_type="soluble",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: activator diffusion (membrane)")
    animate_histories(A_hist, R_hist, save_every, title="Activator decay-only (Neumann)")

def test_activator_propagation_only():
    """
    Propagation test: activator spike with production and decay, but no inhibitor production.
    Expect wave propagation toward steady-state value.
    """
    p = params.copy()
    p["act_prod_rate"] = 3.0
    p["inh_prod_rate"] = 0.0

    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="activator_spike",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: activator decay only")
    animate_histories(A_hist, R_hist, save_every, title="Activator decay-only (Neumann)")

def main():
    parser = argparse.ArgumentParser(description="Run specific test cases.")
    parser.add_argument(
        "test",
        choices=[
            "inhibitor_diffusion_only",
            "activator_no_diffusion",
            "activator_with_diffusion",
            "activator_decay_only",
            "activator_propagation_only",
        ],
        help="Choose which test to run"
    )
    args = parser.parse_args()

    if args.test == "inhibitor_diffusion_only":
        test_inhibitor_diffusion_only()
    elif args.test == "activator_no_diffusion":
        test_activator_no_diffusion()
    elif args.test == "activator_with_diffusion":
        test_activator_with_diffusion()
    elif args.test == "activator_decay_only":
        test_activator_decay_only()
    elif args.test == "activator_propagation_only":
        test_activator_propagation_only()

if __name__ == "__main__":
    main()
