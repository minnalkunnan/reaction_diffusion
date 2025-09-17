from parameters import params, N, steps, dt, dx, save_every, spike_value
from simulation import run_coupled_neumann
from visualize import animate_histories


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
        N, steps, dt, dx, p,
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
        N, steps, dt, dx, p,
        init_mode="activator_spike",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: activator decay only")
    animate_histories(A_hist, R_hist, save_every, title="Activator decay-only (Neumann)")

def test_activator_propagation_only():
    """
    Propagation test: activator spike with production and decay, but no inhibitor production.
    Expect wave propagation toward steady-state value.
    """
    p = params.copy()
    p["act_prod_rate"] = 1.0
    p["inh_prod_rate"] = 0.0

    A_hist, R_hist = run_coupled_neumann(
        N, steps, dt, dx, p,
        init_mode="activator_spike",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: activator decay only")
    animate_histories(A_hist, R_hist, save_every, title="Activator decay-only (Neumann)")

if __name__ == "__main__":
    #test_inhibitor_diffusion_only()
    #test_activator_decay_only()
    test_activator_propagation_only()
