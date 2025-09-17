import numpy as np


def hill_function(act_signal, inh_signal,
                  act_half_sat, inh_half_sat,
                  act_hill_coeff, inh_hill_coeff, basal_prod):
    """Hill-type activation/repression function."""
    act_term = (act_signal / act_half_sat) ** act_hill_coeff if act_signal > 0 else 0.0
    inh_term = (inh_signal / inh_half_sat) ** inh_hill_coeff if inh_signal > 0 else 0.0
    return (act_term + basal_prod) / (act_term + inh_term + 1.0 + basal_prod)


def initialize_fields(N, init_mode, spike_value):
    """Initialize activator/inhibitor concentrations depending on mode."""
    activator = np.zeros(N)
    inhibitor = np.zeros(N)

    if init_mode == "spikes":          # two activator spikes
        activator[5] = spike_value
        activator[85] = spike_value
    elif init_mode == "activator_spike":   # single activator spike
        activator[N // 2] = spike_value
    elif init_mode == "inhibitor_spike":   # single inhibitor spike
        inhibitor[N // 2] = spike_value
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    return activator, inhibitor


def update_interior(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p):
    """Update interior grid points (1 .. N-2)."""
    for i in range(1, N - 1):
        act_signal = 0.5 * (activator[i - 1] + activator[i + 1])
        inh_signal = inhibitor[i]

        activator_new[i] = activator[i] + dt * (
            p["act_prod_rate"] * hill_function(
                act_signal, inh_signal,
                p["act_half_sat"], p["inh_half_sat"],
                p["act_hill_coeff"], p["inh_hill_coeff"],
                p["basal_prod"]
            )
            - p["act_decay_rate"] * activator[i]
        )
        inhibitor_new[i] = (
            inhibitor[i]
            + dt * (p["inh_prod_rate"] * hill_function(
                act_signal, inh_signal,
                p["act_half_sat"], p["inh_half_sat"],
                p["act_hill_coeff"], p["inh_hill_coeff"],
                p["basal_prod"]
            )
            - p["inh_decay_rate"] * inhibitor[i])
            + p["inh_diffusion"] * dt / dx**2 * (inhibitor[i + 1] - 2.0 * inhibitor[i] + inhibitor[i - 1])
        )


def update_boundaries(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p):
    """Update Neumann boundary conditions (zero-flux)."""
    # Left boundary
    idx, left = 0, 1
    act_signal = activator[left]
    inh_signal = inhibitor[idx]
    activator_new[idx] = activator[idx] + dt * (
        p["act_prod_rate"] * hill_function(
            act_signal, inh_signal,
            p["act_half_sat"], p["inh_half_sat"],
            p["act_hill_coeff"], p["inh_hill_coeff"],
            p["basal_prod"]
        )
        - p["act_decay_rate"] * activator[idx]
    )
    inhibitor_new[idx] = (
        inhibitor[idx]
        + dt * (p["inh_prod_rate"] * hill_function(
            act_signal, inh_signal,
            p["act_half_sat"], p["inh_half_sat"],
            p["act_hill_coeff"], p["inh_hill_coeff"],
            p["basal_prod"]
        )
        - p["inh_decay_rate"] * inhibitor[idx])
        + p["inh_diffusion"] * dt / dx**2 * (2.0 * (inhibitor[left] - inhibitor[idx]))
    )

    # Right boundary
    idx, right = N - 1, N - 2
    act_signal = activator[right]
    inh_signal = inhibitor[idx]
    activator_new[idx] = activator[idx] + dt * (
        p["act_prod_rate"] * hill_function(
            act_signal, inh_signal,
            p["act_half_sat"], p["inh_half_sat"],
            p["act_hill_coeff"], p["inh_hill_coeff"],
            p["basal_prod"]
        )
        - p["act_decay_rate"] * activator[idx]
    )
    inhibitor_new[idx] = (
        inhibitor[idx]
        + dt * (p["inh_prod_rate"] * hill_function(
            act_signal, inh_signal,
            p["act_half_sat"], p["inh_half_sat"],
            p["act_hill_coeff"], p["inh_hill_coeff"],
            p["basal_prod"]
        )
        - p["inh_decay_rate"] * inhibitor[idx])
        + p["inh_diffusion"] * dt / dx**2 * (2.0 * (inhibitor[right] - inhibitor[idx]))
    )


def run_coupled_neumann(
    N, steps, dt, dx, p,
    init_mode="spikes",
    spike_value=5.0,
    save_every=10
):
    """Run activator–inhibitor simulation with Neumann boundary conditions."""
    activator, inhibitor = initialize_fields(N, init_mode, spike_value)

    activator_history = [activator.copy()]
    inhibitor_history = [inhibitor.copy()]

    for step in range(steps):
        activator_new = activator.copy()
        inhibitor_new = inhibitor.copy()

        update_interior(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p)
        update_boundaries(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p)

        # Enforce non-negativity
        activator_new = np.maximum(activator_new, 0.0)
        inhibitor_new = np.maximum(inhibitor_new, 0.0)

        activator, inhibitor = activator_new, inhibitor_new

        if step % save_every == 0:
            activator_history.append(activator.copy())
            inhibitor_history.append(inhibitor.copy())

    return activator_history, inhibitor_history
