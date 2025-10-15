import numpy as np
import random
from finding_steady_states import fast_stable_steady_state


def hill_function(act_signal, inh_signal,
                  act_half_sat, inh_half_sat,
                  act_hill_coeff, inh_hill_coeff, basal_prod):
    """Hill-type activation/repression function."""
    act_term = (act_signal / act_half_sat) ** act_hill_coeff if act_signal > 0 else 0.0
    inh_term = (inh_signal / inh_half_sat) ** inh_hill_coeff if inh_signal > 0 else 0.0
    return (act_term + basal_prod) / (act_term + inh_term + 1.0 + basal_prod)

def initialize_fields(N, init_mode, spike_value, spike_value_a = 0, spike_value_i = 0):
    """Initialize activator/inhibitor concentrations depending on mode."""
    activator = np.zeros(N)
    inhibitor = np.zeros(N)

    if init_mode == "random_tight": #5% random noise around steady state value of a and i
        activator = [random.uniform(spike_value_a - 0.05 * spike_value_a, spike_value_a + 0.05 * spike_value_a) for _ in range(N)]
        inhibitor = [random.uniform(spike_value_i - 0.05 * spike_value_i, spike_value_i + 0.05 * spike_value_i) for _ in range(N)]
    elif init_mode == "spike_steady_state": #Starts with calculated steady state value (no diffusion) at one point in space
        activator[N // 2] = spike_value_a
        inhibitor[N // 2] = spike_value_i
    elif init_mode == "activator_spike_steady_state":   # single activator spike
        activator[N // 2] = spike_value_a
    elif init_mode == "two_activator_spikes":          # two activator spikes
        activator[5] = spike_value
        activator[85] = spike_value
    elif init_mode == "activator_spike":   # single activator spike
        activator[N // 2] = spike_value
    elif init_mode == "activator_spike_with_background":   # single activator spike
        activator = [0.2] * N
        activator[N // 2] = spike_value
    elif init_mode == "both_spike":
        activator[N // 2] = spike_value
        inhibitor[N // 2] = spike_value
    elif init_mode == "inhibitor_spike":   # single inhibitor spike
        inhibitor[N // 2] = spike_value
    elif init_mode == "random":
        activator = [random.uniform(0, spike_value) for _ in range(N)]
        inhibitor = [random.uniform(0, spike_value) for _ in range(N)]
    elif init_mode == "activator_on":
        activator = [spike_value] * N
    elif init_mode == "inhibitor_on":
        inhibitor = [spike_value] * N
    elif init_mode == "both_on":
        activator = [spike_value] * N
        inhibitor = [spike_value] * N
    elif init_mode == "all_off":
        pass
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    # Ensure float arrays in all cases
    activator = np.array(activator, dtype=float)
    inhibitor = np.array(inhibitor, dtype=float)

    return activator, inhibitor


def update_interior(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p, activator_type):
    """Update interior grid points (1 .. N-2)."""
    for i in range(1, N - 1):
        #Set input activator value as self or as neighbours
        if activator_type == "soluble":
            act_signal = activator[i]
        else:
            act_signal = 0.5 * (activator[i - 1] + activator[i + 1])
        #set inhibitor value as self
        inh_signal = inhibitor[i]
        #calculate transcriptional reaction (Hill function)
        hill_value = hill_function(
            act_signal, inh_signal,
            p["act_half_sat"], p["inh_half_sat"],
            p["act_hill_coeff"], p["inh_hill_coeff"],
            p["basal_prod"]
        )
        #How much are we updating the activation concentration? Computing reaction and diffusion separately
        reaction = dt * (p["act_prod_rate"] * hill_value - p["act_decay_rate"] * activator[i])
        diffusion = (p["act_diffusion"] * dt / dx**2 * (activator[i + 1] - 2.0 * activator[i] + activator[i - 1])
             if activator_type == "soluble" else 0.0) #NO diffusion if activator is membrane-tethered
        #Actual update
        activator_new[i] = activator[i] + reaction + diffusion

        #inhibitor is always soluble, so we can compute all at once
        inhibitor_new[i] = (
            inhibitor[i]
            + dt * (p["inh_prod_rate"] * hill_value - p["inh_decay_rate"] * inhibitor[i]) #reaction
            + p["inh_diffusion"] * dt / dx**2 * (inhibitor[i + 1] - 2.0 * inhibitor[i] + inhibitor[i - 1]) #diffusion
        )


def update_boundaries(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p, activator_type):
    """Update Neumann boundary conditions (zero-flux)."""
    # Left boundary
    idx, left = 0, 1
    if activator_type == "soluble":
        act_signal = activator[idx]
    else:
        act_signal = activator[left]
    inh_signal = inhibitor[idx]

    #Calculate transcriptional reaction (Hill function)
    hill_value = hill_function(
        act_signal, inh_signal,
        p["act_half_sat"], p["inh_half_sat"],
        p["act_hill_coeff"], p["inh_hill_coeff"],
        p["basal_prod"]
    )

    #How much are we updating the activation concentration? Computing reaction and diffusion separately
    reaction = dt * (p["act_prod_rate"] * hill_value - p["act_decay_rate"] * activator[idx])
    diffusion = (p["act_diffusion"] * dt / dx**2 * (activator[left] - activator[idx])
         if activator_type == "soluble" else 0.0) #NO diffusion if activator is membrane-tethered
    #Actual update
    activator_new[idx] = activator[idx] + reaction + diffusion

    inhibitor_new[idx] = (
        inhibitor[idx] +
        dt * (p["inh_prod_rate"] * hill_value - p["inh_decay_rate"] * inhibitor[idx]) #reaction
        + p["inh_diffusion"] * dt / dx**2 * (inhibitor[left] - inhibitor[idx]) #diffusion
    )

    # Right boundary
    idx, right = N - 1, N - 2
    if activator_type == "soluble":
        act_signal = activator[idx]
    else:
        act_signal = activator[right]
    inh_signal = inhibitor[idx]

    #calculate transcriptional reaction (Hill function)
    hill_value = hill_function(
        act_signal, inh_signal,
        p["act_half_sat"], p["inh_half_sat"],
        p["act_hill_coeff"], p["inh_hill_coeff"],
        p["basal_prod"]
    )

    #How much are we updating the activation concentration? Computing reaction and diffusion separately
    reaction = dt * (p["act_prod_rate"] * hill_value - p["act_decay_rate"] * activator[idx])
    diffusion = (p["act_diffusion"] * dt / dx**2 * (activator[right] - activator[idx])
         if activator_type == "soluble" else 0.0) #NO diffusion if activator is membrane-tethered
    #Actual update
    activator_new[idx] = activator[idx] + reaction + diffusion

    inhibitor_new[idx] = (
        inhibitor[idx] +
        dt * (p["inh_prod_rate"] * hill_value - p["inh_decay_rate"] * inhibitor[idx]) #reaction
        + p["inh_diffusion"] * dt / dx**2 * (inhibitor[right] - inhibitor[idx]) #diffusion
    )


def run_coupled_neumann(
    N, steps, dt, dx, p, stopping_threshold, min_steps,
    init_mode="spikes",
    activator_type="juxtacrine",
    spike_value=5.0,
    save_every=10
):
    """Run activator–inhibitor simulation with Neumann boundary conditions."""

    # --- Build initial fields ---
    if init_mode == "random_tight" or init_mode == "peak_steady_state" or init_mode == "activator_spike_steady_state":
        # Try to get the non-null, reaction-stable steady state (fast)
        try:
            a_ss, i_ss, H_ss = fast_stable_steady_state(p, tol=5e-4, max_newton=12)
        except Exception:
            a_ss = i_ss = 0.0

        # Fallback to provided spike_value if solver didn't find a non-null state
        if not (a_ss > 0.0 and i_ss > 0.0 and np.isfinite(a_ss) and np.isfinite(i_ss)):
            a_ss = float(spike_value)
            i_ss = float(spike_value)

        # Use the steady-state values as per-species spikes/levels
        activator, inhibitor = initialize_fields(
            N,
            init_mode,
            spike_value,               # keep generic spike_value if your initializer uses it
            spike_value_a=float(a_ss),
            spike_value_i=float(i_ss),
        )
    else:
        # Default path: whatever your initializer already does
        activator, inhibitor = initialize_fields(N, init_mode, spike_value)

    activator_history = [activator.copy()]
    inhibitor_history = [inhibitor.copy()]

    for step in range(steps):
        activator_previous = activator.copy()
        inhibitor_previous = inhibitor.copy()

        activator_new = np.empty_like(activator)
        inhibitor_new = np.empty_like(inhibitor)

        update_interior(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p, activator_type)
        update_boundaries(activator, inhibitor, activator_new, inhibitor_new, N, dt, dx, p, activator_type)

        # Enforce non-negativity
        #activator_new = np.maximum(activator_new, 0.0)
        #inhibitor_new = np.maximum(inhibitor_new, 0.0)

        activator, inhibitor = activator_new, inhibitor_new

        if step % save_every == 0:
            #Compare the two steps to decide when to stop simulation
            diff = np.sum(np.abs(activator_new - activator_history[-1])) + np.sum(np.abs(inhibitor_new - inhibitor_history[-1]))

            #Add new values to history
            activator_history.append(activator.copy())
            inhibitor_history.append(inhibitor.copy())

            #Sum of differences for each point for activator + inhibitor between new and previous steps
            if step > min_steps and diff/(2*N) < stopping_threshold: #average change per step per tile of less than 0.000001
                print(f"Converged at step {step}, total average difference per tile over {save_every} steps = {diff/(2*N)}")
                break

    return activator_history, inhibitor_history, step


def run_simulation(params):
    """
    Thin wrapper to call run_coupled_neumann with a parameter dict.
    """
    result = run_coupled_neumann(
        params["N"],
        params["steps"],
        params["dt"],
        params["dx"],
        params,
        params.get("stopping_threshold", 1e-4),
        params.get("min_steps", 10000),
        init_mode=params.get("init_mode", "activator_spike"),
        activator_type=params.get("activator_type", "juxtacrine"),
        spike_value=params.get("spike_value", 5.0),
        save_every=params.get("save_every", 100),

    )

    activator_hist, inhibitor_hist, steps_used = result

    return {
        "status": "done",  # your loop prints convergence info already
        "steps_used": steps_used,
        "parameters": params,
        "activator_initial": activator_hist[0],
        "activator_final": activator_hist[-1],
        "inhibitor_initial": inhibitor_hist[0],
        "inhibitor_final": inhibitor_hist[-1]
    }
