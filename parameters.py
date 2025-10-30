# -------------------------
# Shared simulation parameters
# -------------------------
N = 101
dx = 1.0
steps = 100_000
dt = 0.01
save_every = 200
spike_value = 5.0
stopping_threshold = 1e-4
min_steps = 10000

#Define initiation mode and activator type
init_mode = "activator_spike_steady_state"
activator_type = "soluble"

# -------------------------
# Default reaction-diffusion parameters
# -------------------------
params = {
    #fixed by nondimensionalization
    "act_half_sat": 1.0,      # activator half-saturation constant
    "inh_half_sat": 1.0,      # inhibitor half-saturation constant
    "act_decay_rate": 1.0,    # activator decay rate
    "basal_prod": 0.0,        # basal leakiness of production (for both activator and inhibitor)
    "act_diffusion": 1.0,      # if the activator is soluble, diffusion coefficient
    #Hill coefficients
    "act_hill_coeff": 3,      # activator Hill coefficient
    "inh_hill_coeff": 3,      # inhibitor Hill coefficient

    #Free parameters
    "inh_diffusion": 10.0,     # inhibitor diffusion coefficient
    "act_prod_rate": 3.33,    # activator production rate
    "inh_prod_rate": 3.33,     # inhibitor production rate
    "inh_decay_rate": 1.0,    # inhibitor decay rate
}
