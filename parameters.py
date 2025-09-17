# -------------------------
# Shared simulation parameters
# -------------------------
N = 101
dx = 1.0
steps = 50_000
dt = 0.01
save_every = 10
spike_value = 5.0

# -------------------------
# Default reaction-diffusion parameters
# -------------------------
params = {
    "inh_diffusion": 1.0,     # inhibitor diffusion coefficient
    "act_half_sat": 1.0,      # activator half-saturation constant
    "inh_half_sat": 1.0,      # inhibitor half-saturation constant
    "act_hill_coeff": 5,      # activator Hill coefficient
    "inh_hill_coeff": 1,      # inhibitor Hill coefficient
    "basal_prod": 0.0,        # basal activator production
    "act_prod_rate": 1.14,    # activator production rate
    "act_decay_rate": 1.0,    # activator decay rate
    "inh_prod_rate": 1.0,     # inhibitor production rate
    "inh_decay_rate": 1.0,    # inhibitor decay rate
}
