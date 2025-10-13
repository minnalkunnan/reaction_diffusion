# batch_runner.py
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm
from simulation import run_simulation

# --- Step 1: create parameter sets ---

param_list = [
    {
        #General parameters
        "N": 500,
        "steps": 100000,
        "dt": 1e-2,
        "dx": 1.0,
        "init_mode": "random_tight",
        "activator_type": "membrane-tethered",
        "spike_value": 3,
        "save_every": 200,
        "stopping_threshold": 1e-4,
        "min_steps": 10000,

        #Non-dimensionalized
        "act_decay_rate": 1.0,
        "act_half_sat": 1.0,
        "inh_half_sat": 1.0,
        "basal_prod": 0.0,
        #Hill coefficients
        "act_hill_coeff": 3,
        "inh_hill_coeff": 3,
        #Other parameters
        "act_prod_rate": 3.0,
        "inh_prod_rate": 3.0,
        "inh_decay_rate": i,
        "inh_diffusion": 10.0
    }
    for i in np.linspace(1.1, 2, 10)
]


# --- Step 2: run them in parallel ---
def run_and_summarize(p):
    result = run_simulation(p)
    return {
        "parameters": result["parameters"],
        "steps_used": result["steps_used"],
        "activator_initial": result["activator_initial"],
        "activator_final": result["activator_final"],
        "inhibitor_initial": result["inhibitor_initial"],
        "inhibitor_final": result["inhibitor_final"]
    }

results = Parallel(n_jobs=-1)(
    delayed(run_and_summarize)(p) for p in tqdm(param_list, desc="Running simulations")
)

# --- Step 3: save to CSV ---
df = pd.DataFrame(results)
df.to_csv("batch_results.csv", index=False)

print("Saved", len(results), "results to batch_results.csv")
