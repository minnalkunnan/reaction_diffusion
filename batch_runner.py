# batch_runner.py
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm
from simulation import run_simulation

# --- Step 1: create parameter sets ---

param_list = [
    {
        "N": 100,
        "steps": 50000,
        "dt": 1e-2,
        "dx": 1.0,
        "act_prod_rate": 1.0,
        "inh_prod_rate": i,   # varies systematically
        "act_decay_rate": 0.1,
        "inh_decay_rate": 0.1,
        "inh_diffusion": 0.5,   # fixed
        "act_half_sat": 1.0,
        "inh_half_sat": 1.0,
        "act_hill_coeff": 2.0,
        "inh_hill_coeff": 2.0,
        "basal_prod": 0.01,
        "init_mode": "two_activator_spikes",
        "spike_value": 5.0,
        "save_every": 100,
        "tol": 1e-6,
        "patience": 500,
    }
    for i in range(1, 101)  # 100 sims, inh_prod_rate = 1, 2, ..., 100
]


# --- Step 2: run them in parallel ---
def run_and_summarize(p):
    result = run_simulation(p)
    return {
        "status": result["status"],
        "steps_used": result["steps_used"],
        "inh_prod_rate": p["inh_prod_rate"],   # keep varied parameter
        "A_diff": result["activator_final"].max() - result["activator_final"].min(),
        "I_diff": result["inhibitor_final"].max() - result["inhibitor_final"].min(),
    }

results = Parallel(n_jobs=-1)(
    delayed(run_and_summarize)(p) for p in tqdm(param_list, desc="Running simulations")
)

# --- Step 3: save to CSV ---
df = pd.DataFrame(results)
df.to_csv("batch_results.csv", index=False)

print("Saved", len(results), "results to batch_results.csv")
