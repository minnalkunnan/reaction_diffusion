from pathlib import Path
import sys, os
import argparse
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import yaml  # <— requires PyYAML: pip install pyyaml

# allow importing simulation.py from parent directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simulation import run_simulation

from grid import make_param_grid
from io_utils import _to_json_list, write_constants_txt

OUTPUT_COLS = [
    "steps_used", "activator_steady-state", "inhibitor_steady-state",
    "activator_final", "inhibitor_final"
]

def run_one(p, varied_keys):
    r = run_simulation(p)
    row = {k: p[k] for k in varied_keys}
    row.update({
        "steps_used": r.get("steps_used"),
        "activator_steady-state": _to_json_list(r.get("activator_steady-state")),
        "inhibitor_steady-state": _to_json_list(r.get("inhibitor_steady-state")),

        "activator_final": _to_json_list(r.get("activator_final")),
        "inhibitor_final": _to_json_list(r.get("inhibitor_final"))

    })
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # read config
    OUTDIR = cfg.get("outdir", "runs/default")
    MODE = cfg.get("mode", "grid")
    BASE = cfg["base"]
    SWEEPS = cfg.get("sweeps", {})

    os.makedirs(OUTDIR, exist_ok=True)
    varied_keys = list(SWEEPS.keys())

    # build grid and constants
    param_list = make_param_grid(BASE, sweeps=SWEEPS, mode=MODE)
    constants = {k: v for k, v in BASE.items() if k not in varied_keys}
    write_constants_txt(constants, os.path.join(OUTDIR, "constants.txt"))

    # run sims
    results = Parallel(n_jobs=-1)(
        delayed(run_one)(p, varied_keys) for p in tqdm(param_list, desc="Running simulations")
    )

    # save CSV with only varied params + outputs
    df = pd.DataFrame(results)[varied_keys + OUTPUT_COLS]
    df.to_csv(os.path.join(OUTDIR, "batch_results.csv"), index=False)

    print(f"Wrote {len(constants)} constants to {OUTDIR}/constants.txt")
    print(f"Saved {len(df)} rows to {OUTDIR}/batch_results.csv")

if __name__ == "__main__":
    main()
