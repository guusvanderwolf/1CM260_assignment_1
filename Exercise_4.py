# file: exercise_4_small_grasp_alpha.py
import os
import numpy as np
import pandas as pd
import random
from pathlib import Path
from TSP import TSP   # must contain getTour_GRASPedInsertion(start, alpha_pct, seed=None)

# --- controls (kept similar to Exercise_2) ---
MAX_STARTS = 100         # cap starts per instance
BASE_SEED  = 260         # reproducibility
ALPHAS = [5, 10, 15, 20] # α in percent
INST_DIRS = ["Instances/Small"]  # only small instances, as requested

def choose_starts(n: int, file_path: Path, max_starts: int = MAX_STARTS, base_seed: int = BASE_SEED):
    if n <= max_starts:
        return list(range(n))
    seed = (hash(file_path.as_posix()) ^ base_seed) & 0xFFFFFFFF
    rng = random.Random(seed)
    return rng.sample(range(n), max_starts)

def get_stats(costs):
    min_cost = np.min(costs)
    mean_cost = np.mean(costs)
    var_cost  = np.var(costs, ddof=1)
    std_cost  = np.std(costs, ddof=1)
    cv_cost   = std_cost / mean_cost if mean_cost > 0 else 0.0
    return min_cost, mean_cost, var_cost, cv_cost

def run_one_alpha(tsp: TSP, instance_path: Path, alpha_pct: int):
    n = tsp.nCities
    starts = choose_starts(n, instance_path)
    costs = []
    # fix a seed per (instance, alpha) for reproducibility, but vary by start idx
    base = (hash(instance_path.as_posix()) ^ (alpha_pct * 1009) ^ BASE_SEED) & 0xFFFFFFFF
    for k, s in enumerate(starts):
        tour = tsp.getTour_GRASPedInsertion(s, alpha_pct=alpha_pct, seed=base + k)
        costs.append(tsp.computeCosts(tour))
    return get_stats(costs)

def main():
    rows = []
    for d in INST_DIRS:
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".tsp"):
                continue
            path = Path(d) / fname
            print(f"Running {path}...")
            tsp = TSP(str(path))
            for a in ALPHAS:
                min_c, mean_c, var_c, cv_c = run_one_alpha(tsp, path, a)
                rows.append({
                    "Instance": fname,
                    "alpha%": a,
                    "Min Cost": round(float(min_c), 2),
                    "Mean Cost": round(float(mean_c), 2),
                    "Variance": round(float(var_c), 2),
                    "CV": round(float(cv_c), 3),
                })

    df = pd.DataFrame(rows)
    print("\n=== RESULTS TABLE (GRASPed Outlier Insertion on Small) ===")
    print(df.to_string(index=False))
    df.to_csv("grasp_outlier_small_alpha_stats.csv", index=False)

if __name__ == "__main__":
    main()
