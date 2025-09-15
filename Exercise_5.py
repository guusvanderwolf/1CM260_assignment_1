# file: exercise_5_small_twoopt.py
import os
import numpy as np
import pandas as pd
import random
from pathlib import Path
from TSP import TSP

# --- controls ---
MAX_STARTS = 100
BASE_SEED = 260
INST_DIRS = ["Instances/Small"]

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

def run_instance(file_path: Path):
    tsp = TSP(str(file_path))
    n = tsp.nCities
    starts = choose_starts(n, file_path)

    costs = []
    for s in starts:
        # Step 1: build initial tour (Outlier Insertion)
        tour = tsp.getTour_OutlierInsertion(s)
        # Step 2: improve with 2-opt
        improved_tour = tsp.makeTwoOpt(tour)
        cost = tsp.computeCosts(improved_tour)
        costs.append(cost)

    return get_stats(costs)

def main():
    rows = []
    for d in INST_DIRS:
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".tsp"):
                continue
            fpath = Path(d) / fname
            print(f"Running {fpath}...")
            min_cost, mean_cost, var_cost, cv_cost = run_instance(fpath)
            rows.append({
                "Instance": fname,
                "Min Cost": round(float(min_cost), 2),
                "Mean Cost": round(float(mean_cost), 2),
                "Variance": round(float(var_cost), 2),
                "CV": round(float(cv_cost), 3),
            })

    df = pd.DataFrame(rows)
    print("\n=== RESULTS TABLE (Outlier Insertion + 2-Opt, Small) ===")
    print(df.to_string(index=False))
    df.to_csv("twoopt_small_stats.csv", index=False)

if __name__ == "__main__":
    main()
