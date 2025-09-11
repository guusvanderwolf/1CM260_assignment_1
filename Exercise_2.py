import os
import numpy as np
import pandas as pd
import random
from pathlib import Path
from TSP import TSP

MAX_STARTS = 100
BASE_SEED = 260   # fixed seed for reproducibility
INST_DIRS = ["Instances/Small", "Instances/Medium", "Instances/Large"]

def choose_starts(n: int, file_path: Path, max_starts: int = MAX_STARTS, base_seed: int = BASE_SEED):
    """
    Return a list of starting nodes.
    - If n <= max_starts → all starts [0..n-1]
    - Else → exactly max_starts sampled without replacement (reproducible per file)
    """
    if n <= max_starts:
        return list(range(n))
    seed = (hash(file_path.as_posix()) ^ base_seed) & 0xFFFFFFFF
    rng = random.Random(seed)
    return rng.sample(range(n), max_starts)

def get_stats(costs):
    """Return min, mean, variance, coefficient of variation"""
    min_cost = np.min(costs)
    mean_cost = np.mean(costs)
    var_cost = np.var(costs, ddof=1)   # sample variance
    std_cost = np.std(costs, ddof=1)
    cv_cost = std_cost / mean_cost if mean_cost > 0 else 0.0
    return min_cost, mean_cost, var_cost, cv_cost

def run_instance(file_path: Path):
    tsp = TSP(str(file_path))
    n = tsp.nCities
    starts = choose_starts(n, file_path)
    costs = []
    for s in starts:
        tour = tsp.getTour_NN(s)
        cost = tsp.computeCosts(tour)
        costs.append(cost)
    return get_stats(costs)

def main():
    rows = []
    for d in INST_DIRS:
        for fname in os.listdir(d):
            if fname.endswith(".tsp"):
                file_path = Path(d) / fname
                print(f"Running {file_path}...")
                min_cost, mean_cost, var_cost, cv_cost = run_instance(file_path)
                rows.append({
                    "Instance": fname,
                    "Min Cost": min_cost.round(2),
                    "Mean Cost": mean_cost.round(2),
                    "Variance": var_cost.round(2),
                    "CV": cv_cost.round(3)
                })
    df = pd.DataFrame(rows)
    print("\n=== RESULTS TABLE ===")
    print(df.to_string(index=False))
    df.to_csv("nn_stats.csv", index=False)

if __name__ == "__main__":
    main()
