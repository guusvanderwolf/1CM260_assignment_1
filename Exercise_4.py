"""
@author: Guus van der Wolf
"""

import os
import numpy as np
import pandas as pd
import random
from pathlib import Path
from TSP import TSP  

BASE_SEED = 129 
MAX_STARTS = 100
ALPHAS = [5, 10, 15, 20] #α in percent

def select_instances(base_dir="Instances", n_small=5, n_medium=3, n_large=2, seed=BASE_SEED):
    """
    Randomly (but reproducibly) select instances per size group

    Parameters
    ----------
    base_dir : str
        path to the base directory containing 'Small', 'Medium', and 'Large' subfolders
    n_small : int
        number of small instances to select (default 5)
    n_medium : int
        number of medium instances to select (default 3)
    n_large : int
        number of large instances to select (default 2)
    seed : int
        random seed for reproducibility (default BASE_SEED)

    Returns
    -------
    dict
        keys {"Small", "Medium", "Large"} and values that are lists of file paths to the selected .tsp instances
    """
    random.seed(seed)
    selected = {}
    for group, n_pick in (("Small", n_small), ("Medium", n_medium), ("Large", n_large)):
        folder = os.path.join(base_dir, group)
        pool = sorted(f for f in os.listdir(folder) if f.endswith(".tsp"))
        if n_pick > len(pool):
            raise ValueError(f"Requested {n_pick} {group}, but only {len(pool)} available.")
        chosen = random.sample(pool, n_pick)
        selected[group] = [os.path.join(folder, f) for f in chosen]
    return selected

def choose_starts(n: int, file_path: Path, max_starts: int = MAX_STARTS, base_seed: int = BASE_SEED):
    """
    Select up to MAX_STARTS distinct start cities for an instance. If the number of cities is less than or equal to MAX_STARTS, all start cities are used.

    Parameters
    ----------
    n : int
        number of cities in the instance
    file_path : path
        path to the .tsp file, used to generate a reproducible seed
    max_starts : int
        maximum number of starts (default MAX_STARTS)
    base_seed : int
        base seed to combine with file path for reproducibility

    Returns
    -------
    list of int
        selected start cities
    """
    if n <= max_starts:
        return list(range(n))
    seed = (hash(file_path.as_posix()) ^ base_seed) & 0xFFFFFFFF
    rng = random.Random(seed)
    return rng.sample(range(n), max_starts)

def get_stats(costs):
    """
    Compute summary statistics from a list of tour costs

    Parameters
    ----------
    costs : list of float
        list of tour costs from different runs

    Returns
    -------
    tuple
        where
        - min_cost : float
            minimum cost observed
        - mean_cost : float
            average cost
        - var_cost : float
            variance of costs
        - cv_cost : float
            coefficient of variation (std/mean)
    """
    min_cost = np.min(costs)
    mean_cost = np.mean(costs)
    var_cost = np.var(costs, ddof=1) 
    std_cost = np.std(costs, ddof=1)
    cv_cost = std_cost / mean_cost if mean_cost > 0 else 0.0 #prevents division by 0
    return min_cost, mean_cost, var_cost, cv_cost

def run_instance(file_path: Path, alpha_pct: int):
    """
    Run the GRASPed Insertion heuristic on a single instance for multiple start cities

    Parameters
    ----------
    file_path : path
        path to the .tsp instance file
    alpha_pct : int
        α in percent 

    Returns
    -------
    tuple
        statistics (min_cost, mean_cost, var_cost, cv_cost) over all starts
    """
    #get the number of cities (DIMENSIONS) of the instance
    tsp = TSP(str(file_path))
    n = tsp.nCities

    starts = choose_starts(n, file_path) #get list of start cities

    #run the heuristic and save the costs for each start city and α
    costs = []
    base = (hash(file_path.as_posix()) ^ (alpha_pct * 1009) ^ BASE_SEED) & 0xFFFFFFFF
    for k, s in enumerate(starts):
        tour = tsp.getTour_GRASPedInsertion(s, alpha_pct=alpha_pct, seed=base + k)
        cost = tsp.computeCosts(tour)
        costs.append(cost)
    return get_stats(costs)

def main():
    picked = select_instances(seed=BASE_SEED)
    instance_paths = picked["Small"] + picked["Medium"] + picked["Large"]

    rows = []
    for fpath in instance_paths:
        fpath = Path(fpath)
        print(f"Running {fpath}")
        for a in ALPHAS:
            min_c, mean_c, var_c, cv_c = run_instance(fpath, a)
            rows.append({
                "Instance": fpath.name,
                "alpha%": a,
                "Min Cost": round(float(min_c), 2),
                "Mean Cost": round(float(mean_c), 2),
                "Variance": round(float(var_c), 2),
                "CV": round(float(cv_c), 3),
            })

    df = pd.DataFrame(rows)
    print(f"\nResults Exercise 4, with {MAX_STARTS} starts and {str(ALPHAS)} α%")
    print(df.to_string(index=False))
    df.to_csv("exercise_4.csv", index=False)

if __name__ == "__main__":
    main()
