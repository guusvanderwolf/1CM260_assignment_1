"""
@author: Guus van der Wolf
"""

import os
import numpy as np
import pandas as pd
import random
from pathlib import Path
from TSP import TSP

BASE_SEED = 131  
MAX_STARTS = 10

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
        keys Small, Medium, Large and values that are lists of file paths to the selected .tsp instances
    """
    random.seed(seed) #ensures reproducibility where the same instances are picked each run
    selected = {}
    for group, n_pick in (("Small", n_small), ("Medium", n_medium), ("Large", n_large)):
        folder = os.path.join(base_dir, group)
        pool = sorted(f for f in os.listdir(folder) if f.endswith(".tsp")) #collect all available .tsp files in this group
        chosen = random.sample(pool, n_pick) #pick exactly n_pick instances at random
        selected[group] = [os.path.join(folder, f) for f in chosen] #store the paths so they can be used later
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
    #for small instances, use all cities as start points
    if n <= max_starts:
        return list(range(n))
    
    seed = (hash(file_path.as_posix()) ^ base_seed) & 0xFFFFFFFF #combine file path and base seed to get an instance-specific seed
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
    std_cost = np.std(costs, ddof=1)
    cv_cost = std_cost / mean_cost if mean_cost > 0 else 0.0 #prevents division by 0
    return min_cost, mean_cost, std_cost, cv_cost

def run_instance(file_path: Path):
    """
    Run the Outlier Insertion heuristic on a single instance for multiple start cities

    Parameters
    ----------
    file_path : path
        path to the .tsp instance file

    Returns
    -------
    tuple
        statistics (min_cost, mean_cost, var_cost, cv_cost) over all starts
    """
    #get the number of cities (DIMENSIONS) of the instance
    tsp = TSP(str(file_path))
    n = tsp.nCities 

    starts = choose_starts(n, file_path) #get list of start cities

    #run the heuristic and save the costs for each start city
    costs = []
    for s in starts:
        tour = tsp.getTour_OutlierInsertion(s)
        tsp.evaluateSolution(tour)
        cost = tsp.computeCosts(tour)
        costs.append(cost)
    return get_stats(costs)

def main():
    """
    Main entry point. Selects 5 small, 3 medium, and 2 large instances reproducibly, runs the Outlier Insertion heuristic with up to MAX_STARTS different starts per instance, and prints a summary results table
    """
    picked = select_instances(seed=BASE_SEED)
    instance_paths = picked["Small"] + picked["Medium"] + picked["Large"]

    #run the Outlier Insertion heuristic for the instances and collect the statistics
    rows = []
    for fpath in instance_paths:
        fpath = Path(fpath)
        print(f"Running {fpath}")
        min_cost, mean_cost, std_cost, cv_cost = run_instance(fpath)
        rows.append({
            "Instance": fpath.name,
            "Min. Obj.": round(float(min_cost), 2),
            "Mean. Obj.": round(float(mean_cost), 2),
            "St. Dev.": round(float(std_cost), 2),
            "Cof. Var.": round(float(cv_cost), 3),
        })

    #convert results to a DataFrame for printing and .csv
    df = pd.DataFrame(rows)
    print(f"\nResults Exercise 3, with {MAX_STARTS} starts")
    print(df.to_string(index=False))
    df.to_csv("exercise_3.csv", index=False)

if __name__ == "__main__":
    main()
