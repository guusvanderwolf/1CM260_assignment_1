"""
@author: Guus van der Wolf
"""

import os
import numpy as np
import pandas as pd
import random
from pathlib import Path
from TSP import TSP
import matplotlib.pyplot as plt

BASE_SEED = 131 
STARTS_BY_GROUP = {"Small": 20, "Medium": 10, "Large": 5}
ALPHA  = 10  #α in percent

#where to save the figure and CSV
OUT_DIR = Path("fig_grasp2opt_selected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def choose_starts(n, file_path, max_starts, base_seed = BASE_SEED):
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
        - std_cost : float
            standard deviation of costs
        - cv_cost : float
            coefficient of variation (std/mean)
    """
    min_cost = np.min(costs)
    mean_cost = np.mean(costs)
    std_cost = np.std(costs, ddof=1)
    cv_cost = std_cost / mean_cost if mean_cost > 0 else 0.0 #prevents division by 0
    return min_cost, mean_cost, std_cost, cv_cost

def scatter_xy(fig_path, xs, ys, title):
    """
    Make a scatter plot with x = GRASPed Insertion cost (before 2-opt) and y = GRASPed Insertion+2-opt cost
    
    Parameters
    ----------
    path : Path
        output file path where the figure will be saved
    xs : list or array-like of float
        objective values before applying local search
    ys : list or array-like of float
        objective values after applying local search
    title : str
        title for the scatter plot
    """
    plt.figure(figsize=(7.5, 5.5), dpi=140)
    plt.scatter(xs, ys, s=22, facecolors="none", edgecolors="#1f77b4", linewidths=1.0)

    #add diagonal line (y=x) to indicate no improvement
    lo, hi = min(xs.min(), ys.min()), max(xs.max(), ys.max())
    plt.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1) 
    
    plt.xlabel("Constructive heuristic - GRASPed Insertion cost (before 2-opt)")
    plt.ylabel("After local search - GRASPed Insertion + 2-opt cost")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def run_instance(file_path, max_starts):
    """
    Run GRASPed Insertion and GRASPed Insertion + 2-opt on a single instance for multiple start cities, returning the per-start costs before and after local search.

    Parameters
    ----------
    file_path : path
        path to the .tsp instance file

    Returns
    -------
    tuple
        xs, ys where xs are costs before 2-opt (GRASPed Insertion) and ys are costs after 2-opt (GRASPed Insertion + 2-opt)
    """
    #get the number of cities (DIMENSIONS) of the instance
    tsp = TSP(str(file_path))
    n = tsp.nCities

    starts = choose_starts(n, file_path, max_starts=max_starts, base_seed=BASE_SEED) #get list of start cities
    base = (hash(file_path.as_posix()) ^ (ALPHA * 1009) ^ BASE_SEED) & 0xFFFFFFFF #use seed for the instance and alpha and vary by start index for reproducibility

    before_costs, after_costs = [], []
    for k, s in enumerate(starts):
        #cost before local search (GRASPed Insertion)
        tour_before = tsp.getTour_GRASPedInsertion(start=s, alpha_pct=ALPHA, seed=base + k)
        c_before = tsp.computeCosts(tour_before)

        #cost after local search (GRASPed Insertion + 2-opt)
        tour_after = tsp.getTour_GRASP2Opt(start=s, alpha_pct=ALPHA, seed=base + k)
        tsp.evaluateSolution(tour_after)
        c_after = tsp.computeCosts(tour_after)

        before_costs.append(c_before)
        after_costs.append(c_after)

    return np.array(before_costs), np.array(after_costs)

def main():
    """
    Main entry point. Selects 5 small, 3 medium, and 2 large instances reproducibly, runs GRASPed Insertion and GRASPed Insertion + 2-opt with up to MAX_STARTS starts per instance, saves a scatter per instance, and prints a summary results table.
    """
    picked = select_instances()  

    #run the GRASPed insertion/2-opt for the instances, save figures, and collect the statistics
    rows = []
    for group, paths in picked.items():                
        for f in paths:
            fpath = Path(f)
            max_starts = STARTS_BY_GROUP[group]
            print(f"Running {fpath}")
            xs, ys = run_instance(fpath, max_starts)

            #create scatter with x=before, y=after for each instance
            fig_path = OUT_DIR / f"{fpath.stem}_grasp2opt.png"
            scatter_xy(fig_path, xs, ys, f"{fpath.name} — GRASP(α={ALPHA}%) + 2-opt")

            b_min, b_mean, b_std, b_cv = get_stats(list(xs))
            a_min, a_mean, a_std, a_cv = get_stats(list(ys))

            #create Dataframe with both before and after statistics
            rows.append({
                "Group": group,
                "Instance": fpath.name,
                "Starts": max_starts,

                "Before_Min": round(float(b_min), 2),
                "Before_Mean": round(float(b_mean), 2),
                "Before_Std": round(float(b_std), 2),
                "Before_CV": round(float(b_cv), 3),

                "After_Min": round(float(a_min), 2),
                "After_Mean": round(float(a_mean), 2),
                "After_Std": round(float(a_std), 2),
                "After_CV": round(float(a_cv), 3),

                "Mean_Improvement": round(float(b_mean - a_mean), 2),
                "Mean_Improvement_%": round(float((b_mean - a_mean) / b_mean * 100.0), 2),
            })

    #convert results to a DataFrame for printing and .csv
    df = pd.DataFrame(rows)
    print(f"\nResults Exercise 6 (Small=20, Medium=10, Large=5; α={ALPHA}%)")
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "exercise_6_summary.csv", index=False)
    print(f"\nSaved figures, per-start CSVs, and summary to: {OUT_DIR}/")

if __name__ == "__main__":
    main()
