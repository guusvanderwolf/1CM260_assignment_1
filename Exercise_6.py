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

BASE_SEED = 129 
MAX_STARTS = 100
ALPHA_PCT  = 10  # α% for GRASP RCLs

# where to save figures/CSV
OUT_DIR = Path("fig_grasp2opt_small")
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
        keys {"Small", "Medium", "Large"} and values that are lists of file paths to the selected .tsp instances
    """
    import os, random
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

def stats(costs: list[float]):
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
    costs = np.asarray(costs, dtype=float)
    min_c  = float(np.min(costs))
    mean_c = float(np.mean(costs))
    var_c  = float(np.var(costs, ddof=1)) if costs.size > 1 else 0.0
    std_c  = float(np.std(costs, ddof=1)) if costs.size > 1 else 0.0
    cv_c   = (std_c / mean_c) if mean_c > 0 else 0.0
    return min_c, mean_c, var_c, cv_c

def scatter_xy(fig_path: Path, xs: np.ndarray, ys: np.ndarray, title: str):
    plt.figure(figsize=(7.5, 5.5), dpi=140)
    plt.scatter(xs, ys, s=22, facecolors="none", edgecolors="#1f77b4", linewidths=1.0)
    lo, hi = min(xs.min(), ys.min()), max(xs.max(), ys.max())
    plt.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)  # y=x reference
    plt.xlabel("Constructive heuristic — GRASP cost (before 2-opt)")
    plt.ylabel("After local search — GRASP + 2-opt cost")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def run_instance(path: Path):
    tsp = TSP(str(path))
    starts = choose_starts(tsp.nCities, path)

    # tie seed to (instance, alpha) and vary by start index for reproducibility
    base = (hash(path.as_posix()) ^ (ALPHA_PCT * 1009) ^ BASE_SEED) & 0xFFFFFFFF

    before_costs, after_costs = [], []
    for k, s in enumerate(starts):
        # cost BEFORE local search (pure GRASP)
        tour_before = tsp.getTour_GRASPedInsertion(start=s, alpha_pct=ALPHA_PCT, seed=base + k)
        c_before = tsp.computeCosts(tour_before)

        # cost AFTER local search (GRASP + 2-opt) via your new method
        tour_after = tsp.getTour_GRASP2Opt(start=s, alpha_pct=ALPHA_PCT, seed=base + k)
        c_after = tsp.computeCosts(tour_after)

        before_costs.append(c_before)
        after_costs.append(c_after)

    return np.array(before_costs), np.array(after_costs)

def main():
    picked = select_instances(seed=BASE_SEED)
    instance_paths = picked["Small"] + picked["Medium"] + picked["Large"]

    rows = []
    for fpath in instance_paths:
        fpath = Path(fpath)
        print(f"Running {fpath} …")
        xs, ys = run_instance(fpath)

        fig_path = OUT_DIR / f"{fpath.stem}_grasp2opt.png"
        scatter_xy(fig_path, xs, ys, f"{fpath.name} — GRASP(α={ALPHA_PCT}%) + 2-opt")

        min_c, mean_c, var_c, cv_c = stats(list(ys))
        rows.append({
            "Instance": fpath.name,
            "Min Cost": round(min_c, 2),
            "Mean Cost": round(mean_c, 2),
            "Variance": round(var_c, 2),
            "CV": round(cv_c, 3),
        })

    df = pd.DataFrame(rows)
    print("\n=== RESULTS TABLE (GRASP + 2-Opt, Selected Instances) ===")
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "grasp2opt_stats.csv", index=False)
    print(f"\nSaved figures & CSV to: {OUT_DIR}/")

if __name__ == "__main__":
    main()
