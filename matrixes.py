# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import contextlib, io, gc

# Folders (edit if you like)
INST_ROOT = Path("Instances")
OUT_ROOT  = Path("Matrices")

# Hide prints from TSP.py so only our two lines show up
SILENCE_TSP = False

# Import your TSP class
if SILENCE_TSP:
    with contextlib.redirect_stdout(io.StringIO()):
        import TSP as tspmod
else:
    import TSP as tspmod
TSP = tspmod.TSP  # uses inst.distMatrix built in __init__

def main():
    files = sorted(INST_ROOT.rglob("*.tsp"))
    if not files:
        print(f"No .tsp files found under '{INST_ROOT}'.")
        return

    for tsp_path in files:
        name = tsp_path.stem
        rel  = tsp_path.relative_to(INST_ROOT)
        out_path = OUT_ROOT / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Starting computing matrix of {name} instance", flush=True)

        # Build the instance (this computes the full distance matrix)
        if SILENCE_TSP:
            with contextlib.redirect_stdout(io.StringIO()):
                inst = TSP(tsp_path)
        else:
            inst = TSP(tsp_path)

        # Save the matrix exactly as produced by TSP.py
        np.save(out_path, inst.distMatrix)

        # free memory before next instance
        del inst
        gc.collect()

        print(f"Finished computing matrix of {name} instance", flush=True)

if __name__ == "__main__":
    main()
