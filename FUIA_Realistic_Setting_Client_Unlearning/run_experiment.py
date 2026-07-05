#!/usr/bin/env python3

import os
import sys
import runpy
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
EXP_DIR = os.path.join(HERE, "experiments")
RESULTS_DIR = os.path.join(HERE, "results")


def available_experiments():
    names = []
    for f in sorted(os.listdir(EXP_DIR)):
        if f.endswith(".py") and not f.startswith("__"):
            names.append(f[:-3])
    return names


def main():
    exps = available_experiments()

    if len(sys.argv) < 2:
        print("Available experiments:")
        for name in exps:
            print(f"  - {name}")
        print("\nUsage: python run_experiment.py <experiment_name>")
        return

    name = sys.argv[1]
    if name not in exps:
        print(f"Unknown experiment: {name}")
        print("Available:", ", ".join(exps))
        sys.exit(1)

    #Make both the flat src/ modules and the experiments package importable
    sys.path.insert(0, SRC)
    sys.path.insert(0, HERE)

    #config.py reads this at import time and applies the overrides
    os.environ["FUIA_EXPERIMENT"] = name

    print(f"Running experiment: {name}\n")
    #Execute src/main.py exactly as a script (its __main__ block runs)
    runpy.run_path(os.path.join(SRC, "main.py"), run_name="__main__")

    #attack.py saves the reconstruction as fuia_result.png in the project root;
    #archive it under results/ with the experiment name so runs don't overwrite
    #each other.
    produced = os.path.join(HERE, "fuia_result.png")
    if os.path.exists(produced):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        dest = os.path.join(RESULTS_DIR, f"{name}.png")
        shutil.move(produced, dest)
        print(f"\nReconstruction archived to results/{name}.png")


if __name__ == "__main__":
    main()
