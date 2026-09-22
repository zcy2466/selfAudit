#!/usr/bin/env python3
"""Run parameter sensitivity experiment (tau and K sweeps)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from experiments.sensitivity_experiment import SensitivityExperiment


def main():
    settings = get_settings()
    experiment = SensitivityExperiment(settings)
    results = experiment.run()

    # Tau sweep
    print("\n=== Tau Sensitivity ===")
    tau_results = results.get("tau_sweep", {})
    print(f"{'tau':>6}  {'F1[W]':>6}")
    print("-" * 16)
    for key in sorted(tau_results.keys()):
        r = tau_results[key]
        print(f"{r['tau']:>6.2f}  {r['F1[W]']:>6.3f}")

    # K sweep
    print("\n=== K Sensitivity ===")
    k_results = results.get("k_sweep", {})
    print(f"{'K':>3}  {'F1[W]':>6}")
    print("-" * 13)
    for key in sorted(k_results.keys()):
        r = k_results[key]
        print(f"{r['K']:>3}  {r['F1[W]']:>6.3f}")


if __name__ == "__main__":
    main()