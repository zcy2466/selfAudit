#!/usr/bin/env python3
"""Run the CUAD experiment (Table 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from experiments.cuad_experiment import CUADExperiment


def main():
    settings = get_settings()
    experiment = CUADExperiment(settings)
    results = experiment.run()

    print("\n=== CUAD Results (Table 2) ===")
    print(f"{'Method':<20} {'AUPR':>6} {'Micro-F1':>8} {'Precision':>9} {'Recall':>6}")
    print("-" * 53)
    for method, metrics in results.items():
        print(
            f"{method:<20} "
            f"{metrics['AUPR']:>6.3f} "
            f"{metrics['Micro-F1']:>8.3f} "
            f"{metrics['Precision']:>9.3f} "
            f"{metrics['Recall']:>6.3f}"
        )


if __name__ == "__main__":
    main()