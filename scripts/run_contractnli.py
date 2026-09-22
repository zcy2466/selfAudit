#!/usr/bin/env python3
"""Run the ContractNLI experiment (Table 1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from experiments.contractnli_experiment import ContractNLIExperiment


def main():
    settings = get_settings()
    experiment = ContractNLIExperiment(settings)
    results = experiment.run()

    print("\n=== ContractNLI Results (Table 1) ===")
    print(f"{'Method':<20} {'Acc':>6} {'F1[W]':>6} {'F1[E]':>6} {'F1[C]':>6} {'F1[N]':>6}")
    print("-" * 56)
    for method, metrics in results.items():
        print(
            f"{method:<20} "
            f"{metrics['Acc']:>6.3f} "
            f"{metrics['F1[W]']:>6.3f} "
            f"{metrics['F1[E]']:>6.3f} "
            f"{metrics['F1[C]']:>6.3f} "
            f"{metrics['F1[N]']:>6.3f}"
        )


if __name__ == "__main__":
    main()