#!/usr/bin/env python3
"""Run computational efficiency benchmark (Table 8)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from experiments.efficiency_experiment import EfficiencyExperiment


def main():
    settings = get_settings()
    experiment = EfficiencyExperiment(settings)
    results = experiment.run()

    print("\n=== Computational Efficiency (Table 8) ===")
    print(f"{'Method':<25} {'LLM Calls':>10} {'Input Tokens':>12} {'Output Tokens':>13} {'Wall Time':>10}")
    print("-" * 75)
    for method, metrics in results.items():
        print(
            f"{method:<25} "
            f"{metrics['LLM Calls']:>10} "
            f"{metrics['Input Tokens']:>12} "
            f"{metrics['Output Tokens']:>13} "
            f"{metrics.get('Wall Time (s)', 0):>10.2f}s"
        )


if __name__ == "__main__":
    main()