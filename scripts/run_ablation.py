#!/usr/bin/env python3
"""Run ablation experiments (Tables 3 and 4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from experiments.ablation_experiment import AblationExperiment


def main():
    settings = get_settings()
    experiment = AblationExperiment(settings)
    results = experiment.run()

    # Table 3: Agent ablation
    print("\n=== Agent Ablation (Table 3) ===")
    agent_results = results.get("agent_ablation", {})
    print(f"{'Variant':<20} {'Acc':>6} {'F1[W]':>6} {'F1[E]':>6} {'F1[C]':>6}")
    print("-" * 50)
    for variant, metrics in agent_results.items():
        print(
            f"{variant:<20} "
            f"{metrics['Acc']:>6.3f} "
            f"{metrics['F1[W]']:>6.3f} "
            f"{metrics['F1[E]']:>6.3f} "
            f"{metrics['F1[C]']:>6.3f}"
        )

    # Table 4: HCD-EA component ablation
    print("\n=== HCD-EA Component Ablation (Table 4) ===")
    hcd_results = results.get("hcdea_ablation", {})
    print(f"{'Variant':<20} {'Acc':>6} {'F1[W]':>6} {'F1[E]':>6} {'F1[C]':>6}")
    print("-" * 50)
    for variant, metrics in hcd_results.items():
        print(
            f"{variant:<20} "
            f"{metrics['Acc']:>6.3f} "
            f"{metrics['F1[W]']:>6.3f} "
            f"{metrics['F1[E]']:>6.3f} "
            f"{metrics['F1[C]']:>6.3f}"
        )


if __name__ == "__main__":
    main()