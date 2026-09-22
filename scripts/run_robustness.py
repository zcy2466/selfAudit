#!/usr/bin/env python3
"""Run cross-model robustness experiment (Table 5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from experiments.robustness_experiment import RobustnessExperiment


def main():
    settings = get_settings()
    experiment = RobustnessExperiment(settings)
    results = experiment.run()

    print("\n=== Cross-Model Robustness (Table 5) ===")
    print(f"{'Backbone':<25} {'Method':<12} {'Acc':>6} {'F1[W]':>6}")
    print("-" * 53)
    for backbone in settings.BACKBONE_LIST:
        if backbone in results:
            r = results[backbone]
            print(f"{backbone:<25} {'ZS':<12} {r['ZS_Acc']:>6.3f} {r['ZS_F1W']:>6.3f}")
            print(f"{'':<25} {'SelfAudit':<12} {r['SelfAudit_Acc']:>6.3f} {r['SelfAudit_F1W']:>6.3f}")

    print("\nAverages:")
    for avg_key in ["Average_ZS", "Average_SelfAudit"]:
        if avg_key in results:
            r = results[avg_key]
            print(f"  {avg_key}: Acc={r['Acc']:.3f}, F1[W]={r['F1[W]']:.3f}")

    for cv_key in ["CV_ZS", "CV_SelfAudit"]:
        if cv_key in results:
            r = results[cv_key]
            print(f"  {cv_key}: Acc={r['Acc']:.1f}%, F1[W]={r['F1[W]']:.1f}%")


if __name__ == "__main__":
    main()