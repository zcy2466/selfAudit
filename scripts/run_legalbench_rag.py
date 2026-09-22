#!/usr/bin/env python3
"""Run the LegalBench-RAG experiment (Table 6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from experiments.legalbench_rag_experiment import LegalBenchRAGExperiment


def main():
    settings = get_settings()
    experiment = LegalBenchRAGExperiment(settings)
    results = experiment.run()

    print("\n=== LegalBench-RAG Results (Table 6) ===")
    for domain, metrics in results.items():
        print(f"\n  Domain: {domain}")
        for k in [1, 8, 64]:
            p_key = f"P@{k}"
            r_key = f"R@{k}"
            if p_key in metrics:
                print(f"    {p_key}: {metrics[p_key]:.2f}%  {r_key}: {metrics[r_key]:.2f}%")


if __name__ == "__main__":
    main()