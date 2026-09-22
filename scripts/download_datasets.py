#!/usr/bin/env python3
"""Download all three benchmark datasets for SelfAudit experiments.

Datasets:
- ContractNLI: 2,091 test samples (NDAs + hypotheses)
- CUAD: 41 contract review categories
- LegalBench-RAG: 4 legal domains for retrieval evaluation
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))


def download_contractnli(data_dir: Path):
    """Download ContractNLI from HuggingFace datasets."""
    print("Downloading ContractNLI...")
    target = data_dir / "contractnli"
    target.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        dataset = load_dataset("kldisc/contract-nli", trust_remote_code=True)

        for split in ["train", "test", "validation"]:
            if split in dataset:
                samples = []
                for item in dataset[split]:
                    samples.append({
                        "document_id": item.get("document_id", ""),
                        "nda_text": item.get("text", ""),
                        "hypothesis": item.get("hypothesis", ""),
                        "label": item.get("label", ""),
                        "spans": item.get("spans", item.get("evidence_spans", [])),
                    })
                filepath = target / f"{split}.json"
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
                print(f"  Saved {len(samples)} {split} samples to {filepath}")
    except Exception as e:
        print(f"  ContractNLI download failed: {e}")
        print("  Creating placeholder file. Please download manually from:")
        print("  https://huggingface.co/datasets/kldisc/contract-nli")

        for split in ["train", "test"]:
            filepath = target / f"{split}.json"
            if not filepath.exists():
                with open(filepath, "w") as f:
                    json.dump([], f)


def download_cuad(data_dir: Path):
    """Download CUAD from HuggingFace datasets."""
    print("Downloading CUAD...")
    target = data_dir / "cuad"
    target.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        dataset = load_dataset("cuad", trust_remote_code=True)

        if "test" in dataset:
            samples = []
            for item in dataset["test"]:
                samples.append({
                    "contract_id": item.get("title", ""),
                    "contract_text": item.get("context", ""),
                    "categories": [
                        {
                            "name": "CUAD category",
                            "query": item.get("question", ""),
                            "spans": item.get("answers", {}).get("text", []),
                        }
                    ],
                })
            filepath = target / "cuad_data.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(samples)} samples to {filepath}")
    except Exception as e:
        print(f"  CUAD download failed: {e}")
        print("  Creating placeholder file. Please download manually from:")
        print("  https://huggingface.co/datasets/cuad")

        filepath = target / "cuad_data.json"
        if not filepath.exists():
            with open(filepath, "w") as f:
                json.dump([], f)


def download_legalbench_rag(data_dir: Path):
    """Create placeholder for LegalBench-RAG."""
    print("Setting up LegalBench-RAG...")
    target = data_dir / "legalbench_rag"
    target.mkdir(parents=True, exist_ok=True)

    domains = ["nda", "ma", "commercial", "privacy"]
    for domain in domains:
        domain_dir = target / domain
        domain_dir.mkdir(exist_ok=True)
        filepath = domain_dir / "queries.json"
        if not filepath.exists():
            with open(filepath, "w") as f:
                json.dump({"queries": []}, f)

    print("  LegalBench-RAG currently requires manual setup.")
    print("  Place query/corpus files in data/legalbench_rag/<domain>/queries.json")
    print("  Reference: https://github.com/... (see paper for source)")


def main():
    data_dir = PROJECT_DIR / "data"
    print(f"Data directory: {data_dir}")

    download_contractnli(data_dir)
    download_cuad(data_dir)
    download_legalbench_rag(data_dir)

    print("\nDone. All datasets prepared in:", data_dir)


if __name__ == "__main__":
    main()