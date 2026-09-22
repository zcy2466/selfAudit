"""Dataset loaders for ContractNLI, CUAD, and LegalBench-RAG benchmarks."""

import json
import os
from pathlib import Path
from typing import Optional

from app.utils.logger import logger


def load_contractnli(
    path: str | Path,
    split: str = "test",
    max_samples: int | None = None,
) -> list[dict]:
    """
    Load the ContractNLI dataset.

    ContractNLI contains 2,091 test samples. Each sample has:
    - nda_text: Full NDA document text
    - hypothesis: Legal hypothesis to verify
    - label: Entailment, Contradiction, or Not Mentioned

    Args:
        path: Path to the dataset directory or JSON file.
        split: "train", "test", or "validation".
        max_samples: If set, limit to this many samples.

    Returns:
        List of dicts with keys: document_id, nda_text, hypothesis, label, spans.
    """
    path = Path(path)
    data_file = path / f"{split}.json" if path.is_dir() else path

    samples = []
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("samples", []))

        for i, item in enumerate(raw):
            samples.append({
                "document_id": item.get("document_id", item.get("id", str(i))),
                "nda_text": item.get("nda_text", item.get("text", item.get("document", ""))),
                "hypothesis": item.get("hypothesis", ""),
                "label": _normalize_label(item.get("label", "")),
                "spans": item.get("spans", item.get("evidence_spans", [])),
            })

        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Loaded {len(samples)} ContractNLI samples from {data_file}")
    except FileNotFoundError:
        logger.warning(f"ContractNLI data not found at {data_file}. Using empty dataset.")
    except Exception as e:
        logger.error(f"Error loading ContractNLI: {e}")

    return samples


def load_cuad(
    path: str | Path,
    max_samples: int | None = None,
) -> list[dict]:
    """
    Load the CUAD (Contract Understanding Atticus Dataset).

    CUAD covers 41 contract review categories as binary span extraction tasks.

    Args:
        path: Path to the dataset directory or JSON file.
        max_samples: If set, limit to this many contracts.

    Returns:
        List of dicts with keys: contract_id, contract_text, category, query, gold_spans.
    """
    path = Path(path)
    data_file = path / "cuad_data.json" if path.is_dir() else path

    samples = []
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("contracts", []))

        for contract in raw:
            contract_id = contract.get("contract_id", contract.get("title", ""))
            contract_text = contract.get("text", contract.get("content", ""))

            for category in contract.get("categories", []):
                samples.append({
                    "contract_id": contract_id,
                    "contract_text": contract_text,
                    "category": category.get("name", category.get("category", "")),
                    "query": category.get("query", category.get("question", "")),
                    "gold_spans": category.get("spans", category.get("answers", [])),
                })

        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Loaded {len(samples)} CUAD samples from {data_file}")
    except FileNotFoundError:
        logger.warning(f"CUAD data not found at {data_file}. Using empty dataset.")
    except Exception as e:
        logger.error(f"Error loading CUAD: {e}")

    return samples


def load_legalbench_rag(
    path: str | Path,
    domain: str | None = None,
    max_queries: int | None = None,
) -> dict[str, list[dict]]:
    """
    Load the LegalBench-RAG dataset.

    Covers four legal domains: NDAs, M&A agreements, commercial contracts,
    and consumer-facing privacy policies.

    Args:
        path: Path to the dataset directory.
        domain: Specific domain to load, or None for all.
        max_queries: If set, limit queries per domain.

    Returns:
        Dict mapping domain name to list of query dicts.
    """
    path = Path(path)
    domains = {}

    domain_names = [domain] if domain else ["nda", "ma", "commercial", "privacy"]

    for domain_name in domain_names:
        domain_file = path / domain_name / "queries.json"
        if not domain_file.exists():
            domain_file = path / f"{domain_name}.json"

        try:
            with open(domain_file, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict):
                raw = raw.get("queries", raw.get("data", []))

            queries = []
            for item in raw:
                queries.append({
                    "query_id": item.get("query_id", item.get("id", "")),
                    "domain": domain_name,
                    "query": item.get("query", item.get("question", "")),
                    "relevant_doc_ids": item.get(
                        "relevant_doc_ids",
                        item.get("relevant_ids", item.get("relevant", [])),
                    ),
                    "corpus": item.get("corpus", {}),
                })

            if max_queries:
                queries = queries[:max_queries]

            domains[domain_name] = queries
            logger.info(f"Loaded {len(queries)} LegalBench-RAG queries for domain '{domain_name}'")
        except FileNotFoundError:
            logger.warning(f"LegalBench-RAG data not found for domain '{domain_name}'")
            domains[domain_name] = []
        except Exception as e:
            logger.error(f"Error loading LegalBench-RAG domain '{domain_name}': {e}")
            domains[domain_name] = []

    return domains


def _normalize_label(label: str) -> str:
    """Normalize label strings to standard forms."""
    label = label.strip().lower()
    label_map = {
        "entailment": "Entailment",
        "contradiction": "Contradiction",
        "not mentioned": "Not Mentioned",
        "not_mentioned": "Not Mentioned",
        "neutral": "Not Mentioned",
    }
    return label_map.get(label, label.capitalize())


def split_dataset(
    samples: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split a dataset into train and validation sets."""
    import random
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    train_idx = set(indices[:split_point])
    val_idx = set(indices[split_point:])
    train_samples = [s for i, s in enumerate(samples) if i in train_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]
    return train_samples, val_samples