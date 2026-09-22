# SelfAudit: Heterogeneous Multi-Dimensional Reflection for Complex Document Auditing

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"SelfAudit: Heterogeneous Multi-Dimensional Reflection for Complex Document Auditing"**.

SelfAudit is a reflection-driven multi-agent framework that audits complex documents (contracts, legal agreements, corporate reports) through four specialized agents and a Heterogeneous Multi-Dimensional Confidence Decomposition with Dependency-Aware Error Attribution (HCD-EA) mechanism.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SelfAudit Pipeline                      │
│                                                             │
│  ┌─────┐    ┌─────┐    ┌─────┐    ┌───────────────────┐    │
│  │ TPA │───→│ ERA │───→│ IA  │───→│ SRA (HCD-EA)      │    │
│  │     │    │     │    │     │    │  ┌─────────────┐   │    │
│  │Task │    │Evid │    │Insp.│    │  │RFS│ESS│RCS  │   │    │
│  │Plan │    │Retr.│    │D1-D4│    │  │ c=min(RFS,   │   │    │
│  └─────┘    └─────┘    └─────┘    │  │    ESS,RCS)  │   │    │
│       ↑         ↑          ↑      │  └──────┬───────┘   │    │
│       └─────────┴──────────┘      │         │           │    │
│         HCD-EA Selective          │    c < τ? re-exec   │    │
│         Re-Execution Loop         └───────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

- **TPA** (Task Planning Agent): Decomposes queries into `(objective, rule, audit-point)` triples
- **ERA** (Evidence Retrieval Agent): Hierarchical semantic tree + hybrid dense-sparse index + fusion reranking
- **IA** (Inspector Agent): 4-dimension verification (D1: Numerical, D2: Completeness, D3: Consistency, D4: Risk)
- **SRA** (Self-Reflection Agent): HCD-EA with 3D confidence decomposition (RFS, ESS, RCS), dependency-aware error attribution, and heterogeneous verification

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (for GPT-4o backbone)
- DashScope API key (for Qwen2.5-7B heterogeneous verifier)

### Installation

```bash
git clone <repository-url>
cd selfAudit

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### Download Datasets

```bash
python scripts/download_datasets.py
```

This downloads ContractNLI and CUAD from HuggingFace. LegalBench-RAG requires manual setup (see `data/README.md`).

### Run Experiments

```bash
# Main ContractNLI benchmark (Table 1)
python scripts/run_contractnli.py

# CUAD benchmark (Table 2)
python scripts/run_cuad.py

# LegalBench-RAG retrieval evaluation (Table 6)
python scripts/run_legalbench_rag.py

# Ablation studies (Tables 3, 4)
python scripts/run_ablation.py

# Cross-model robustness (Table 5)
python scripts/run_robustness.py

# Parameter sensitivity (tau, K)
python scripts/run_parameter_sensitivity.py

# Computational efficiency (Table 8)
python scripts/run_efficiency.py
```

Results are saved to `results/` as JSON files.

## Configuration

All parameters are configured via `.env` file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o` | Backbone model for TPA, ERA, IA |
| `HETEROGENEOUS_LLM_MODEL` | `qwen2.5-7b-instruct` | Heterogeneous verifier for SRA |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model (3072d) |
| `HCD_TAU` | `0.75` | Confidence threshold |
| `HCD_K` | `3` | Max HCD-EA iterations |
| `HCD_ALPHA` | `0.5` | RFS interpolation weight |
| `ERA_TOPK` | `10` | Number of evidence chunks retrieved |
| `LLM_TEMPERATURE` | `0` | LLM temperature |

## Project Structure

```
selfAudit/
├── app/
│   ├── core/config.py              # Pydantic settings
│   ├── services/
│   │   ├── llm/                    # LLM service + prompts
│   │   ├── embedding/              # text-embedding-3-large
│   │   ├── retrieval/              # ERA pipeline
│   │   │   ├── hierarchical_tree.py
│   │   │   ├── hybrid_index.py     # Dense + BM25
│   │   │   ├── fusion_reranker.py  # RRF + rerank
│   │   │   └── era_retriever.py
│   │   ├── rerank/                 # Rerank client
│   │   └── verifier/
│   │       ├── dependency_graph.py # G=(V,E)
│   │       └── heterogeneous.py    # Qwen2.5-7B verifier
│   ├── graphs/selfAudit/
│   │   ├── selfaudit_graph.py     # LangGraph StateGraph
│   │   ├── state/audit_state.py
│   │   └── nodes/
│   │       ├── tpa_node.py
│   │       ├── era_node.py
│   │       ├── ia_node.py
│   │       └── sra_node.py         # HCD-EA
│   ├── schemas/                    # Pydantic models
│   └── utils/
│       ├── metrics.py              # All evaluation metrics
│       ├── dataset_utils.py        # Dataset loaders
│       └── value_parser.py         # JSON extraction
├── experiments/                    # Experiment runners
├── scripts/                        # Entry points
├── data/                           # Dataset directory
├── results/                        # Experiment outputs
└── tests/                          # Unit tests
```

## Reproduced Tables

| Table | Description | Script |
|-------|-------------|--------|
| Table 1 | ContractNLI main results (14 methods) | `run_contractnli.py` |
| Table 2 | CUAD results (5 methods) | `run_cuad.py` |
| Table 3 | Agent ablation | `run_ablation.py` |
| Table 4 | HCD-EA component ablation | `run_ablation.py` |
| Table 5 | Cross-model robustness | `run_robustness.py` |
| Table 6 | LegalBench-RAG retrieval | `run_legalbench_rag.py` |
| Table 7 | Confidence decomposition | (logged during SRA execution) |
| Table 8 | Computational efficiency | `run_efficiency.py` |

## Baselines Implemented

1. **Zero-shot (ZS)** — Direct prompting
2. **Few-shot (FS)** — Few-shot prompting with 3 examples
3. **Chain-of-Thought (CoT)** — Step-by-step reasoning
4. **Self-Consistency** — Majority vote over 5 reasoning paths
5. **ReAct** — Interleaved reasoning and action
6. **Standard RAG** — Retrieve-then-answer
7. **Self-RAG** — Retrieval with self-reflection
8. **Reflexion** — Iterative reflection loop
9. **Self-Refine** — Output refinement
10. **CRITIC** — Critique-based correction
11. **AutoGen** — Multi-agent coordination
12. **MetaGPT** — Structured multi-agent roles
13. **PAKTON** — Multi-agent + RAG for legal documents

## Running Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker build -t selfaudit .
docker run -it --env-file .env selfaudit python scripts/run_contractnli.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chen2026selfaudit,
  title={SelfAudit: Heterogeneous Multi-Dimensional Reflection for Complex Document Auditing},
  author={Chen, Fei and Zhang, Chenyu and Jiang, Meng and Chen, Yun},
  journal={Applied Sciences},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

The architecture follows patterns from the mingjing (明镜) document auditing backend. We thank the authors of ContractNLI, CUAD, and LegalBench-RAG for making their datasets publicly available.