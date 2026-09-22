# Datasets

Place downloaded datasets here. Run `python scripts/download_datasets.py` to fetch them.

## ContractNLI
- **Source**: Koreeda and Manning (2021)
- **Contents**: 2,091 test samples (NDA + hypothesis → Entailment/Contradiction/Not Mentioned)
- **Files**: `contractnli/train.json`, `contractnli/test.json`

## CUAD (Contract Understanding Atticus Dataset)
- **Source**: Hendrycks et al. (2021)
- **Contents**: 41 contract review categories, binary span extraction
- **Files**: `cuad/cuad_data.json`

## LegalBench-RAG
- **Source**: Pipitone et al. (2024)
- **Contents**: Four legal domains: NDAs, M&A, commercial contracts, privacy policies
- **Files**: `legalbench_rag/<domain>/queries.json`