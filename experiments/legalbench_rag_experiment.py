"""LegalBench-RAG experiment: Table 6 (retrieval evaluation).

Evaluates the ERA retrieval component independently on four legal domains:
NDAs, M&A agreements, commercial contracts, and privacy policies.

Metrics: Precision@k, Recall@k for k in {1, 2, 4, 8, 16, 32, 64}.
"""

import numpy as np
from tqdm import tqdm

from app.core.config import Settings
from app.utils.dataset_utils import load_legalbench_rag
from app.utils.logger import logger
from app.utils.metrics import compute_precision_at_k, compute_recall_at_k
from experiments.base_experiment import BaseExperiment


class LegalBenchRAGExperiment(BaseExperiment):
    """LegalBench-RAG retrieval evaluation (Table 6)."""

    K_VALUES = [1, 2, 4, 8, 16, 32, 64]
    REPORT_K = [1, 8, 64]

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def run(self) -> dict:
        self.log_run_info()

        dataset_path = self.settings.DATASET_PATH + "/legalbench_rag"
        domains_data = load_legalbench_rag(dataset_path)

        results = {}

        for domain, queries in domains_data.items():
            if not queries:
                continue
            logger.info(f"Evaluating domain '{domain}' with {len(queries)} queries")

            domain_results = self._evaluate_domain(domain, queries)
            results[domain] = domain_results

        self.save_results(results, "legalbench_rag_results.json")
        return results

    def _evaluate_domain(self, domain: str, queries: list[dict]) -> dict:
        """Evaluate retrieval for one legal domain."""
        precision_sums = {k: [] for k in self.K_VALUES}
        recall_sums = {k: [] for k in self.K_VALUES}

        for q in tqdm(queries, desc=f"  {domain}"):
            # Build index for this domain's corpus
            corpus = q.get("corpus", {})
            if not corpus:
                continue

            doc_ids = list(corpus.keys())
            doc_texts = [corpus[did] for did in doc_ids]

            from app.services.retrieval.hybrid_index import HybridIndex
            from app.services.retrieval.fusion_reranker import FusionReranker

            hybrid_index = HybridIndex()
            embeddings = self.embedding_service.get_embeddings(doc_texts)
            hybrid_index.build(doc_texts, embeddings)

            fusion_reranker = FusionReranker(rerank_client=self.rerank_client)

            # Retrieve
            query_text = q["query"]
            query_emb = self.embedding_service.get_single_embedding(query_text)

            reranked = fusion_reranker.retrieve(
                query=query_text,
                query_embedding=query_emb,
                hybrid_index=hybrid_index,
                k=max(self.K_VALUES),
            )

            retrieved_ids = []
            seen = set()
            for rd in reranked:
                if rd.index < len(doc_ids) and doc_ids[rd.index] not in seen:
                    retrieved_ids.append(doc_ids[rd.index])
                    seen.add(doc_ids[rd.index])

            relevant = set(q.get("relevant_doc_ids", []))

            for k in self.K_VALUES:
                precision_sums[k].append(compute_precision_at_k(retrieved_ids, relevant, k))
                recall_sums[k].append(compute_recall_at_k(retrieved_ids, relevant, k))

        result = {}
        for k in self.REPORT_K:
            p_vals = precision_sums[k]
            r_vals = recall_sums[k]
            result[f"P@{k}"] = float(np.mean(p_vals)) * 100 if p_vals else 0.0
            result[f"R@{k}"] = float(np.mean(r_vals)) * 100 if r_vals else 0.0

        logger.info(
            f"  Domain '{domain}': P@1={result.get('P@1', 0):.2f}%, "
            f"R@1={result.get('R@1', 0):.2f}%, "
            f"P@8={result.get('P@8', 0):.2f}%, "
            f"R@8={result.get('R@8', 0):.2f}%"
        )

        return result