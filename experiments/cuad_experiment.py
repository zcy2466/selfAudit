"""CUAD experiment: Table 2.

Evaluates SelfAudit on CUAD (41 contract review categories).
Each category = binary span extraction task.

Metrics: AUPR, Micro-F1, Precision, Recall
"""

import numpy as np
from tqdm import tqdm

from app.core.config import Settings
from app.utils.dataset_utils import load_cuad
from app.utils.logger import logger
from app.utils.metrics import compute_aupr, compute_f1
from experiments.base_experiment import BaseExperiment


class CUADExperiment(BaseExperiment):
    """CUAD benchmark experiment (Table 2)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.samples = []

    def run(self) -> dict:
        self.log_run_info()

        dataset_path = self.settings.DATASET_PATH + "/cuad"
        self.samples = load_cuad(dataset_path)
        logger.info(f"Running CUAD on {len(self.samples)} samples")

        results_table = {}

        methods = {
            "GPT-4o ZS": self._run_zs,
            "Standard RAG": self._run_rag,
            "Self-RAG": self._run_self_rag,
            "PAKTON": self._run_pakton,
            "SelfAudit": self._run_selfaudit_wrapper,
        }

        for method_name, method_fn in methods.items():
            logger.info(f"--- Running {method_name} ---")

            # Group by category
            by_category: dict[str, list[dict]] = {}
            for s in self.samples:
                by_category.setdefault(s["category"], []).append(s)

            global_tp = 0
            global_fp = 0
            global_fn = 0
            all_auprs = []
            all_precisions = []
            all_recalls = []

            for category, cat_samples in tqdm(by_category.items(), desc=method_name):
                precisions, recalls, f1s = [], [], []
                for sample in cat_samples:
                    pred_spans = method_fn(sample)
                    gold_spans = sample.get("gold_spans", [])
                    p, r, f1 = self._span_metrics(pred_spans, gold_spans)
                    precisions.append(p)
                    recalls.append(r)
                    f1s.append(f1)

                    # Accumulate for true Micro-F1
                    pred_tokens = set()
                    gold_tokens = set()
                    for s, e in pred_spans:
                        pred_tokens.update(range(s, e))
                    for s, e in gold_spans:
                        gold_tokens.update(range(s, e))
                    global_tp += len(pred_tokens & gold_tokens)
                    global_fp += len(pred_tokens - gold_tokens)
                    global_fn += len(gold_tokens - pred_tokens)

                avg_p = np.mean(precisions) if precisions else 0
                avg_r = np.mean(recalls) if recalls else 0
                avg_f1 = np.mean(f1s) if f1s else 0

                # AUPR approximation per category
                sorted_pairs = sorted(zip(recalls, precisions))
                if sorted_pairs:
                    rs, ps = zip(*sorted_pairs)
                    aupr = compute_aupr(list(ps), list(rs))
                else:
                    aupr = 0.0

                all_precisions.append(avg_p)
                all_recalls.append(avg_r)
                all_auprs.append(aupr)

            micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
            micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

            results_table[method_name] = {
                "AUPR": float(np.mean(all_auprs)) if all_auprs else 0.0,
                "Micro-F1": micro_f1,
                "Precision": micro_p,
                "Recall": micro_r,
            }
            logger.info(
                f"{method_name}: AUPR={results_table[method_name]['AUPR']:.3f}, "
                f"F1={results_table[method_name]['Micro-F1']:.3f}"
            )

        self.save_results(results_table, "cuad_results.json")
        return results_table

    def _run_selfaudit_wrapper(self, sample: dict) -> list[tuple[int, int]]:
        result = self._run_selfaudit(
            query=sample["query"],
            document_text=sample["contract_text"],
        )
        return self._extract_spans(result, sample["contract_text"])

    def _run_zs(self, sample: dict) -> list[tuple[int, int]]:
        prompt = (
            f"Extract the exact text span from the contract that answers this query.\n"
            f"Query: {sample['query']}\n"
            f"Category: {sample['category']}\n\n"
            f"Contract:\n{sample['contract_text'][:4000]}\n\n"
            f"Output the exact matching text. If not found, output 'None'."
        )
        response = self.llm_service.generate_completion(prompt)
        return self._match_span(response, sample["contract_text"])

    def _run_rag(self, sample: dict) -> list[tuple[int, int]]:
        chunks = self._simple_retrieve(sample["query"], sample["contract_text"])
        prompt = (
            f"Query: {sample['query']}\n"
            f"Relevant excerpts:\n{chr(10).join(chunks[:3])}\n"
            f"Output the exact matching text span. If not found, output 'None'."
        )
        response = self.llm_service.generate_completion(prompt)
        return self._match_span(response, sample["contract_text"])

    def _run_self_rag(self, sample: dict) -> list[tuple[int, int]]:
        """Self-RAG: retrieve, critique relevance, re-retrieve if needed."""
        chunks = self._simple_retrieve(sample["query"], sample["contract_text"], k=5)
        chunks_text = "\n".join(chunks[:3])

        critique_prompt = (
            f"Query: {sample['query']}\n"
            f"Retrieved excerpts:\n{chunks_text}\n\n"
            f"Are these excerpts relevant to the query? Answer YES or NO."
        )
        critique = self.llm_service.generate_completion(critique_prompt)
        is_relevant = critique and "yes" in critique.strip().lower()

        if not is_relevant:
            refined_query = f"Find specific contract clauses about: {sample['query']}"
            chunks2 = self._simple_retrieve(refined_query, sample["contract_text"], k=5)
            chunks_text = "\n".join(chunks2[:3])

        prompt = (
            f"Query: {sample['query']}\n"
            f"Relevant excerpts:\n{chunks_text}\n"
            f"Output the exact matching text span. If not found, output 'None'."
        )
        response = self.llm_service.generate_completion(prompt)
        return self._match_span(response, sample["contract_text"])

    def _run_pakton(self, sample: dict) -> list[tuple[int, int]]:
        """PAKTON: orchestrated retrieval + reasoning for contract analysis."""
        chunks = self._simple_retrieve(sample["query"], sample["contract_text"], k=5)
        evidence_text = "\n".join(chunks[:5])

        orchestrator_prompt = (
            f"As an audit orchestrator for contract review, find the exact text "
            f"that answers this query:\n"
            f"Query: {sample['query']}\n"
            f"Category: {sample['category']}\n\n"
            f"Contract excerpts:\n{evidence_text}\n\n"
            f"Output the exact matching text span from the contract. "
            f"If not found, output 'None'."
        )
        response = self.llm_service.generate_completion(orchestrator_prompt)
        return self._match_span(response, sample["contract_text"])

    @staticmethod
    def _span_metrics(
        pred_spans: list[tuple[int, int]],
        gold_spans: list[tuple[int, int]],
    ) -> tuple[float, float, float]:
        """Token-level span metrics."""
        pred_tokens = set()
        gold_tokens = set()
        for s, e in pred_spans:
            pred_tokens.update(range(s, e))
        for s, e in gold_spans:
            gold_tokens.update(range(s, e))
        tp = len(pred_tokens & gold_tokens)
        fp = len(pred_tokens - gold_tokens)
        fn = len(gold_tokens - pred_tokens)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    @staticmethod
    def _extract_spans(result: dict, text: str) -> list[tuple[int, int]]:
        """Extract spans from SelfAudit result."""
        evidence = result.get("evidence_chain", [])
        spans = []
        for ev in evidence:
            idx = text.find(ev.text)
            if idx >= 0:
                spans.append((idx, idx + len(ev.text)))
        return spans

    @staticmethod
    def _match_span(response: str | None, text: str) -> list[tuple[int, int]]:
        if not response or response.strip() == "None":
            return []
        idx = text.find(response.strip())
        if idx >= 0:
            return [(idx, idx + len(response.strip()))]
        return []

    def _simple_retrieve(self, query: str, document: str, k: int = 5) -> list[str]:
        import numpy as np
        query_emb = self.embedding_service.get_single_embedding(query)
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
        if not paragraphs:
            return [document]
        para_embs = self.embedding_service.get_embeddings(paragraphs)
        query_vec = np.array(query_emb)
        scores = []
        for i, emb in enumerate(para_embs):
            vec = np.array(emb)
            sim = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8))
            scores.append((sim, paragraphs[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scores[:k]]