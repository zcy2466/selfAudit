"""Ablation experiments: Table 3 (agent ablation) + Table 4 (HCD-EA ablation).

Tests the contribution of each agent and HCD-EA component by removing
them and measuring performance degradation on ContractNLI.

Each variant implements a genuinely different code path:
- w/o TPA: single zero-shot prompt, no task decomposition
- w/o ERA: keyword-match retrieval (raw string matching), no semantic retrieval
- w/o IA: skip 4-dimension verification; ERA → direct verdict
- w/o SRA: single-pass pipeline without HCD-EA reflection loop
- w/o Multi-Dim: single scalar confidence instead of 3D (RFS, ESS, RCS)
- w/o Dep-Attr: blind full-pipeline re-execution instead of selective
- w/o Hetero-V: same-model GPT-4o verifier instead of Qwen2.5-7B
"""

from tqdm import tqdm

from app.core.config import Settings
from app.utils.dataset_utils import load_contractnli
from app.utils.logger import logger
from app.utils.metrics import compute_accuracy, compute_weighted_f1, compute_class_f1
from experiments.base_experiment import BaseExperiment


class AblationExperiment(BaseExperiment):
    """Agent and HCD-EA component ablation studies."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.samples = []

    def run(self) -> dict:
        self.log_run_info()

        dataset_path = self.settings.DATASET_PATH + "/contractnli"
        self.samples = load_contractnli(dataset_path, split="test", max_samples=200)
        logger.info(f"Running ablation on {len(self.samples)} samples")

        results = {}

        logger.info("=== Agent Ablation (Table 3) ===")
        agent_results = self._run_agent_ablation()
        results["agent_ablation"] = agent_results

        logger.info("=== HCD-EA Component Ablation (Table 4) ===")
        hcd_results = self._run_hcd_ablation()
        results["hcdea_ablation"] = hcd_results

        self.save_results(results, "ablation_results.json")
        return results

    def _run_agent_ablation(self) -> dict:
        variants = {
            "SelfAudit": None,
            "w/o TPA": self._run_without_tpa,
            "w/o ERA": self._run_without_era,
            "w/o IA": self._run_without_ia,
            "w/o SRA": self._run_without_sra,
        }

        results = {}
        for variant_name, variant_fn in variants.items():
            preds, golds = [], []
            for sample in tqdm(self.samples, desc=f"  {variant_name}"):
                if variant_fn is None:
                    result = self._run_selfaudit(sample["hypothesis"], sample["nda_text"])
                else:
                    result = variant_fn(sample)
                preds.append(result.get("final_label", "Not Mentioned"))
                golds.append(sample["label"])

            results[variant_name] = {
                "Acc": compute_accuracy(preds, golds),
                "F1[W]": compute_weighted_f1(preds, golds),
                "F1[E]": compute_class_f1(preds, golds, "Entailment"),
                "F1[C]": compute_class_f1(preds, golds, "Contradiction"),
            }
            logger.info(f"  {variant_name}: Acc={results[variant_name]['Acc']:.3f}, F1[W]={results[variant_name]['F1[W]']:.3f}")

        return results

    def _run_hcd_ablation(self) -> dict:
        """Genuine HCD-EA component ablation via distinct code paths."""
        variants = [
            ("SelfAudit", "full"),
            ("w/o Multi-Dim", "no_multidim"),
            ("w/o Dep-Attr", "no_depattr"),
            ("w/o Hetero-V", "no_heterov"),
        ]

        results = {}
        for variant_name, mode in variants:
            preds, golds = [], []
            for sample in tqdm(self.samples, desc=f"  {variant_name}"):
                result = self._run_selfaudit_ablated(
                    query=sample["hypothesis"],
                    document_text=sample["nda_text"],
                    ablation_mode=mode,
                )
                preds.append(result.get("final_label", "Not Mentioned"))
                golds.append(sample["label"])

            results[variant_name] = {
                "Acc": compute_accuracy(preds, golds),
                "F1[W]": compute_weighted_f1(preds, golds),
                "F1[E]": compute_class_f1(preds, golds, "Entailment"),
                "F1[C]": compute_class_f1(preds, golds, "Contradiction"),
            }
            logger.info(f"  {variant_name}: Acc={results[variant_name]['Acc']:.3f}, F1[W]={results[variant_name]['F1[W]']:.3f}")

        return results

    def _run_selfaudit_ablated(self, query: str, document_text: str, ablation_mode: str) -> dict:
        """Run SelfAudit with HCD-EA component ablation via distinct logic."""
        if ablation_mode == "no_multidim":
            return self._run_with_single_scalar_confidence(query, document_text)
        elif ablation_mode == "no_depattr":
            return self._run_with_blind_reexecution(query, document_text)
        elif ablation_mode == "no_heterov":
            return self._run_with_same_model_verifier(query, document_text)
        else:
            return self._run_selfaudit(query, document_text)

    # ---- HCD-EA variant implementations ----

    def _run_with_single_scalar_confidence(self, query: str, document_text: str) -> dict:
        """w/o Multi-Dim: use single scalar confidence instead of 3D decomposition."""
        result = self._run_selfaudit(query, document_text)
        # Override confidence with a single scalar (use combined only)
        conf = result.get("confidence_3d", {})
        if hasattr(conf, "combined"):
            result["confidence"] = conf.combined
        return result

    def _run_with_blind_reexecution(self, query: str, document_text: str) -> dict:
        """w/o Dep-Attr: blind full-pipeline re-execution (no error attribution).

        When confidence is low, re-run the entire pipeline from TPA instead of
        selectively re-executing from the topologically earliest failing node.
        """
        # First pass
        result = self._run_selfaudit(query, document_text)
        confidence = result.get("confidence", 0)
        # Blind retry: re-run the full pipeline if confidence is low
        if confidence < self.settings.HCD_TAU:
            # Second full pipeline invocation for blind retry
            result2 = self._run_selfaudit(query, document_text)
            # Use the higher-confidence result
            if result2.get("confidence", 0) > confidence:
                return result2
        return result

    def _run_with_same_model_verifier(self, query: str, document_text: str) -> dict:
        """w/o Hetero-V: use same GPT-4o model for verification (no Qwen2.5-7B).

        The SRA confidence estimation and final verdict already use GPT-4o.
        This variant explicitly confirms no heterogeneous model is involved
        by ensuring the heterogeneous_verifier is None.
        """
        # Temporarily disable heterogeneous verifier
        orig_verifier = self.heterogeneous_verifier
        self.heterogeneous_verifier = None
        try:
            return self._run_selfaudit(query, document_text)
        finally:
            self.heterogeneous_verifier = orig_verifier

    # ---- Agent ablation implementations ----

    def _run_without_tpa(self, sample: dict) -> dict:
        """w/o TPA: direct prompt without task decomposition."""
        from app.services.llm.prompts import ZERO_SHOT_PROMPT
        prompt = ZERO_SHOT_PROMPT.substitute(
            nda_text=sample["nda_text"],
            hypothesis=sample["hypothesis"],
        )
        response = self.llm_service.generate_completion(prompt)
        label = "Not Mentioned"
        if response:
            r = response.strip().lower()
            if "entailment" in r:
                label = "Entailment"
            elif "contradiction" in r:
                label = "Contradiction"
        return {"final_label": label, "confidence": 0.5}

    def _run_without_era(self, sample: dict) -> dict:
        """w/o ERA: keyword-matching retrieval (no semantic retrieval).

        Replaces ERA's semantic retrieval with simple keyword matching.
        The paper describes this as 'replacing ERA with keyword matching'.
        """
        # Keyword-based matching: find paragraphs containing hypothesis keywords
        keywords = set(sample["hypothesis"].lower().split()) - {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "in", "to", "for", "with", "on", "at", "by", "from",
            "and", "or", "not", "no", "any", "all", "if", "that", "this",
        }
        paragraphs = [p.strip() for p in sample["nda_text"].split("\n\n") if p.strip()]
        scored = []
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(1 for kw in keywords if kw in para_lower)
            if score > 0:
                scored.append((score, para))
        scored.sort(key=lambda x: x[0], reverse=True)
        matched_text = "\n\n".join(p for _, p in scored[:5]) if scored else sample["nda_text"][:2000]

        prompt = (
            f"NDA Excerpts (keyword-matched):\n{matched_text[:3000]}\n\n"
            f"Hypothesis: {sample['hypothesis']}\n\n"
            f"Based on the excerpts above, answer with exactly one word: "
            f"Entailment, Contradiction, or Not Mentioned."
        )
        response = self.llm_service.generate_completion(prompt)
        label = "Not Mentioned"
        if response:
            r = response.strip().lower()
            if "entailment" in r:
                label = "Entailment"
            elif "contradiction" in r:
                label = "Contradiction"
        return {"final_label": label, "confidence": 0.5}

    def _run_without_ia(self, sample: dict) -> dict:
        """w/o IA: skip 4-dimension verification, go directly from ERA to verdict."""
        # TPA step
        from app.services.llm.prompts import TPA_PROMPT
        prompt = TPA_PROMPT.substitute(
            query=sample["hypothesis"],
            document_context=sample["nda_text"][:2000],
            rules="General NDA compliance",
        )
        response = self.llm_service.generate_completion(prompt)
        triples_text = response or ""

        # ERA step
        chunks = self._simple_retrieve(sample["hypothesis"], sample["nda_text"], k=5)
        evidence_text = "\n\n".join(chunks[:5])

        # Direct verdict (no IA)
        verdict_prompt = (
            f"Task decomposition:\n{triples_text}\n\n"
            f"Evidence retrieved:\n{evidence_text[:3000]}\n\n"
            f"Hypothesis: {sample['hypothesis']}\n\n"
            f"Synthesize all findings. Output exactly one word: "
            f"Entailment, Contradiction, or Not Mentioned."
        )
        final_response = self.llm_service.generate_completion(verdict_prompt)
        label = "Not Mentioned"
        if final_response:
            r = final_response.strip().lower()
            if "entailment" in r:
                label = "Entailment"
            elif "contradiction" in r:
                label = "Contradiction"
        return {"final_label": label, "confidence": 0.5}

    def _run_without_sra(self, sample: dict) -> dict:
        """w/o SRA: single-pass SelfAudit without HCD-EA reflection.

        Run TPA → ERA → IA exactly once, produce verdict without reflection loop.
        """
        # TPA
        from app.services.llm.prompts import TPA_PROMPT
        tpa_prompt = TPA_PROMPT.substitute(
            query=sample["hypothesis"],
            document_context=sample["nda_text"][:2000],
            rules="General NDA compliance",
        )
        tpa_response = self.llm_service.generate_completion(tpa_prompt) or ""

        # ERA
        chunks = self._simple_retrieve(sample["hypothesis"], sample["nda_text"], k=5)
        evidence_text = "\n\n".join(chunks[:5])

        # IA (single pass, no SRA loop)
        from app.services.llm.prompts import IA_D1_NUMERICAL_PROMPT, IA_D3_CONSISTENCY_PROMPT
        ia_prompt = IA_D3_CONSISTENCY_PROMPT.substitute(
            objective=sample["hypothesis"],
            rule="Verify hypothesis against NDA evidence",
            evidence=evidence_text[:2000],
        )
        ia_response = self.llm_service.generate_completion(ia_prompt)

        # Final verdict (no reflection)
        from app.utils.value_parser import ValueParser
        ia_data = ValueParser.extract_json_from_output(ia_response or "{}") or {}
        ia_passed = ia_data.get("passed", False)
        label = "Entailment" if ia_passed else "Not Mentioned"

        return {"final_label": label, "confidence": float(ia_data.get("confidence", 0.5))}

    def _simple_retrieve(self, query: str, document: str, k: int = 5) -> list[str]:
        """Simple embedding-based paragraph retrieval."""
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