"""Heterogeneous Verifier for HCD-EA.

Uses a lightweight model (Qwen2.5-7B-Instruct) with different architecture
and training data than the backbone model (GPT-4o). This separation avoids
circular self-evaluation.

Also computes Cohen's kappa for agreement measurement between primary
and heterogeneous verifiers.
"""

from app.schemas.audit_schema import VerificationResult, Verdict, Evidence
from app.services.llm.llm_service import OpenAIService
from app.services.llm.prompts import HETEROGENEOUS_VERIFY_PROMPT
from app.utils.logger import logger
from app.utils.metrics import compute_cohens_kappa


class HeterogeneousVerifier:
    """Independent verifier using Qwen2.5-7B for HCD-EA heterogeneous verification."""

    def __init__(self, llm_service: OpenAIService):
        self.llm_service = llm_service

    def verify(
        self,
        primary_verdict: Verdict,
        evidence: list[Evidence],
        audit_rule: str,
    ) -> Verdict:
        """
        Perform independent re-verification.

        Args:
            primary_verdict: The verdict from the primary IA model.
            evidence: Evidence used for the verdict.
            audit_rule: The applicable audit rule.

        Returns:
            Independent Verdict from the heterogeneous verifier.
        """
        evidence_text = "\n\n".join(
            f"[{e.chunk_id}] {e.text}" for e in evidence[:5]
        )

        prompt = HETEROGENEOUS_VERIFY_PROMPT.substitute(
            primary_label=primary_verdict.final_label,
            primary_confidence=str(primary_verdict.confidence),
            primary_reasoning=primary_verdict.reasoning,
            evidence=evidence_text,
            rule=audit_rule,
        )

        try:
            response = self.llm_service.generate_completion(prompt)
            if response is None:
                logger.warning("Heterogeneous verifier returned None")
                return primary_verdict

            from app.utils.value_parser import ValueParser
            data = ValueParser.extract_json_from_output(response)
            if data is None:
                return primary_verdict

            return Verdict(
                final_label=data.get("final_label", primary_verdict.final_label),
                confidence=float(data.get("confidence", primary_verdict.confidence)),
                reasoning=data.get("reasoning", "Heterogeneous verification result"),
                iteration=primary_verdict.iteration,
                evidence_chain=primary_verdict.evidence_chain,
            )
        except Exception as e:
            logger.warning(f"Heterogeneous verification failed: {e}")
            return primary_verdict

    @staticmethod
    def compute_agreement(
        primary_labels: list[str],
        heterogeneous_labels: list[str],
        human_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Compute agreement metrics.

        Args:
            primary_labels: Labels from the primary model.
            heterogeneous_labels: Labels from the heterogeneous verifier.
            human_labels: Optional human-annotated gold labels.

        Returns:
            Dict with kappa values:
            - hetero_vs_primary: agreement between the two verifiers
            - hetero_vs_human: agreement between heterogeneous and human (if available)
            - primary_vs_human: agreement between primary and human (if available)
        """
        label_to_int = {"Entailment": 0, "Contradiction": 1, "Not Mentioned": 2}

        primary_int = [label_to_int.get(l, -1) for l in primary_labels]
        hetero_int = [label_to_int.get(l, -1) for l in heterogeneous_labels]

        valid_pairs = [
            (p, h)
            for p, h in zip(primary_int, hetero_int)
            if p >= 0 and h >= 0
        ]
        if not valid_pairs:
            return {"hetero_vs_primary": 0.0}

        p_int, h_int = zip(*valid_pairs)
        results = {
            "hetero_vs_primary": compute_cohens_kappa(list(p_int), list(h_int)),
        }

        if human_labels is not None:
            human_int = [label_to_int.get(l, -1) for l in human_labels]
            valid_human_hetero = [
                (h, hu)
                for h, hu in zip(hetero_int, human_int)
                if h >= 0 and hu >= 0
            ]
            valid_human_primary = [
                (p, hu)
                for p, hu in zip(primary_int, human_int)
                if p >= 0 and hu >= 0
            ]
            if valid_human_hetero:
                hh, hu = zip(*valid_human_hetero)
                results["hetero_vs_human"] = compute_cohens_kappa(list(hh), list(hu))
            if valid_human_primary:
                pp, pu = zip(*valid_human_primary)
                results["primary_vs_human"] = compute_cohens_kappa(list(pp), list(pu))

        return results