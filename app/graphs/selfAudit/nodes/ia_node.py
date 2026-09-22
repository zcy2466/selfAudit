"""Inspector Agent (IA) node.

Performs auditing based on the subtask list (TPA), combining predefined
rules with evidence retrieved by ERA. Conducts verification along four
dimensions: D1 (Numerical Correctness), D2 (Content Completeness),
D3 (Semantic Consistency), D4 (Risk Analysis).

Result_i = IA(t_i, V_i, E_i, Reflection)
"""

from app.graphs.selfAudit.state.audit_state import AuditState
from app.schemas.audit_schema import AuditPoint, Evidence, VerificationResult
from app.services.llm.prompts import (
    IA_D1_NUMERICAL_PROMPT,
    IA_D2_COMPLETENESS_PROMPT,
    IA_D3_CONSISTENCY_PROMPT,
    IA_D4_RISK_PROMPT,
)
from app.utils.logger import logger
from app.utils.value_parser import ValueParser


def ia_node(state: AuditState) -> AuditState:
    """
    Run 4-dimension verification for each audit point.

    D1: Numerical Correctness
    D2: Content Completeness
    D3: Semantic Consistency
    D4: Risk Analysis
    """
    audit_points = state.get("audit_points", [])
    evidence_map = state.get("evidence_map", {})
    llm_service = state.get("llm_service")

    verification_results: dict[str, list[VerificationResult]] = {}

    for ap in audit_points:
        key = f"{ap.objective}||{ap.audit_point}"
        evidence = evidence_map.get(key, [])
        evidence_text = _format_evidence(evidence)

        results = _verify_dimensions(ap, evidence_text, llm_service)
        verification_results[key] = results

    state["verification_results"] = verification_results
    logger.info(f"IA completed verification for {len(audit_points)} audit points")
    return state


def _format_evidence(evidence: list[Evidence]) -> str:
    """Format evidence chunks into a single text block."""
    if not evidence:
        return "No evidence retrieved."
    return "\n\n---\n\n".join(
        f"[Chunk {i+1}] (score={e.relevance_score:.3f}): {e.text}"
        for i, e in enumerate(evidence[:5])
    )


def _verify_dimensions(
    ap: AuditPoint,
    evidence_text: str,
    llm_service,
) -> list[VerificationResult]:
    """Run all four dimension checks for a single audit point."""
    dimensions = [
        ("D1", IA_D1_NUMERICAL_PROMPT),
        ("D2", IA_D2_COMPLETENESS_PROMPT),
        ("D3", IA_D3_CONSISTENCY_PROMPT),
        ("D4", IA_D4_RISK_PROMPT),
    ]

    results = []

    def _check_dim(dim_name: str, prompt_template) -> VerificationResult:
        prompt = prompt_template.substitute(
            objective=ap.objective,
            rule=ap.rule,
            evidence=evidence_text,
        )
        response = llm_service.generate_completion(prompt) if llm_service else None
        if response is None:
            return VerificationResult(
                dimension=dim_name,
                passed=False,
                confidence=0.0,
                reasoning="LLM service unavailable",
            )

        data = ValueParser.extract_json_from_output(response)
        if data is None:
            return VerificationResult(
                dimension=dim_name,
                passed=False,
                confidence=0.5,
                reasoning=response[:200],
            )

        return VerificationResult(
            dimension=data.get("dimension", dim_name),
            passed=data.get("passed", False),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

    # Run dimensions sequentially to avoid rate limits
    for dim_name, prompt_template in dimensions:
        result = _check_dim(dim_name, prompt_template)
        results.append(result)

    return results