"""AuditState TypedDict for the SelfAudit LangGraph.

Mirrors mingjing's state pattern at:
    d:/code/Python/mingjing/app/graphs/fact_judge/state/fact_judge_state.py
"""

from typing import Any, TypedDict

from app.schemas.audit_schema import (
    AuditPoint,
    Confidence3D,
    ErrorAttribution,
    Evidence,
    Verdict,
    VerificationResult,
)


class AuditState(TypedDict, total=False):
    """State flowing through the SelfAudit agent pipeline."""

    # Inputs
    query: str
    document_text: str
    rules: list[str]

    # TPA outputs
    audit_points: list[AuditPoint]

    # ERA outputs
    evidence_map: dict[str, list[Evidence]]

    # IA outputs
    verification_results: dict[str, list[VerificationResult]]

    # SRA / HCD-EA outputs
    confidence_3d: Confidence3D
    verdict: Verdict
    iteration: int
    error_attribution: ErrorAttribution | None
    reexecute_from: str
    heterogeneous_verdict: Verdict | None
    heterogeneous_agreement: float | None

    # Hierarchical semantic tree (built once, reused across re-executions)
    document_tree: Any

    # Injected dependencies (not serialized)
    llm_service: Any
    heterogeneous_llm_service: Any
    embedding_service: Any
    era_retriever: Any
    dependency_graph: Any
    heterogeneous_verifier: Any
    settings: Any