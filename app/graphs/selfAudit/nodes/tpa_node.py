"""Task Planning Agent (TPA) node.

Decomposes complex audit queries into structured subtasks represented as
(objective, rule, audit_point) triples. Inputs include the user query Q,
the preprocessed document collection D, the predefined rule set M, and
reflection feedback from SRA.
"""

from app.graphs.selfAudit.state.audit_state import AuditState
from app.schemas.audit_schema import AuditPoint
from app.services.llm.prompts import TPA_PROMPT
from app.utils.logger import logger
from app.utils.value_parser import ValueParser


def tpa_node(state: AuditState) -> AuditState:
    """
    Decompose the audit query into (objective, rule, audit_point) triples.

    TPA(Q, D, M, Reflection) -> T = {t_1, ..., t_n}
    where t_i = (obj_i, rule_i, point_i)
    """
    query = state.get("query", "")
    document_text = state.get("document_text", "")
    rules = state.get("rules", [])
    llm_service = state.get("llm_service")

    document_context = document_text[:2000]
    rules_text = "\n".join(f"- {r}" for r in rules) if rules else "No specific rules provided"

    prompt = TPA_PROMPT.substitute(
        query=query,
        document_context=document_context,
        rules=rules_text,
    )

    response = llm_service.generate_completion(prompt) if llm_service else None

    if response:
        triples = ValueParser.parse_audit_triples(response)
        audit_points = []
        for t in triples:
            audit_points.append(
                AuditPoint(
                    objective=t.get("objective", ""),
                    rule=t.get("rule", ""),
                    audit_point=t.get("audit_point", ""),
                )
            )
        if audit_points:
            logger.info(f"TPA generated {len(audit_points)} audit points")
            state["audit_points"] = audit_points
            return state

    # Fallback: single audit point from the query
    logger.warning("TPA fallback: using query as single audit point")
    state["audit_points"] = [
        AuditPoint(
            objective=query,
            rule=rules[0] if rules else "General compliance",
            audit_point=query,
        )
    ]
    return state