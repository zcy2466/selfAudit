"""Main SelfAudit LangGraph StateGraph.

Assembles the TPA -> ERA -> IA -> SRA pipeline with conditional edges
for HCD-EA iteration. When SRA detects low confidence, execution loops
back to the error-attributed upstream node for selective re-execution.

Mirrors mingjing's graph builder pattern at:
    d:/code/Python/mingjing/app/graphs/fact_judge/fact_judge_graph.py
"""

from langgraph.graph import StateGraph, END
from langgraph.constants import START

from app.graphs.selfAudit.state.audit_state import AuditState
from app.graphs.selfAudit.nodes.tpa_node import tpa_node
from app.graphs.selfAudit.nodes.era_node import era_node
from app.graphs.selfAudit.nodes.ia_node import ia_node
from app.graphs.selfAudit.nodes.sra_node import sra_node
from app.utils.logger import logger


def _route_after_sra(state: AuditState) -> str:
    """Conditional routing after SRA: loop back or end."""
    reexecute_from = state.get("reexecute_from", "")
    iteration = state.get("iteration", 0)
    settings = state.get("settings")
    k_max = settings.HCD_K if settings else 3

    if reexecute_from and iteration < k_max:
        logger.info(f"Routing back to '{reexecute_from}' for re-execution")
        return reexecute_from
    return END


def _route_after_era(state: AuditState) -> str:
    """Always route ERA -> IA."""
    return "ia"


def _route_after_ia(state: AuditState) -> str:
    """Always route IA -> SRA."""
    return "sra"


def build_selfaudit_graph(
    llm_service,
    embedding_service,
    era_retriever,
    dependency_graph,
    heterogeneous_verifier=None,
    heterogeneous_llm_service=None,
    settings=None,
):
    """
    Build the SelfAudit LangGraph StateGraph.

    Graph structure:
        START -> TPA -> ERA -> IA -> SRA -> (loop or END)

    Conditional edges:
        SRA -> TPA  (if tpa is the root cause)
        SRA -> ERA  (if era is the root cause)
        SRA -> IA   (if ia is the root cause)
        SRA -> END  (if confidence >= tau or iterations exhausted)
    """
    workflow = StateGraph(AuditState)

    # Register nodes
    workflow.add_node("tpa", tpa_node)
    workflow.add_node("era", era_node)
    workflow.add_node("ia", ia_node)
    workflow.add_node("sra", sra_node)

    # Entry point
    workflow.add_edge(START, "tpa")

    # Forward edges
    workflow.add_edge("tpa", "era")
    workflow.add_conditional_edges("era", _route_after_era, {"ia": "ia"})
    workflow.add_conditional_edges("ia", _route_after_ia, {"sra": "sra"})

    # SRA conditional routing: loop or end
    workflow.add_conditional_edges(
        "sra",
        _route_after_sra,
        {
            "tpa": "tpa",
            "era": "era",
            "ia": "ia",
            END: END,
        },
    )

    compiled_graph = workflow.compile()

    logger.info("SelfAudit graph compiled successfully")
    return compiled_graph


def create_initial_state(
    query: str,
    document_text: str,
    rules: list[str],
    llm_service,
    embedding_service,
    era_retriever,
    dependency_graph,
    heterogeneous_verifier=None,
    heterogeneous_llm_service=None,
    settings=None,
) -> AuditState:
    """Create the initial AuditState with all injected dependencies."""
    return AuditState(
        query=query,
        document_text=document_text,
        rules=rules,
        audit_points=[],
        evidence_map={},
        verification_results={},
        iteration=0,
        reexecute_from="",
        llm_service=llm_service,
        embedding_service=embedding_service,
        era_retriever=era_retriever,
        dependency_graph=dependency_graph,
        heterogeneous_verifier=heterogeneous_verifier,
        heterogeneous_llm_service=heterogeneous_llm_service,
        settings=settings,
    )