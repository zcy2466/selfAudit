"""Evidence Retrieval Agent (ERA) node.

Performs semantic retrieval over the document collection D based on each
subtask from TPA. Constructs a hierarchical semantic tree, maps each
subtask to a retrieval query, computes cosine similarity, applies hybrid
indexing and fusion reranking, and returns Top-k=10 evidence chunks.
"""

from app.graphs.selfAudit.state.audit_state import AuditState
from app.utils.logger import logger


def era_node(state: AuditState) -> AuditState:
    """
    Retrieve evidence for each audit point.

    For each subtask t_i:
      q_i = Embed(t_i), d_j = Embed(s_j)
      Candidates(t_i) = TopK[cos(q_i, d_j)]

    ERA(t_i, D, Reflection) -> (V, E)
    """
    audit_points = state.get("audit_points", [])
    era_retriever = state.get("era_retriever")

    if not audit_points or era_retriever is None:
        state["evidence_map"] = {}
        return state

    logger.info(f"ERA retrieving evidence for {len(audit_points)} audit points")
    document_tree = state.get("document_tree")
    evidence_map = era_retriever.retrieve(audit_points, tree=document_tree)
    state["evidence_map"] = evidence_map

    total_evidence = sum(len(v) for v in evidence_map.values())
    logger.info(f"ERA retrieved {total_evidence} evidence chunks total")
    return state