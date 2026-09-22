"""ERA orchestration: hierarchical tree + hybrid index + fusion reranking.

The Evidence Retrieval Agent (ERA) performs the full retrieval pipeline:
1. Query expansion via LLM
2. Hierarchical tree search
3. Hybrid dense+sparse index search
4. Fusion reranking

Returns Top-k=10 evidence chunks per audit point.
"""

from app.schemas.audit_schema import AuditPoint, Evidence
from app.services.llm.llm_service import OpenAIService
from app.services.llm.prompts import ERA_QUERY_PROMPT
from app.services.retrieval.hierarchical_tree import HierarchicalTreeBuilder, TreeNode
from app.services.retrieval.hybrid_index import HybridIndex
from app.services.retrieval.fusion_reranker import FusionReranker
from app.utils.logger import logger


class ERARetriever:
    """ERA orchestration for multi-query evidence retrieval."""

    def __init__(
        self,
        tree: HierarchicalTreeBuilder,
        index: HybridIndex,
        reranker: FusionReranker,
        embedding_service,
        llm_service: OpenAIService | None = None,
    ):
        self.tree = tree
        self.index = index
        self.reranker = reranker
        self.embedding_service = embedding_service
        self.llm_service = llm_service

    def retrieve(
        self, audit_points: list[AuditPoint], tree: TreeNode | None = None
    ) -> dict[str, list[Evidence]]:
        """
        Retrieve evidence for each audit point.

        Args:
            audit_points: List of AuditPoint from TPA.
            tree: Pre-built hierarchical tree (optional).

        Returns:
            Dict mapping audit_point.objective+audit_point to list of Evidence.
        """
        evidence_map: dict[str, list[Evidence]] = {}
        for ap in audit_points:
            key = f"{ap.objective}||{ap.audit_point}"
            queries = self._expand_queries(ap)
            all_evidence: list[Evidence] = []

            for query in queries:
                query_emb = self.embedding_service.get_single_embedding(query)

                if tree is not None:
                    tree_chunks = self.tree.search_tree(tree, query_emb, k=5)
                    for chunk in tree_chunks:
                        # Compute actual cosine similarity for tree chunks
                        import numpy as np
                        if chunk.embedding is not None:
                            chunk_vec = np.array(chunk.embedding)
                            query_vec = np.array(query_emb)
                            sim = float(
                                np.dot(query_vec, chunk_vec)
                                / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8)
                            )
                        else:
                            sim = 0.0
                        all_evidence.append(
                            Evidence(
                                chunk_id=chunk.id,
                                text=chunk.text,
                                relevance_score=sim,
                                source_document="",
                                section_location=chunk.section_title,
                                page_number=chunk.page_number,
                            )
                        )

                reranked = self.reranker.retrieve(
                    query=query,
                    query_embedding=query_emb,
                    hybrid_index=self.index,
                    k=5,
                )
                for rd in reranked:
                    all_evidence.append(
                        Evidence(
                            chunk_id=f"idx_{rd.index}",
                            text=rd.content,
                            relevance_score=rd.relevance_score,
                            source_document=rd.metadata.get("source", ""),
                            section_location=rd.metadata.get("section", ""),
                            page_number=rd.metadata.get("page", 0),
                        )
                    )

            all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
            evidence_map[key] = all_evidence[:10]

        return evidence_map

    def _expand_queries(self, audit_point: AuditPoint) -> list[str]:
        """Generate multiple search queries for an audit point."""
        queries = [
            f"{audit_point.objective} {audit_point.audit_point}",
            audit_point.rule,
        ]

        if self.llm_service is not None:
            try:
                prompt = ERA_QUERY_PROMPT.substitute(
                    objective=audit_point.objective,
                    rule=audit_point.rule,
                    audit_point=audit_point.audit_point,
                )
                response = self.llm_service.generate_completion(prompt)
                if response:
                    llm_queries = [
                        q.strip() for q in response.split("\n") if q.strip()
                    ]
                    queries.extend(llm_queries[:3])
            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}")

        return list(dict.fromkeys(queries))