"""Fusion reranker combining dense and sparse retrieval results.

Uses Reciprocal Rank Fusion (RRF) to merge dense and sparse results,
then applies an external reranker model for final ordering.
"""

from app.services.rerank.rerank_service import RerankClient, RerankedDocument
from app.utils.logger import logger


class FusionReranker:
    """Fuses dense + sparse retrieval results and reranks them."""

    def __init__(self, rerank_client: RerankClient | None = None):
        self.rerank_client = rerank_client

    def fuse(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        k_rrf: int = 60,
    ) -> list[tuple[int, float]]:
        """
        Combine dense and sparse rankings using Reciprocal Rank Fusion.

        Args:
            dense_results: (chunk_idx, similarity) from dense search.
            sparse_results: (chunk_idx, bm25_score) from sparse search.
            k_rrf: RRF constant (default 60 as commonly used).

        Returns:
            List of (chunk_idx, fused_score) sorted by descending score.
        """
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)

        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def rerank(
        self,
        query: str,
        chunks: list[str],
        metadata: list[dict] | None = None,
        top_k: int = 10,
    ) -> list[RerankedDocument]:
        """
        Rerank fused candidates using the external reranker model.

        Args:
            query: Search query.
            chunks: List of chunk texts to rerank.
            metadata: Optional metadata per chunk.
            top_k: Number of top results to return.

        Returns:
            List of RerankedDocument sorted by relevance_score descending.
        """
        if metadata is None:
            metadata = [{}] * len(chunks)

        documents = [
            {"content": text, "metadata": meta}
            for text, meta in zip(chunks, metadata)
        ]

        if self.rerank_client is not None:
            try:
                return self.rerank_client.rerank(query, documents, top_n=top_k)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, returning RRF-only order")

        return [
            RerankedDocument(
                content=doc["content"],
                metadata=doc["metadata"],
                relevance_score=1.0 - i / max(len(documents), 1),
                index=i,
            )
            for i, doc in enumerate(documents[:top_k])
        ]

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        hybrid_index,
        k: int = 10,
    ) -> list[RerankedDocument]:
        """
        Full retrieval pipeline: dense + sparse -> RRF fuse -> rerank.

        Args:
            query: Natural language query.
            query_embedding: Dense embedding of the query.
            hybrid_index: HybridIndex instance.
            k: Number of final results.

        Returns:
            List of RerankedDocument.
        """
        dense = hybrid_index.dense_search(query_embedding, k=k * 3)
        sparse = hybrid_index.sparse_search(query, k=k * 3)

        fused = self.fuse(dense, sparse)

        candidate_indices = [idx for idx, _ in fused[: k * 3]]
        candidate_chunks = [hybrid_index.get_chunk(i) for i in candidate_indices]
        candidate_meta = [hybrid_index.get_metadata(i) for i in candidate_indices]

        return self.rerank(query, candidate_chunks, candidate_meta, top_k=k)