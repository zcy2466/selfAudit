"""Hybrid dense-sparse index for document retrieval.

Combines embedding-based dense search (cosine similarity) with
BM25 sparse keyword search for complementary retrieval signals.
"""

import numpy as np
from rank_bm25 import BM25Okapi


class HybridIndex:
    """Dual-index combining dense embeddings and BM25 sparse retrieval."""

    def __init__(self):
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] | None = None
        self.bm25: BM25Okapi | None = None
        self.metadata: list[dict] = []

    def build(self, chunks: list[str], embeddings: list[list[float]] | None = None, metadata: list[dict] | None = None) -> None:
        """
        Build the hybrid index from text chunks.

        Args:
            chunks: List of text chunks to index.
            embeddings: Corresponding dense embeddings. If None, only BM25 is available.
            metadata: Optional metadata for each chunk.
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.metadata = metadata or [{}] * len(chunks)

        tokenized = [self._tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def dense_search(
        self, query_embedding: list[float], k: int = 10
    ) -> list[tuple[int, float]]:
        """
        Cosine similarity search using dense embeddings.

        Returns:
            List of (chunk_index, similarity_score) sorted by descending score.
        """
        if self.embeddings is None or not self.embeddings:
            return []

        query_vec = np.array(query_embedding)
        chunk_matrix = np.array(self.embeddings)
        norms = np.linalg.norm(chunk_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-8
        similarities = np.dot(chunk_matrix, query_vec) / norms

        top_indices = np.argsort(similarities)[::-1][:k]
        return [(int(i), float(similarities[i])) for i in top_indices]

    def sparse_search(
        self, query_text: str, k: int = 10
    ) -> list[tuple[int, float]]:
        """
        BM25 keyword search.

        Returns:
            List of (chunk_index, bm25_score) sorted by descending score.
        """
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def get_chunk(self, idx: int) -> str:
        """Retrieve chunk text by index."""
        if 0 <= idx < len(self.chunks):
            return self.chunks[idx]
        return ""

    def get_metadata(self, idx: int) -> dict:
        """Retrieve chunk metadata by index."""
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return {}