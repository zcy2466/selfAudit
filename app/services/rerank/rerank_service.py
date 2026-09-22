"""Rerank client for fusion reranking of retrieval results.

Mirrors mingjing's RerankClient pattern at:
    d:/code/Python/mingjing/app/services/rerank/rerank_service.py
"""

import requests
from pydantic import BaseModel

from app.core.config import Settings
from app.utils.logger import logger


class RerankedDocument(BaseModel):
    """A document with a relevance score after reranking."""
    content: str
    metadata: dict = {}
    relevance_score: float = 0.0
    index: int = 0


class RerankClient:
    """Client for calling an external reranker API."""

    def __init__(self, settings: Settings):
        self.rerank_url = settings.RERANK_BASE_URL
        self.model = settings.RERANK_MODEL

    def rerank(
        self, query: str, documents: list[dict], top_n: int = 10
    ) -> list[RerankedDocument]:
        """Rerank documents by relevance to the query."""
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": [d.get("content", "") for d in documents],
            "top_n": top_n,
        }

        try:
            response = requests.post(self.rerank_url, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json().get("results", [])

            reranked = []
            for r in results:
                idx = r.get("index", 0)
                doc = documents[idx] if idx < len(documents) else {}
                reranked.append(
                    RerankedDocument(
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                        relevance_score=r.get("relevance_score", 0.0),
                        index=idx,
                    )
                )
            return reranked
        except Exception as e:
            logger.warning(f"Rerank API call failed: {e}, returning original order")
            return [
                RerankedDocument(
                    content=d.get("content", ""),
                    metadata=d.get("metadata", {}),
                    relevance_score=0.0,
                    index=i,
                )
                for i, d in enumerate(documents[:top_n])
            ]