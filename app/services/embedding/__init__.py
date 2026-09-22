"""Embedding service layer — text-embedding-3-large (3072d) client."""

from app.services.embedding.base import EmbeddingServiceBase
from app.services.embedding.embedding_service import EmbeddingService
from app.services.embedding.factory import EmbeddingServiceFactory

__all__ = ["EmbeddingServiceBase", "EmbeddingService", "EmbeddingServiceFactory"]