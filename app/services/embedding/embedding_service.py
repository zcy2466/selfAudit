"""Embedding service using text-embedding-3-large (3,072 dimensions).

Mirrors mingjing's EmbeddingService pattern at:
    d:/code/Python/mingjing/app/services/embedding/
"""

from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.services.embedding.base import EmbeddingServiceBase
from app.utils.logger import logger


class EmbeddingService(EmbeddingServiceBase):
    """OpenAI-compatible embedding service."""

    def __init__(self, settings: Settings):
        self.model = settings.EMBEDDING_MODEL
        self.client = OpenAIEmbeddings(
            model=self.model,
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
        )

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        logger.debug(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.client.embed_documents(texts)
        return embeddings

    def get_single_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        result = self.get_embeddings([text])
        return result[0] if result else []