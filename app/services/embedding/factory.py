"""Factory for embedding service instances."""

from app.core.config import Settings
from app.services.embedding.embedding_service import EmbeddingService


class EmbeddingServiceFactory:
    """Factory for embedding services."""

    @staticmethod
    def create_service(settings: Settings) -> EmbeddingService:
        return EmbeddingService(settings)