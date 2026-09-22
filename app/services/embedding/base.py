"""Abstract base class for embedding services."""

from abc import ABC, abstractmethod


class EmbeddingServiceBase(ABC):
    """Abstract base for text embedding services."""

    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def get_single_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        ...