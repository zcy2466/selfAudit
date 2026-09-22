"""Abstract base class for LLM completion services."""

from abc import ABC, abstractmethod
from typing import Any


class CompletionService(ABC):
    """Abstract base for LLM completion services."""

    @abstractmethod
    def generate_completion(self, prompt: str) -> str | None:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    def generate_structured(self, prompt: str, max_attempts: int = 5) -> dict:
        """Generate a completion and parse it as structured JSON."""
        ...