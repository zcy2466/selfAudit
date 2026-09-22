"""Factory for creating LLM service instances."""

from app.core.config import Settings
from app.services.llm.llm_service import OpenAIService


class CompletionServiceFactory:
    """Factory for LLM completion services."""

    @staticmethod
    def create_service(
        settings: Settings, model_name: str | None = None
    ) -> OpenAIService:
        """Create an OpenAIService with optional model override."""
        return OpenAIService(settings, model_name=model_name)

    @staticmethod
    def create_heterogeneous_service(settings: Settings) -> OpenAIService:
        """Create the heterogeneous verifier service (Qwen2.5-7B)."""
        return OpenAIService(
            settings,
            model_name=settings.HETEROGENEOUS_LLM_MODEL,
        )