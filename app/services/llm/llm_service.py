"""LLM service with retry and exponential backoff.

Mirrors mingjing's QwenService pattern at:
    d:/code/Python/mingjing/app/services/llm/llm_service.py
"""

import random
import time
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.utils.logger import logger
from app.utils.value_parser import ValueParser


class OpenAIService(CompletionService):
    """OpenAI-compatible LLM service with retry and rate-limit handling."""

    def __init__(self, settings: Settings, model_name: str | None = None):
        self.model = model_name or settings.LLM_MODEL
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL
        self.temperature = settings.LLM_TEMPERATURE
        self.client = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def generate_completion(self, prompt: str) -> str | None:
        """Generate a completion with retry and rate-limit backoff."""
        logger.info(f"[{self.model}] Generating completion...")
        max_retries = 5
        base_delay = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.invoke(prompt).content
                if response is None or response == "None":
                    logger.warning("Received None response from LLM")
                    return None
                return response
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    delay = min(base_delay * (2 ** (attempt - 1)), 30)
                    delay *= 0.5 + random.random()
                    logger.warning(
                        f"[RateLimit] Attempt {attempt}/{max_retries} failed. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.exception("Unexpected error while calling LLM")
                    raise e

        raise RuntimeError("LLM rate limit exceeded after all retries")

    def generate_structured(self, prompt: str, max_attempts: int = 5) -> dict:
        """Generate a completion and parse it as structured JSON."""
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = self.client.invoke(prompt).content
                logger.debug(f"LLM Raw Response (Attempt {attempt + 1}): {response}")
                json_res = ValueParser.extract_json_from_output(response)
                if not json_res:
                    raise ValueError("Empty or invalid JSON structure")
                logger.info(f"Validation success at attempt {attempt + 1}")
                return json_res
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt == max_attempts - 1:
                    logger.warning("Maximum retry attempts reached")

        return {
            "result": 2,
            "reasoning": f"LLM validation failed: {last_error or 'unknown error'}",
        }