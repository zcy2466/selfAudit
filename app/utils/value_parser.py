import json
import re

from app.utils.logger import logger


class ValueParser:
    """JSON extraction and validation from LLM outputs."""

    @classmethod
    def extract_json_from_output(cls, output: str) -> dict | list | None:
        """
        Extract and parse JSON from LLM output that may contain markdown fencing.
        Returns None if no valid JSON is found.
        """
        cleaned_output = re.sub(r'```(json)?', '', output, flags=re.IGNORECASE)

        for match in re.finditer(r'\{|\[', cleaned_output):
            start_pos = match.start()
            try:
                obj, _ = json.JSONDecoder().raw_decode(cleaned_output[start_pos:])
                return obj
            except json.JSONDecodeError:
                continue
        return None

    @classmethod
    def parse_audit_triples(cls, output: str) -> list[dict]:
        """
        Parse TPA output into a list of (objective, rule, audit_point) triples.
        """
        data = cls.extract_json_from_output(output)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "triples" in data:
            return data["triples"]
        if isinstance(data, dict) and "subtasks" in data:
            return data["subtasks"]
        logger.warning(f"Unexpected TPA output format: {output[:200]}")
        return []

    @classmethod
    def parse_verdict(cls, output: str) -> dict:
        """Parse verification result into a structured dict."""
        data = cls.extract_json_from_output(output)
        if data is None:
            return {"result": 2, "reasoning": "Failed to parse LLM output"}
        return data

    @classmethod
    def parse_confidence(cls, output: str) -> dict:
        """Parse SRA confidence decomposition output."""
        data = cls.extract_json_from_output(output)
        if data is None:
            return {"rfs": 0.0, "ess": 0.0, "rcs": 0.0}
        return data