from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.logger import logger


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Backbone LLM (TPA, ERA, IA)
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_TEMPERATURE: float = 0.0

    # Heterogeneous verifier (SRA)
    HETEROGENEOUS_LLM_MODEL: str = "qwen2.5-7b-instruct"
    HETEROGENEOUS_LLM_API_KEY: str = ""
    HETEROGENEOUS_LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"

    # Reranker
    RERANK_MODEL: str = "bge-reranker-large"
    RERANK_BASE_URL: str = "http://localhost:9997/v1/rerank"

    # HCD-EA parameters
    HCD_TAU: float = 0.75
    HCD_K: int = 3
    HCD_ALPHA: float = 0.5

    # ERA parameters
    ERA_TOPK: int = 10

    # Cross-model robustness backbones
    BACKBONE_LIST: list[str] = [
        "gpt-4o",
        "qwen2.5-72b-instruct",
        "llama-3.3-70b-instruct",
        "deepseek-v4",
    ]

    # Paths
    DATASET_PATH: str = "./data"
    RESULTS_PATH: str = "./results"

    # Misc
    RANDOM_SEED: int = 42


@lru_cache()
def get_settings() -> Settings:
    logger.info("Loading SelfAudit config settings from environment...")
    return Settings()