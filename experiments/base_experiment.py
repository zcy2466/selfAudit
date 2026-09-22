"""Abstract base class for all experiment runners."""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from app.core.config import Settings
from app.services.llm.factory import CompletionServiceFactory
from app.services.embedding.factory import EmbeddingServiceFactory
from app.services.rerank.rerank_service import RerankClient
from app.services.retrieval.hierarchical_tree import HierarchicalTreeBuilder
from app.services.retrieval.hybrid_index import HybridIndex
from app.services.retrieval.fusion_reranker import FusionReranker
from app.services.retrieval.era_retriever import ERARetriever
from app.services.verifier.dependency_graph import DependencyGraph
from app.services.verifier.heterogeneous import HeterogeneousVerifier
from app.graphs.selfAudit.selfaudit_graph import build_selfaudit_graph, create_initial_state
from app.utils.logger import logger


class BaseExperiment(ABC):
    """Abstract base for SelfAudit experiment runners."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.results_path = Path(settings.RESULTS_PATH)
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.llm_service = CompletionServiceFactory.create_service(settings)
        self.embedding_service = EmbeddingServiceFactory.create_service(settings)
        self.rerank_client = RerankClient(settings)

        # Heterogeneous verifier
        self.heterogeneous_llm_service = CompletionServiceFactory.create_heterogeneous_service(settings)
        self.heterogeneous_verifier = HeterogeneousVerifier(self.heterogeneous_llm_service)

        # Build SelfAudit graph
        self.graph = self._init_graph()

        logger.info(f"Experiment initialized: {self.__class__.__name__}")

    def _init_graph(self):
        """Build the SelfAudit LangGraph with injected services."""
        tree_builder = HierarchicalTreeBuilder(embedding_service=self.embedding_service)
        hybrid_index = HybridIndex()
        fusion_reranker = FusionReranker(rerank_client=self.rerank_client)
        era_retriever = ERARetriever(
            tree=tree_builder,
            index=hybrid_index,
            reranker=fusion_reranker,
            embedding_service=self.embedding_service,
            llm_service=self.llm_service,
        )
        dep_graph = DependencyGraph()

        return build_selfaudit_graph(
            llm_service=self.llm_service,
            embedding_service=self.embedding_service,
            era_retriever=era_retriever,
            dependency_graph=dep_graph,
            heterogeneous_verifier=self.heterogeneous_verifier,
            heterogeneous_llm_service=self.heterogeneous_llm_service,
            settings=self.settings,
        )

    def _create_state(self, query: str, document_text: str, rules: list[str] | None = None, ablation_mode: str = ""):
        """Create an initial AuditState for a single sample."""
        tree_builder = HierarchicalTreeBuilder(embedding_service=self.embedding_service)
        hybrid_index = HybridIndex()
        fusion_reranker = FusionReranker(rerank_client=self.rerank_client)
        era_retriever = ERARetriever(
            tree=tree_builder,
            index=hybrid_index,
            reranker=fusion_reranker,
            embedding_service=self.embedding_service,
            llm_service=self.llm_service,
        )
        dep_graph = DependencyGraph()

        return create_initial_state(
            query=query,
            document_text=document_text,
            rules=rules or [],
            llm_service=self.llm_service,
            embedding_service=self.embedding_service,
            era_retriever=era_retriever,
            dependency_graph=dep_graph,
            heterogeneous_verifier=self.heterogeneous_verifier,
            heterogeneous_llm_service=self.heterogeneous_llm_service,
            settings=self.settings,
            ablation_mode=ablation_mode,
        )

    def _run_selfaudit(self, query: str, document_text: str, rules: list[str] | None = None, ablation_mode: str = "") -> dict:
        """Run the SelfAudit pipeline on a single sample and return results."""
        state = self._create_state(query, document_text, rules, ablation_mode=ablation_mode)

        # Build index for document
        self._build_document_index(state, document_text)

        result = self.graph.invoke(state)

        return {
            "final_label": result.get("verdict", {}).final_label if hasattr(result.get("verdict", {}), "final_label") else "Not Mentioned",
            "confidence": result.get("verdict", {}).confidence if hasattr(result.get("verdict", {}), "confidence") else 0.0,
            "confidence_3d": result.get("confidence_3d", {}),
            "iteration": result.get("iteration", 0),
        }

    def _build_document_index(self, state: dict, document_text: str):
        """Build retrieval index for a document."""
        tree_builder = HierarchicalTreeBuilder(embedding_service=self.embedding_service)
        tree = tree_builder.build(document_text)

        chunks = [c.text for c in tree_builder._collect_chunks(tree)]
        if not chunks:
            chunks = [document_text]

        hybrid_index = HybridIndex()
        embeddings = self.embedding_service.get_embeddings(chunks)
        hybrid_index.build(chunks, embeddings)

        fusion_reranker = FusionReranker(rerank_client=self.rerank_client)
        era_retriever = ERARetriever(
            tree=tree_builder,
            index=hybrid_index,
            reranker=fusion_reranker,
            embedding_service=self.embedding_service,
            llm_service=self.llm_service,
        )
        state["era_retriever"] = era_retriever
        state["document_tree"] = tree

    @abstractmethod
    def run(self) -> dict:
        """Run the experiment and return results."""
        ...

    def save_results(self, results: dict, filename: str):
        """Save experiment results to JSON."""
        filepath = self.results_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {filepath}")

    def log_run_info(self):
        logger.info(f"Experiment: {self.__class__.__name__}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Model: {self.settings.LLM_MODEL}")
        logger.info(f"HCD tau={self.settings.HCD_TAU}, K={self.settings.HCD_K}, alpha={self.settings.HCD_ALPHA}")
        logger.info(f"ERA TopK={self.settings.ERA_TOPK}")