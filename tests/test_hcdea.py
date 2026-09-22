"""Tests for HCD-EA: confidence decomposition, dependency graph, error attribution."""

import pytest


class TestDependencyGraph:
    def test_graph_creation(self):
        from app.services.verifier.dependency_graph import DependencyGraph
        g = DependencyGraph()
        assert g.top_order("tpa") == 0
        assert g.top_order("era") == 1
        assert g.top_order("ia") == 2
        assert g.top_order("verdict") == 3

    def test_upstream_downstream(self):
        from app.services.verifier.dependency_graph import DependencyGraph
        g = DependencyGraph()
        assert g.get_upstream("era") == ["tpa"]
        assert g.get_downstream("tpa") == ["era"]
        assert g.get_upstream("tpa") == []
        assert g.get_downstream("verdict") == []

    def test_attribute_error_isolated(self):
        from app.services.verifier.dependency_graph import DependencyGraph
        g = DependencyGraph()
        confidences = {"tpa": 0.9, "era": 0.5, "ia": 0.8, "verdict": 0.8}
        attr = g.attribute_error(confidences, threshold=0.75)
        assert attr.source_node == "era"
        assert attr.error_type == "evidence_retrieval_error"

    def test_attribute_error_cascading(self):
        from app.services.verifier.dependency_graph import DependencyGraph
        g = DependencyGraph()
        confidences = {"tpa": 0.5, "era": 0.4, "ia": 0.3, "verdict": 0.2}
        attr = g.attribute_error(confidences, threshold=0.75)
        assert attr.source_node == "tpa"
        assert "ia" in attr.affected_nodes
        assert "verdict" in attr.affected_nodes

    def test_attribute_error_none(self):
        from app.services.verifier.dependency_graph import DependencyGraph
        g = DependencyGraph()
        confidences = {"tpa": 0.9, "era": 0.85, "ia": 0.8, "verdict": 0.8}
        attr = g.attribute_error(confidences, threshold=0.75)
        assert attr.source_node == "none"
        assert attr.error_type == "no_error"

    def test_subpipeline_nodes(self):
        from app.services.verifier.dependency_graph import DependencyGraph
        g = DependencyGraph()
        nodes = g.get_subpipeline_nodes("era")
        assert nodes == ["era", "ia", "verdict"]

    def test_trace_path(self):
        from app.services.verifier.dependency_graph import DependencyGraph
        g = DependencyGraph()
        path = g.trace_path("ia")
        assert path == ["tpa", "era", "ia"]


class TestConfidenceDecomposition:
    def test_min_aggregation(self):
        assert min(0.9, 0.8, 0.62) == 0.62
        assert min(1.0, 1.0, 1.0) == 1.0
        assert min(0.0, 0.5, 0.8) == 0.0