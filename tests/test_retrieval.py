"""Tests for hybrid index, fusion reranker, and tree builder."""

import pytest


class TestHybridIndex:
    def test_build_and_search(self):
        from app.services.retrieval.hybrid_index import HybridIndex
        chunks = [
            "Confidentiality obligation lasts 5 years",
            "Governing law is Delaware",
            "Party shall not disclose to third parties",
            "Payment due within 30 days",
        ]
        index = HybridIndex()
        index.build(chunks)

        results = index.sparse_search("confidentiality obligation", k=2)
        assert len(results) > 0
        assert results[0][0] == 0  # First chunk should match best

    def test_dense_search(self):
        from app.services.retrieval.hybrid_index import HybridIndex
        import numpy as np
        chunks = ["hello world", "goodbye world", "hello there"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
        ]
        index = HybridIndex()
        index.build(chunks, embeddings)

        results = index.dense_search([1.0, 0.0, 0.0], k=2)
        assert results[0][0] == 0  # "hello world" closest

    def test_empty_index(self):
        from app.services.retrieval.hybrid_index import HybridIndex
        index = HybridIndex()
        assert index.sparse_search("anything") == []
        assert index.dense_search([1.0, 0.0]) == []


class TestFusionReranker:
    def test_rrf_fusion(self):
        from app.services.retrieval.fusion_reranker import FusionReranker
        reranker = FusionReranker()
        dense = [(0, 0.9), (1, 0.7), (2, 0.5)]
        sparse = [(1, 0.8), (0, 0.6), (3, 0.4)]
        fused = reranker.fuse(dense, sparse)
        assert len(fused) > 0
        assert fused[0][0] in [0, 1]  # Chunk 0 or 1 should be top

    def test_rerank_no_client(self):
        from app.services.retrieval.fusion_reranker import FusionReranker
        reranker = FusionReranker()
        results = reranker.rerank("query", ["a", "b", "c"], top_k=2)
        assert len(results) == 2
        assert results[0].content == "a"


class TestHierarchicalTreeBuilder:
    def test_build_simple_tree(self):
        from app.services.retrieval.hierarchical_tree import HierarchicalTreeBuilder
        builder = HierarchicalTreeBuilder()
        tree = builder.build("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
        chunks = builder._collect_chunks(tree)
        assert len(chunks) == 3

    def test_build_with_sections(self):
        from app.services.retrieval.hierarchical_tree import HierarchicalTreeBuilder
        builder = HierarchicalTreeBuilder()
        sections = [
            {
                "title": "Section 1",
                "content": "Content of section one.",
                "page": 1,
                "subsections": [
                    {"title": "1.1", "content": "Subsection content.", "page": 1},
                ],
            },
            {"title": "Section 2", "content": "Content of section two.", "page": 2},
        ]
        tree = builder.build("", sections=sections)
        assert len(tree.children) == 2
        assert len(tree.children[0].children) == 1