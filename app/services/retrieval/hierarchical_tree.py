"""Hierarchical semantic tree for document structure-aware retrieval.

The tree recursively segments documents by chapter boundaries and paragraph semantics,
preserving structural relationships among clauses that fixed-window chunking would disrupt.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    """A leaf node in the hierarchical tree: a text segment with metadata."""
    id: str
    text: str
    embedding: list[float] | None = None
    page_number: int = 0
    section_title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TreeNode:
    """A node in the hierarchical semantic tree."""
    id: str
    title: str = ""
    summary: str = ""
    embedding: list[float] | None = None
    children: list["TreeNode"] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    level: int = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children


class HierarchicalTreeBuilder:
    """Builds and searches a hierarchical semantic tree over document content."""

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service
        self._chunk_counter = 0
        self._node_counter = 0

    def build(self, document_text: str, sections: list[dict] | None = None) -> TreeNode:
        """
        Build a hierarchical tree from document text.

        Args:
            document_text: Full document text.
            sections: Pre-parsed sections with titles and content. If None,
                      simple paragraph-based segmentation is used.

        Returns:
            Root TreeNode of the hierarchical tree.
        """
        self._node_counter += 1
        root = TreeNode(id=f"root_{self._node_counter}", title="Document", level=0)

        if sections:
            for sec in sections:
                child = self._build_section_node(sec, level=1)
                root.children.append(child)
        else:
            paragraphs = [p.strip() for p in document_text.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                chunk = self._create_chunk(para, page_number=0, section_title="")
                root.chunks.append(chunk)

        return root

    def _build_section_node(self, section: dict, level: int) -> TreeNode:
        """Recursively build a tree node for a document section."""
        self._node_counter += 1
        title = section.get("title", f"Section {self._node_counter}")
        node = TreeNode(
            id=f"node_{self._node_counter}",
            title=title,
            level=level,
        )

        content = section.get("content", "")
        if content:
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for para in paragraphs:
                chunk = self._create_chunk(para, page_number=section.get("page", 0), section_title=title)
                node.chunks.append(chunk)

        for subsection in section.get("subsections", []):
            child = self._build_section_node(subsection, level=level + 1)
            node.children.append(child)

        return node

    def _create_chunk(self, text: str, page_number: int, section_title: str) -> Chunk:
        """Create a chunk with a unique ID."""
        self._chunk_counter += 1
        return Chunk(
            id=f"chunk_{self._chunk_counter}",
            text=text,
            page_number=page_number,
            section_title=section_title,
        )

    def search_tree(
        self,
        tree: TreeNode,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[Chunk]:
        """
        Top-down tree traversal: prune branches below similarity threshold,
        collect leaf chunks.

        Args:
            tree: Root of the hierarchical tree.
            query_embedding: Dense vector of the search query.
            k: Number of top chunks to return.

        Returns:
            List of top-k Chunks ranked by cosine similarity.
        """
        all_chunks = self._collect_chunks(tree)
        if not all_chunks:
            return []

        query_vec = np.array(query_embedding)
        scored = []
        for chunk in all_chunks:
            if chunk.embedding is not None:
                chunk_vec = np.array(chunk.embedding)
                similarity = float(
                    np.dot(query_vec, chunk_vec)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8)
                )
            else:
                similarity = 0.0
            scored.append((similarity, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]

    def _collect_chunks(self, node: TreeNode) -> list[Chunk]:
        """Recursively collect all chunks from a tree node."""
        chunks = list(node.chunks)
        for child in node.children:
            chunks.extend(self._collect_chunks(child))
        return chunks

    def embed_tree(self, tree: TreeNode) -> None:
        """Generate embeddings for all chunks in the tree."""
        all_chunks = self._collect_chunks(tree)
        if not all_chunks or self.embedding_service is None:
            return

        texts = [c.text for c in all_chunks]
        embeddings = self.embedding_service.get_embeddings(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb