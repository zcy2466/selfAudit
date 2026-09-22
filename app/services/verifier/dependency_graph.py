"""Dependency graph G = (V, E) for HCD-EA error attribution.

The 4-node graph (TPA -> ERA -> IA -> Verdict) encodes the directional
dependencies of the SelfAudit pipeline. When SRA detects low confidence,
this graph backtracks to attribute errors to specific upstream nodes.
"""

from app.schemas.audit_schema import ErrorAttribution


class DependencyGraph:
    """Fixed 4-node dependency graph for the SelfAudit pipeline."""

    NODE_ORDER = ["tpa", "era", "ia", "verdict"]
    EDGES = [
        ("tpa", "era"),
        ("era", "ia"),
        ("ia", "verdict"),
    ]

    def __init__(self):
        self._upstream: dict[str, list[str]] = {}
        self._downstream: dict[str, list[str]] = {}
        self._topological_order: dict[str, int] = {}

        for i, node in enumerate(self.NODE_ORDER):
            self._topological_order[node] = i
            self._upstream[node] = []
            self._downstream[node] = []

        for src, dst in self.EDGES:
            self._upstream[dst].append(src)
            self._downstream[src].append(dst)

    def get_upstream(self, node_id: str) -> list[str]:
        """Return nodes that feed into this node."""
        return self._upstream.get(node_id, [])

    def get_downstream(self, node_id: str) -> list[str]:
        """Return nodes affected by this node."""
        return self._downstream.get(node_id, [])

    def top_order(self, node_id: str) -> int:
        """Return topological position (lower = earlier)."""
        return self._topological_order.get(node_id, -1)

    def attribute_error(
        self,
        node_confidences: dict[str, float],
        threshold: float = 0.75,
    ) -> ErrorAttribution:
        """
        Find the topologically earliest failing node.

        Args:
            node_confidences: Dict mapping node_id to aggregate confidence.
            threshold: Confidence threshold for considering a node failed.

        Returns:
            ErrorAttribution with the identified source node.
        """
        failure_set = {
            node for node, conf in node_confidences.items() if conf < threshold
        }

        if not failure_set:
            return ErrorAttribution(
                source_node="none",
                error_type="no_error",
                confidence_breakdown=node_confidences,
            )

        earliest_node = min(failure_set, key=lambda n: self._topological_order.get(n, 999))

        affected = []
        queue = [earliest_node]
        visited = {earliest_node}
        while queue:
            current = queue.pop(0)
            for downstream in self._downstream.get(current, []):
                if downstream not in visited:
                    visited.add(downstream)
                    queue.append(downstream)
            if current != earliest_node:
                affected.append(current)

        error_type = self._classify_error(earliest_node)

        return ErrorAttribution(
            source_node=earliest_node,
            error_type=error_type,
            affected_nodes=affected,
            confidence_breakdown=node_confidences,
        )

    @staticmethod
    def _classify_error(node_id: str) -> str:
        """Classify error type based on which node failed."""
        error_types = {
            "tpa": "task_decomposition_error",
            "era": "evidence_retrieval_error",
            "ia": "verification_error",
            "verdict": "final_judgment_error",
        }
        return error_types.get(node_id, "unknown_error")

    def get_subpipeline_nodes(self, from_node: str) -> list[str]:
        """
        Return nodes in the sub-pipeline from 'from_node' to 'verdict'.
        Used for selective re-execution.
        """
        nodes = []
        current = from_node
        while current:
            nodes.append(current)
            downstream = self._downstream.get(current, [])
            current = downstream[0] if downstream else None
        return nodes

    def trace_path(self, node_id: str) -> list[str]:
        """Return the full path from root (tpa) to this node."""
        path = []
        visited = set()

        def _dfs(current):
            if current in visited or current == node_id:
                path.append(current)
                return True
            visited.add(current)
            for child in self._downstream.get(current, []):
                if _dfs(child):
                    path.append(current)
                    return True
            return False

        _dfs("tpa")
        path.reverse()
        return path