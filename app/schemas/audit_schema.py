"""Pydantic schemas for SelfAudit internal data structures."""

from pydantic import BaseModel, Field


class AuditPoint(BaseModel):
    """A single subtask from TPA: (objective, rule, audit_point) triple."""
    objective: str = Field(description="What needs to be checked")
    rule: str = Field(description="The criterion used to evaluate the objective")
    audit_point: str = Field(description="The specific items to be examined")


class Evidence(BaseModel):
    """A retrieved evidence chunk from ERA."""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="The text content of the chunk")
    relevance_score: float = Field(description="Relevance score after reranking")
    source_document: str = Field(default="", description="Source document identifier")
    section_location: str = Field(default="", description="Section/chapter location")
    page_number: int = Field(default=0, description="Page number in source document")


class VerificationResult(BaseModel):
    """Result of a single dimension verification by IA."""
    dimension: str = Field(description="D1/D2/D3/D4")
    passed: bool = Field(description="Whether the verification passed")
    confidence: float = Field(description="Confidence score in [0, 1]")
    reasoning: str = Field(description="Natural language reasoning")


class Confidence3D(BaseModel):
    """Three-dimensional confidence vector from HCD-EA."""
    rfs: float = Field(default=0.0, description="Retrieval Fidelity Score [0, 1]")
    ess: float = Field(default=0.0, description="Evidence-Support Score [0, 1]")
    rcs: float = Field(default=0.0, description="Rule-Compliance Score [0, 1]")

    @property
    def combined(self) -> float:
        return min(self.rfs, self.ess, self.rcs)

    def to_dict(self) -> dict[str, float]:
        return {"rfs": self.rfs, "ess": self.ess, "rcs": self.rcs, "combined": self.combined}


class Verdict(BaseModel):
    """Final audit verdict from SelfAudit."""
    final_label: str = Field(
        description="One of: Entailment, Contradiction, Not Mentioned"
    )
    confidence: float = Field(description="Aggregate confidence score")
    reasoning: str = Field(description="Natural language reasoning chain")
    iteration: int = Field(default=0, description="Number of HCD-EA iterations used")
    evidence_chain: list[Evidence] = Field(
        default_factory=list, description="Supporting evidence"
    )


class ErrorAttribution(BaseModel):
    """Result of dependency-aware error attribution."""
    source_node: str = Field(description="The topologically earliest failing node")
    error_type: str = Field(description="Nature of the error")
    affected_nodes: list[str] = Field(
        default_factory=list, description="Downstream nodes affected"
    )
    confidence_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Per-node confidence values"
    )


class DependencyNode(BaseModel):
    """A node in the dependency graph G = (V, E)."""
    node_id: str = Field(description="Unique node identifier")
    node_type: str = Field(description="tpa, era, ia, or verdict")
    parents: list[str] = Field(default_factory=list, description="Upstream node IDs")