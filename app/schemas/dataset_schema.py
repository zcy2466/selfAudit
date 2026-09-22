"""Pydantic schemas for dataset samples."""

from pydantic import BaseModel, Field


class ContractNLISample(BaseModel):
    """A single sample from the ContractNLI dataset."""
    document_id: str = Field(description="Unique document identifier")
    nda_text: str = Field(description="Full NDA document text")
    hypothesis: str = Field(description="Legal hypothesis to verify")
    label: str = Field(
        description="Gold label: Entailment, Contradiction, or Not Mentioned"
    )
    spans: list[tuple[int, int]] = Field(
        default_factory=list, description="Supporting evidence spans"
    )


class CUADSample(BaseModel):
    """A single sample from the CUAD dataset."""
    contract_id: str = Field(description="Unique contract identifier")
    contract_text: str = Field(description="Full contract text")
    category: str = Field(description="One of 41 contract review categories")
    query: str = Field(description="Natural language query for this category")
    gold_spans: list[tuple[int, int]] = Field(
        default_factory=list, description="Gold-standard answer spans"
    )


class RAGSample(BaseModel):
    """A single sample from the LegalBench-RAG dataset."""
    query_id: str = Field(description="Unique query identifier")
    domain: str = Field(
        description="Legal domain: NDA, M&A, commercial, or privacy"
    )
    query: str = Field(description="Retrieval query text")
    relevant_doc_ids: list[str] = Field(
        default_factory=list, description="IDs of relevant documents"
    )
    corpus: dict[str, str] = Field(
        default_factory=dict, description="Document ID -> text mapping"
    )