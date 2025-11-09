from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class HealthProbe(BaseModel):
    name: str
    healthy: bool
    detail: Optional[str] = None
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    status: str
    probes: List[HealthProbe]


class ChatRequest(BaseModel):
    question: str
    mode: str = "synthesize"


class CitationSource(BaseModel):
    id: str
    loc: str


class ConflictEntry(BaseModel):
    claim: str
    sources: List[CitationSource]


class ChatAnswer(BaseModel):
    abstain: bool
    answer: str
    bullets: List[str] = Field(default_factory=list)
    conflicts: List[ConflictEntry] = Field(default_factory=list)
    sources: List[CitationSource] = Field(default_factory=list)


class ContextSnippetModel(BaseModel):
    chunk_id: int
    document_id: int
    citation: str
    rationale: str
    content: str
    score_bm25: Optional[float] = None
    score_embed: Optional[float] = None


class ChatResponse(BaseModel):
    run_id: UUID
    latency_ms: int
    answer: ChatAnswer
    context: List[ContextSnippetModel] = Field(default_factory=list)
    question: str
    mode: str
    llm_version: str
    prompt_version: str
    template_hash: str


class DocumentChunkModel(BaseModel):
    id: int
    ordinal: Optional[int]
    preview: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    page_no: Optional[int] = None
    token_count: Optional[int] = None


class DocumentResponse(BaseModel):
    id: int
    path: str
    title: str
    type: str
    size: int
    sha256: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    confidentiality_tag: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    chunk_count: int
    chunks: List[DocumentChunkModel] = Field(default_factory=list)


class RunSummaryModel(BaseModel):
    run_id: UUID
    question: str
    mode: str
    started_at: datetime
    latency_ms: Optional[int] = None
    abstained: bool
