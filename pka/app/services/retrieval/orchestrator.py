from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ..index.bm25 import BM25IndexService
from ..index.embed import EmbeddingService
from ..index.vector import VectorIndexService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: int
    document_id: int
    path: str
    title: str
    content: str
    start_line: Optional[int]
    end_line: Optional[int]
    page_no: Optional[int]
    token_count: Optional[int]
    score_bm25: Optional[float] = None
    score_embed: Optional[float] = None
    distance: Optional[float] = None
    rank_bm25: Optional[int] = None
    rank_embed: Optional[int] = None
    rationale: Optional[str] = None


class RetrievalOrchestrator:
    """Coordinate BM25, vector search, optional reranking, and post-processing."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        bm25_service: BM25IndexService,
        vector_service: VectorIndexService,
        *,
        max_bm25: int = 50,
        max_vector: int = 50,
        final_limit: int = 12,
        diversity_cap: int = 3,
    ) -> None:
        self.embedding_service = embedding_service
        self.bm25_service = bm25_service
        self.vector_service = vector_service
        self.max_bm25 = max_bm25
        self.max_vector = max_vector
        self.final_limit = final_limit
        self.diversity_cap = diversity_cap
        self.reranker_enabled = False

    def retrieve(self, question: str, session: Session) -> List[RetrievalResult]:
        question = question.strip()
        if not question:
            return []

        query_vector = self.embedding_service.embed_query(question)
        bm25_hits = self.bm25_service.search(question, limit=self.max_bm25)
        vector_hits = self.vector_service.search(session, query_vector, limit=self.max_vector)

        merged: Dict[int, RetrievalResult] = {}

        def ensure_result(chunk_id: int, base_payload: dict) -> RetrievalResult:
            if chunk_id not in merged:
                merged[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=int(base_payload.get("document_id", 0)),
                    path=base_payload.get("path", ""),
                    title=base_payload.get("title", ""),
                    content=base_payload.get("content", ""),
                    start_line=base_payload.get("start_line"),
                    end_line=base_payload.get("end_line"),
                    page_no=base_payload.get("page_no"),
                    token_count=base_payload.get("token_count"),
                )
            return merged[chunk_id]

        for rank, hit in enumerate(bm25_hits):
            try:
                chunk_id = int(hit.get("chunk_id"))
            except (TypeError, ValueError):
                logger.warning("Skipping BM25 hit with invalid chunk_id: %s", hit)
                continue
            result = ensure_result(chunk_id, hit)
            result.score_bm25 = float(hit.get("score_bm25") or 0.0)
            result.rank_bm25 = rank

        for rank, hit in enumerate(vector_hits):
            chunk_id = int(hit["chunk_id"])
            result = ensure_result(chunk_id, hit)
            result.score_embed = hit.get("score_embed")
            result.distance = hit.get("distance")
            result.rank_embed = rank

        ordered_ids: List[int] = []
        for hit in bm25_hits:
            try:
                chunk_id = int(hit.get("chunk_id"))
            except (TypeError, ValueError):
                continue
            if chunk_id not in ordered_ids:
                ordered_ids.append(chunk_id)
        for hit in vector_hits:
            chunk_id = int(hit["chunk_id"])
            if chunk_id not in ordered_ids:
                ordered_ids.append(chunk_id)

        selected: List[RetrievalResult] = []
        doc_counts: Dict[int, int] = {}
        for chunk_id in ordered_ids:
            result = merged.get(chunk_id)
            if not result:
                continue
            doc_counts.setdefault(result.document_id, 0)
            if doc_counts[result.document_id] >= self.diversity_cap:
                continue
            selected.append(result)
            doc_counts[result.document_id] += 1
            if len(selected) >= self.final_limit:
                break

        return selected
