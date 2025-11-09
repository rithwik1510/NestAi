from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import List

from sqlalchemy.orm import Session

from ..index.bm25 import BM25IndexService
from ..index.embed import EmbeddingService
from ..index.vector import VectorIndexService
from .context_builder import ContextBuilder, ContextSnippet
from .orchestrator import RetrievalOrchestrator, RetrievalResult
from .store import RetrievalStore


@dataclass
class RetrievalOutcome:
    run_id: uuid.UUID
    latency_ms: int
    contexts: List[RetrievalResult]
    snippets: List[ContextSnippet]


class RetrievalService:
    """High-level retrieval workflow with persistence."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        bm25_service: BM25IndexService,
        vector_service: VectorIndexService,
        context_builder: ContextBuilder | None = None,
        *,
        max_bm25: int = 50,
        max_vector: int = 50,
        final_limit: int = 12,
        diversity_cap: int = 3,
    ) -> None:
        self.orchestrator = RetrievalOrchestrator(
            embedding_service=embedding_service,
            bm25_service=bm25_service,
            vector_service=vector_service,
            max_bm25=max_bm25,
            max_vector=max_vector,
            final_limit=final_limit,
            diversity_cap=diversity_cap,
        )
        self.context_builder = context_builder or ContextBuilder()
        self.embedding_service = embedding_service
        self.bm25_service = bm25_service
        self.vector_service = vector_service

    def retrieve(
        self,
        session: Session,
        *,
        question: str,
        mode: str,
        llm_version: str,
        prompt_version: str,
        template_hash: str,
    ) -> RetrievalOutcome:
        start = time.perf_counter()
        contexts = self.orchestrator.retrieve(question, session)
        latency_ms = int((time.perf_counter() - start) * 1000)
        snippets = self.context_builder.build(contexts)

        store = RetrievalStore(session)
        run_id = store.create_run(
            question=question,
            mode=mode,
            llm_version=llm_version,
            prompt_version=prompt_version,
            template_hash=template_hash,
        )
        store.write_contexts(run_id, contexts)

        return RetrievalOutcome(
            run_id=run_id, latency_ms=latency_ms, contexts=list(contexts), snippets=snippets
        )

    def finalize(
        self,
        session: Session,
        *,
        run_id: uuid.UUID,
        answer_json: dict,
        latency_ms: int,
        abstained: bool,
    ) -> None:
        store = RetrievalStore(session)
        store.write_answer(run_id, answer_json)
        store.finalize_run(run_id, latency_ms=latency_ms, abstained=abstained)


__all__ = ["RetrievalService", "RetrievalOutcome"]
