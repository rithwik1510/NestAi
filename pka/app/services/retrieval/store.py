from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...models.db import Chunk, Document, QAAnswer, QAContext, QARun
from ...models.schema import ChatAnswer
from ..retrieval.context_builder import ContextSnippet
from ..retrieval.orchestrator import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ReplayRecord:
    run_id: uuid.UUID
    question: str
    mode: str
    llm_version: str
    prompt_version: str
    template_hash: str
    latency_ms: int | None
    abstained: bool
    answer: ChatAnswer
    snippets: List[ContextSnippet]


@dataclass
class RunSummary:
    run_id: uuid.UUID
    question: str
    mode: str
    started_at: datetime
    latency_ms: int | None
    abstained: bool


class RetrievalStore:
    """Persist retrieval runs, contexts, and answers for replay."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        question: str,
        mode: str,
        llm_version: str,
        prompt_version: str,
        template_hash: str,
    ) -> uuid.UUID:
        run = QARun(
            question=question,
            mode=mode,
            llm_version=llm_version,
            prompt_version=prompt_version,
            template_hash=template_hash,
            latency_ms=None,
            abstained=False,
        )
        self.session.add(run)
        self.session.flush()
        return run.id

    def write_contexts(self, run_id: uuid.UUID, contexts: Sequence[RetrievalResult]) -> None:
        for rank, context in enumerate(contexts, start=1):
            entry = QAContext(
                run_id=run_id,
                chunk_id=context.chunk_id,
                rank=rank,
                score_bm25=context.score_bm25,
                score_embed=context.score_embed,
                score_rerank=None,
                rationale=context.rationale or "",
            )
            self.session.add(entry)

    def write_answer(self, run_id: uuid.UUID, answer_json: dict) -> None:
        record = QAAnswer(run_id=run_id, answer_json=answer_json)
        self.session.merge(record)

    def finalize_run(self, run_id: uuid.UUID, *, latency_ms: int, abstained: bool) -> None:
        run = self.session.get(QARun, run_id)
        if not run:
            logger.error("Attempted to finalize missing run %s", run_id)
            return
        run.latency_ms = latency_ms
        run.abstained = abstained
        self.session.add(run)

    def replay(self, run_id: uuid.UUID) -> ReplayRecord | None:
        run = self.session.get(QARun, run_id)
        if not run:
            return None
        answer_row = self.session.get(QAAnswer, run_id)
        if not answer_row:
            return None
        answer = ChatAnswer.model_validate(answer_row.answer_json)

        stmt = (
            select(QAContext, Chunk, Document)
            .outerjoin(Chunk, QAContext.chunk_id == Chunk.id)
            .outerjoin(Document, Chunk.document_id == Document.id)
            .where(QAContext.run_id == run_id)
            .order_by(QAContext.rank)
        )
        snippets: List[ContextSnippet] = []
        for ctx, chunk, document in self.session.execute(stmt).all():
            if chunk is None:
                continue
            citation = self._build_citation(chunk, document)
            snippets.append(
                ContextSnippet(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    content=chunk.text,
                    citation=citation,
                    rationale=ctx.rationale or "",
                    score_bm25=ctx.score_bm25,
                    score_embed=ctx.score_embed,
                )
            )

        return ReplayRecord(
            run_id=run.id,
            question=run.question,
            mode=run.mode,
            llm_version=run.llm_version,
            prompt_version=run.prompt_version,
            template_hash=run.template_hash,
            latency_ms=run.latency_ms,
            abstained=run.abstained,
            answer=answer,
            snippets=snippets,
        )

    @staticmethod
    def _build_citation(chunk: Chunk, document: Document | None) -> str:
        path = Path(document.path).name if document else "unknown"
        if chunk.start_line and chunk.end_line:
            return f"{path}:L{chunk.start_line}-L{chunk.end_line}"
        if chunk.page_no:
            return f"{path}:p.{chunk.page_no}"
        return path

    def list_runs(self, limit: int = 20) -> List[RunSummary]:
        stmt = (
            select(QARun)
            .order_by(QARun.started_at.desc())
            .limit(limit)
        )
        rows = self.session.execute(stmt).scalars().all()
        return [
            RunSummary(
                run_id=row.id,
                question=row.question,
                mode=row.mode,
                started_at=row.started_at,
                latency_ms=row.latency_ms,
                abstained=row.abstained,
            )
            for row in rows
        ]


__all__ = ["RetrievalStore", "ReplayRecord", "RunSummary"]
