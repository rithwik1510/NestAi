from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ...models.db import Chunk, Document


@dataclass
class DocumentChunkView:
    id: int
    ordinal: int
    text: str
    start_line: Optional[int]
    end_line: Optional[int]
    page_no: Optional[int]
    token_count: Optional[int]


@dataclass
class DocumentView:
    id: int
    path: str
    title: str
    type: str
    size: int
    sha256: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    confidentiality_tag: Optional[str]
    meta: dict
    chunks: List[DocumentChunkView]


class DocumentService:
    """Fetch document metadata and chunk previews."""

    preview_max_chars = 400

    def fetch_document(self, session: Session, document_id: int) -> DocumentView | None:
        stmt = (
            select(Document)
            .where(Document.id == document_id)
            .options(selectinload(Document.chunks))
        )
        document = session.execute(stmt).scalar_one_or_none()
        if document is None:
            return None

        chunks = sorted(document.chunks, key=lambda chunk: (chunk.ordinal or math.inf, chunk.id))
        chunk_views = [
            DocumentChunkView(
                id=chunk.id,
                ordinal=chunk.ordinal,
                text=self._build_preview(chunk.text),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                page_no=chunk.page_no,
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ]

        return DocumentView(
            id=document.id,
            path=document.path,
            title=document.title,
            type=document.type,
            size=document.size,
            sha256=document.sha256,
            created_at=document.created_at,
            updated_at=document.updated_at,
            confidentiality_tag=document.confidentiality_tag,
            meta=dict(document.meta or {}),
            chunks=chunk_views,
        )

    def _build_preview(self, text: str) -> str:
        if not text:
            return ""
        normalized = " ".join(text.split())
        if len(normalized) <= self.preview_max_chars:
            return normalized
        return normalized[: self.preview_max_chars - 3] + "..."


__all__ = ["DocumentService", "DocumentView", "DocumentChunkView"]
