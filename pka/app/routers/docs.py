from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from ..models.schema import DocumentChunkModel, DocumentResponse

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..services.docs import DocumentService

router = APIRouter(prefix="/api/docs", tags=["documents"])


def get_document_service(request: Request) -> "DocumentService":
    service = request.app.state.document_service  # type: ignore[attr-defined]
    return service


def get_db_session():
    from ..models.db import get_session as real_get_session

    yield from real_get_session()


@router.get("/{document_id}", response_model=DocumentResponse, summary="Document metadata and chunk previews")
def fetch_document(
    document_id: int,
    session: Session = Depends(get_db_session),
    document_service: "DocumentService" = Depends(get_document_service),
) -> DocumentResponse:
    record = document_service.fetch_document(session, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = [
        DocumentChunkModel(
            id=chunk.id,
            ordinal=chunk.ordinal,
            preview=chunk.text,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            page_no=chunk.page_no,
            token_count=chunk.token_count,
        )
        for chunk in record.chunks
    ]

    return DocumentResponse(
        id=record.id,
        path=record.path,
        title=record.title,
        type=record.type,
        size=record.size,
        sha256=record.sha256,
        created_at=record.created_at,
        updated_at=record.updated_at,
        confidentiality_tag=record.confidentiality_tag,
        meta=record.meta,
        chunk_count=len(chunks),
        chunks=chunks,
    )


__all__ = ["router", "get_document_service", "get_db_session"]
