from __future__ import annotations

import logging
from typing import List, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...models.db import Chunk, Document

logger = logging.getLogger(__name__)


class VectorIndexService:
    """Perform pgvector similarity searches over chunk embeddings."""

    def __init__(self, distance_metric: str = "cosine") -> None:
        if distance_metric not in {"cosine", "l2"}:
            raise ValueError("distance_metric must be 'cosine' or 'l2'")
        self.distance_metric = distance_metric

    def search(self, session: Session, query_vector: Sequence[float], limit: int = 50) -> List[dict]:
        if not query_vector:
            return []

        comparator = (
            Chunk.embedding.cosine_distance(query_vector)
            if self.distance_metric == "cosine"
            else Chunk.embedding.l2_distance(query_vector)
        )
        stmt = (
            select(
                Chunk.id.label("chunk_id"),
                Chunk.document_id,
                Chunk.text,
                Chunk.start_line,
                Chunk.end_line,
                Chunk.page_no,
                Chunk.token_count,
                Document.title,
                Document.path,
                Document.type,
                comparator.label("distance"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.embedding.is_not(None))
            .order_by(comparator)
            .limit(limit)
        )

        rows = session.execute(stmt).all()
        results: List[dict] = []
        for row in rows:
            distance = float(row.distance) if row.distance is not None else None
            score = None if distance is None else 1.0 - distance if self.distance_metric == "cosine" else -distance
            results.append(
                {
                    "chunk_id": row.chunk_id,
                    "document_id": row.document_id,
                    "path": row.path,
                    "title": row.title,
                    "content": row.text,
                    "start_line": row.start_line,
                    "end_line": row.end_line,
                    "page_no": row.page_no,
                    "token_count": row.token_count,
                    "distance": distance,
                    "score_embed": score,
                }
            )
        return results
