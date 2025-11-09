from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import tantivy
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("tantivy is required for BM25 index operations") from exc


logger = logging.getLogger(__name__)


class BM25IndexService:
    """Manage a Tantivy index providing BM25 ranking over chunks."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.schema = self._build_schema()
        self.index = self._open_or_create_index()
        self.search_fields = ["title", "content"]

    def _build_schema(self) -> "tantivy.Schema":
        builder = tantivy.SchemaBuilder()
        # Store IDs as raw text to facilitate exact matching/deletion while keeping them stored-only.
        builder.add_text_field("chunk_id", stored=True, tokenizer_name="raw", index_option="basic")
        builder.add_text_field("document_id", stored=True, tokenizer_name="raw", index_option="basic")
        builder.add_text_field("path", stored=True, tokenizer_name="raw", index_option="basic")
        builder.add_text_field("title", stored=True)
        builder.add_text_field("content", stored=True)
        builder.add_text_field("metadata", stored=True)
        builder.add_integer_field("start_line", stored=True)
        builder.add_integer_field("end_line", stored=True)
        return builder.build()

    def _open_or_create_index(self) -> "tantivy.Index":
        path_str = str(self.index_path)
        if tantivy.Index.exists(path_str):
            return tantivy.Index(self.schema, path=path_str, reuse=True)
        index = tantivy.Index(self.schema, path=path_str, reuse=True)
        return index

    def add_documents(self, documents: Sequence[dict]) -> None:
        if not documents:
            return
        writer = self.index.writer()
        try:
            for payload in documents:
                chunk_id = str(payload["chunk_id"])
                writer.delete_documents_by_term("chunk_id", chunk_id)
                document = tantivy.Document.from_dict(
                    {
                        "chunk_id": [chunk_id],
                        "document_id": [str(payload["document_id"])],
                        "path": [str(payload.get("path", ""))],
                        "title": [payload.get("title", "")],
                        "content": [payload.get("content", "")],
                        "metadata": [json.dumps(payload.get("metadata", {}))],
                        "start_line": [int(payload.get("start_line") or 0)],
                        "end_line": [int(payload.get("end_line") or 0)],
                    }
                )
                writer.add_document(document)
            writer.commit()
        finally:
            writer = None  # release index lock
        self.index.reload()

    def bulk_replace(self, documents: Iterable[dict]) -> None:
        writer = self.index.writer()
        try:
            writer.delete_all_documents()
            for payload in documents:
                document = tantivy.Document.from_dict(
                    {
                        "chunk_id": [str(payload["chunk_id"])],
                        "document_id": [str(payload["document_id"])],
                        "path": [str(payload.get("path", ""))],
                        "title": [payload.get("title", "")],
                        "content": [payload.get("content", "")],
                        "metadata": [json.dumps(payload.get("metadata", {}))],
                        "start_line": [int(payload.get("start_line") or 0)],
                        "end_line": [int(payload.get("end_line") or 0)],
                    }
                )
                writer.add_document(document)
            writer.commit()
        finally:
            writer = None
        self.index.reload()

    def remove_chunks(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        writer = self.index.writer()
        try:
            for chunk_id in chunk_ids:
                writer.delete_documents_by_term("chunk_id", str(chunk_id))
            writer.commit()
        finally:
            writer = None
        self.index.reload()

    def search(self, query: str, limit: int = 50) -> List[dict]:
        query = query.strip()
        if not query:
            return []
        tantivy_query = self.index.parse_query(query, self.search_fields)
        searcher = self.index.searcher()
        result = searcher.search(tantivy_query, limit)
        hits: List[dict] = []
        for score, address in result.hits:
            stored = searcher.doc(address).to_dict()
            hits.append(
                {
                    "chunk_id": stored.get("chunk_id", [None])[0],
                    "document_id": stored.get("document_id", [None])[0],
                    "path": stored.get("path", [""])[0],
                    "title": stored.get("title", [""])[0],
                    "content": stored.get("content", [""])[0],
                    "metadata": json.loads(stored.get("metadata", ["{}"])[0] or "{}"),
                    "start_line": stored.get("start_line", [0])[0],
                    "end_line": stored.get("end_line", [0])[0],
                    "score_bm25": float(score),
                }
            )
        return hits

    def clear(self) -> None:
        writer = self.index.writer()
        try:
            writer.delete_all_documents()
            writer.commit()
        finally:
            writer = None
        self.index.reload()
