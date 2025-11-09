from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer

from ...models.db import Chunk, Document, session_scope
from ..index.bm25 import BM25IndexService
from ..index.embed import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class PDFChunkDraft:
    text: str
    page_no: int
    token_count: int


class PDFIngestService:
    """Ingest text-layer PDFs using pdfminer for extraction."""

    def __init__(
        self,
        source_dir: Path,
        embedding_service: EmbeddingService,
        bm25_service: BM25IndexService,
        *,
        max_tokens: int = 800,
        overlap_tokens: int = 120,
    ) -> None:
        self.source_dir = source_dir
        self.embedding_service = embedding_service
        self.bm25_service = bm25_service
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._laparams = LAParams()

    def discover(self) -> List[Path]:
        if not self.source_dir.exists():
            logger.warning("PDF directory %s does not exist.", self.source_dir)
            return []
        return sorted(self.source_dir.glob("**/*.pdf"))

    def ingest(self, limit: Optional[int] = None) -> None:
        files = self.discover()
        if limit is not None:
            files = files[:limit]
        logger.info("Processing %d PDF files.", len(files))
        for path in files:
            try:
                self._ingest_file(path)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to ingest PDF %s: %s", path, exc)

    def _ingest_file(self, pdf_path: Path) -> None:
        raw_bytes = pdf_path.read_bytes()
        checksum = hashlib.sha256(raw_bytes).hexdigest()

        page_texts = list(self._extract_pages(pdf_path))
        if not page_texts:
            logger.warning("PDF %s produced no text; ensure it has a text layer.", pdf_path)
            return

        chunk_drafts: List[PDFChunkDraft] = []
        for index, text in enumerate(page_texts, start=1):
            chunk_drafts.extend(self._chunk_page(index, text))

        if not chunk_drafts:
            logger.warning("PDF %s produced no chunks after tokenization; skipping.", pdf_path)
            return

        embeddings = self.embedding_service.embed_texts(chunk.text for chunk in chunk_drafts)
        if len(embeddings) != len(chunk_drafts):
            raise RuntimeError("Embedding count mismatch during PDF ingestion.")

        document_meta = {"pages": len(page_texts), "ingestion": "pdfminer"}

        removed_chunk_ids: List[int] = []
        bm25_payloads: List[dict] = []
        abs_path = str(pdf_path.resolve())
        with session_scope() as session:
            existing: Document | None = session.query(Document).filter_by(path=abs_path).one_or_none()
            if existing and existing.sha256 == checksum:
                logger.info("Skipping unchanged PDF %s", pdf_path)
                return

            if existing is None:
                document = Document(
                    path=abs_path,
                    title=pdf_path.stem.replace("_", " ").title(),
                    type="pdf",
                    sha256=checksum,
                    size=len(raw_bytes),
                    meta=document_meta,
                )
                session.add(document)
                session.flush()
            else:
                document = existing
                removed_chunk_ids = [chunk.id for chunk in list(document.chunks)]
                document.title = pdf_path.stem.replace("_", " ").title()
                document.sha256 = checksum
                document.size = len(raw_bytes)
                document.meta = document_meta
                document.chunks.clear()
                session.flush()

            for ordinal, (draft, embedding) in enumerate(zip(chunk_drafts, embeddings), start=1):
                chunk = Chunk(
                    document_id=document.id,
                    ordinal=ordinal,
                    text=draft.text,
                    start_line=None,
                    end_line=None,
                    page_no=draft.page_no,
                    token_count=draft.token_count,
                    embedding=embedding,
                )
                session.add(chunk)
                session.flush()
                bm25_payloads.append(
                    {
                        "chunk_id": chunk.id,
                        "document_id": document.id,
                        "path": document.path,
                        "title": document.title,
                        "content": draft.text,
                        "metadata": {"page_no": draft.page_no},
                        "start_line": None,
                        "end_line": None,
                    }
                )

        if removed_chunk_ids:
            self.bm25_service.remove_chunks(removed_chunk_ids)
        if bm25_payloads:
            self.bm25_service.add_documents(bm25_payloads)
        logger.info("Stored %d chunks for PDF %s", len(chunk_drafts), pdf_path)

    def _extract_pages(self, pdf_path: Path) -> Iterable[str]:
        for page_layout in extract_pages(pdf_path, laparams=self._laparams):
            parts: List[str] = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    parts.append(element.get_text())
            yield "\n".join(parts).strip()

    def _chunk_page(self, page_no: int, text: str) -> List[PDFChunkDraft]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        token_counts = [self._count_tokens(paragraph) for paragraph in paragraphs]
        cursor = 0
        chunks: List[PDFChunkDraft] = []
        total = len(paragraphs)
        while cursor < total:
            token_sum = 0
            end = cursor
            while end < total and (token_sum + token_counts[end] <= self.max_tokens or end == cursor):
                token_sum += token_counts[end]
                end += 1
            chunk_text = "\n\n".join(paragraphs[cursor:end]).strip()
            if chunk_text:
                chunks.append(
                    PDFChunkDraft(
                        text=chunk_text,
                        page_no=page_no,
                        token_count=max(token_sum, self._count_tokens(chunk_text)),
                    )
                )
            if end >= total:
                break
            overlap = self._compute_overlap(token_counts[cursor:end], self.overlap_tokens)
            cursor = max(cursor + 1, end - overlap)
        return chunks

    @staticmethod
    def _count_tokens(text: str) -> int:
        stripped = text.strip()
        return len(stripped.split()) if stripped else 0

    @staticmethod
    def _compute_overlap(tokens_segment: List[int], overlap_tokens: int) -> int:
        accumulated = 0
        lines = 0
        for count in reversed(tokens_segment):
            accumulated += count
            lines += 1
            if accumulated >= overlap_tokens:
                break
        return lines
