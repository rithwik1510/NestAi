from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, List, Optional

from ...models.db import Chunk, Document, session_scope
from ..index.bm25 import BM25IndexService
from ..index.embed import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class EmailChunkDraft:
    text: str
    subject: str
    sent_at: Optional[str]


class EmailIngestService:
    """Ingest .eml email exports, chunk, embed, and persist to storage."""

    quote_pattern = re.compile(r"^On .*wrote:$", flags=re.IGNORECASE)

    def __init__(
        self,
        source_dir: Path,
        embedding_service: EmbeddingService,
        bm25_service: BM25IndexService,
        *,
        max_tokens: int = 700,
        overlap_ratio: float = 0.15,
    ) -> None:
        self.source_dir = source_dir
        self.embedding_service = embedding_service
        self.bm25_service = bm25_service
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio

    def discover(self) -> List[Path]:
        files = list(self.source_dir.glob("**/*.eml"))
        files.extend(self.source_dir.glob("**/*.mbox"))
        return sorted({path.resolve() for path in files})

    def ingest(self, limit: Optional[int] = None) -> None:
        if not self.source_dir.exists():
            logger.warning("Email directory %s does not exist.", self.source_dir)
            return
        files = self.discover()
        if limit is not None:
            files = files[:limit]
        logger.info("Processing %d email files.", len(files))
        for path in files:
            if path.suffix.lower() != ".eml":
                logger.info("Skipping non-EML file %s; mbox ingestion pending.", path)
                continue
            try:
                self._ingest_eml(path)
            except Exception as exc:  # pragma: no cover
                logger.exception("Failed to ingest email %s: %s", path, exc)

    def _ingest_eml(self, path: Path) -> None:
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        message = BytesParser(policy=policy.default).parsebytes(raw)

        metadata = {
            "from": message.get("from"),
            "to": message.get("to"),
            "cc": message.get("cc"),
            "subject": message.get("subject"),
            "date": message.get("date"),
        }

        body = self._extract_body(message)
        if not body.strip():
            logger.warning("Email %s has no textual body; skipping.", path)
            return

        cleaned_body = self._strip_quotes(body)
        chunks = self._chunk_text(cleaned_body, metadata["subject"], metadata["date"])
        if not chunks:
            logger.warning("No chunks produced for email %s; skipping.", path)
            return

        embeddings = self.embedding_service.embed_texts(chunk.text for chunk in chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count mismatch during email ingestion.")

        with session_scope() as session:
            abs_path = str(path.resolve())
            existing: Document | None = session.query(Document).filter_by(path=abs_path).one_or_none()
            removed_chunk_ids: List[int] = []
            if existing and existing.sha256 == checksum:
                logger.info("Skipping unchanged email %s", path)
                return

            if existing is None:
                document = Document(
                    path=abs_path,
                    title=metadata["subject"] or path.stem.replace("_", " ").title(),
                    type="email",
                    sha256=checksum,
                    size=len(raw),
                    meta={k: v for k, v in metadata.items() if v},
                )
                session.add(document)
                session.flush()
            else:
                document = existing
                removed_chunk_ids = [chunk.id for chunk in list(document.chunks)]
                document.title = metadata["subject"] or document.title
                document.sha256 = checksum
                document.size = len(raw)
                document.meta = {k: v for k, v in metadata.items() if v}
                document.chunks.clear()
                session.flush()

            chunk_payloads: List[dict] = []
            for ordinal, (chunk_draft, embedding) in enumerate(zip(chunks, embeddings), start=1):
                chunk_model = Chunk(
                    document_id=document.id,
                    ordinal=ordinal,
                    text=chunk_draft.text,
                    start_line=None,
                    end_line=None,
                    page_no=None,
                    token_count=self._count_tokens(chunk_draft.text),
                    embedding=embedding,
                )
                session.add(chunk_model)
                session.flush()
                chunk_payloads.append(
                    {
                        "chunk_id": chunk_model.id,
                        "document_id": document.id,
                        "path": document.path,
                        "title": document.title,
                        "content": chunk_draft.text,
                        "metadata": document.meta,
                        "start_line": None,
                        "end_line": None,
                    }
                )

        if removed_chunk_ids:
            self.bm25_service.remove_chunks(removed_chunk_ids)
        self.bm25_service.add_documents(chunk_payloads)
        logger.info("Stored %d chunks for email %s", len(chunks), path)

    def _extract_body(self, message: EmailMessage) -> str:
        if message.is_multipart():
            parts = []
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    parts.append(part.get_content())
            if parts:
                return "\n".join(parts)
            # fallback to first text/* part
            for part in message.walk():
                if part.get_content_type().startswith("text/"):
                    return part.get_content()
            return ""
        if message.get_content_type().startswith("text/"):
            return message.get_content()
        return ""

    def _strip_quotes(self, body: str) -> str:
        lines = body.splitlines()
        cleaned: List[str] = []
        skip_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(">") or self.quote_pattern.match(stripped):
                skip_block = True
            if skip_block and not stripped:
                skip_block = False
                continue
            if skip_block:
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _chunk_text(self, body: str, subject: Optional[str], sent_at: Optional[str]) -> List[EmailChunkDraft]:
        paragraphs = [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]
        chunks: List[EmailChunkDraft] = []
        max_tokens = self.max_tokens
        overlap_tokens = max(1, int(max_tokens * self.overlap_ratio))
        tokenised_paragraphs = [self._count_tokens(paragraph) for paragraph in paragraphs]
        cursor = 0
        while cursor < len(paragraphs):
            token_sum = 0
            end = cursor
            while end < len(paragraphs) and token_sum < max_tokens:
                token_sum += tokenised_paragraphs[end]
                end += 1
            chunk_text = "\n\n".join(paragraphs[cursor:end]).strip()
            if chunk_text:
                chunks.append(EmailChunkDraft(text=chunk_text, subject=subject or "", sent_at=sent_at))
            if end >= len(paragraphs):
                break
            overlap_lines = self._compute_overlap(tokenised_paragraphs[cursor:end], overlap_tokens)
            next_cursor = end - overlap_lines
            cursor = max(cursor + 1, next_cursor)
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


