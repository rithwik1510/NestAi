from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import frontmatter

from ...models.db import Chunk, Document, session_scope
from ..index.bm25 import BM25IndexService
from ..index.embed import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class Section:
    title: str
    start_line: int
    lines: List[str]


@dataclass
class ChunkPayload:
    text: str
    start_line: int
    end_line: int
    token_count: int


class MarkdownIngestService:
    """Ingest Markdown sources, chunk, embed, and persist to the vector store."""

    heading_pattern = re.compile(r"^(#{1,2})\s+(.+?)\s*$")

    def __init__(
        self,
        source_dir: Path,
        embedding_service: EmbeddingService,
        bm25_service: BM25IndexService,
        max_tokens: int = 800,
        overlap_ratio: float = 0.12,
    ) -> None:
        self.source_dir = source_dir
        self.embedding_service = embedding_service
        self.bm25_service = bm25_service
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio

    def discover(self) -> List[Path]:
        """Return a deterministic list of markdown files under the source directory."""

        if not self.source_dir.exists():
            logger.warning("Markdown directory %s does not exist.", self.source_dir)
            return []
        return sorted(self.source_dir.glob("**/*.md"))

    def ingest(self, limit: Optional[int] = None) -> None:
        """Process markdown files and store their chunks with embeddings."""

        files = self.discover()
        if limit:
            files = files[:limit]

        logger.info("Discovered %d markdown files for ingestion", len(files))
        for path in files:
            try:
                self._ingest_file(path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to ingest %s: %s", path, exc)

    def _ingest_file(self, path: Path) -> None:
        path_str = str(path.resolve())
        logger.info("Ingesting %s", path_str)
        document_data = self._load_markdown(path)
        if document_data is None:
            logger.warning("Skipping empty document %s", path_str)
            return

        content_lines = document_data["content"].splitlines()
        sections = self._split_sections(content_lines, default_title=document_data["title"])
        chunks = list(self._generate_chunks(sections))
        if not chunks:
            logger.warning("No chunks generated for %s", path_str)
            return

        embeddings = self.embedding_service.embed_texts(chunk.text for chunk in chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        bm25_payloads: List[dict] = []
        removed_chunk_ids: List[int] = []
        with session_scope() as session:
            existing: Document | None = session.query(Document).filter_by(path=path_str).one_or_none()
            metadata = document_data["metadata"]
            if existing and existing.sha256 == document_data["sha256"]:
                logger.info("Skipping unchanged document %s", path_str)
                return

            if existing is None:
                document = Document(
                    path=path_str,
                    title=document_data["title"],
                    type="md",
                    sha256=document_data["sha256"],
                    size=document_data["size"],
                    meta=metadata,
                    confidentiality_tag=metadata.get("confidentiality", "private"),
                )
                session.add(document)
                session.flush()
            else:
                document = existing
                removed_chunk_ids = [chunk.id for chunk in list(document.chunks)]
                document.title = document_data["title"]
                document.sha256 = document_data["sha256"]
                document.size = document_data["size"]
                document.meta = metadata
                document.confidentiality_tag = metadata.get("confidentiality", document.confidentiality_tag)
                document.chunks.clear()
                session.flush()

            for ordinal, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
                chunk_model = Chunk(
                    document_id=document.id,
                    ordinal=ordinal,
                    text=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    page_no=None,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
                session.add(chunk_model)
                session.flush()
                bm25_payloads.append(
                    {
                        "chunk_id": chunk_model.id,
                        "document_id": document.id,
                        "path": document.path,
                        "title": document.title,
                        "content": chunk.text,
                        "metadata": metadata,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }
                )
            logger.info("Stored %d chunks for %s", len(chunks), path_str)
        if removed_chunk_ids:
            self.bm25_service.remove_chunks(removed_chunk_ids)
        if bm25_payloads:
            self.bm25_service.add_documents(bm25_payloads)

    def _load_markdown(self, path: Path) -> Optional[dict[str, object]]:
        post = frontmatter.load(path)
        content = post.content.strip()
        if not content:
            return None

        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        metadata = dict(post.metadata or {})
        title = self._resolve_title(metadata, content, path)

        return {
            "title": title,
            "content": content,
            "metadata": metadata,
            "sha256": sha256,
            "size": len(raw),
        }

    def _resolve_title(self, metadata: dict[str, object], content: str, path: Path) -> str:
        if isinstance(metadata.get("title"), str) and metadata["title"].strip():
            return metadata["title"].strip()
        for line in content.splitlines():
            if match := self.heading_pattern.match(line):
                return match.group(2).strip()
        return path.stem.replace("_", " ").title()

    def _split_sections(self, lines: List[str], default_title: str) -> List[Section]:
        sections: List[Section] = []
        current_lines: List[str] = []
        current_title = default_title
        current_start = 1

        for index, line in enumerate(lines, start=1):
            match = self.heading_pattern.match(line)
            if match:
                if current_lines:
                    sections.append(Section(title=current_title, start_line=current_start, lines=current_lines))
                current_title = match.group(2).strip()
                current_start = index
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(Section(title=current_title, start_line=current_start, lines=current_lines))

        return sections

    def _generate_chunks(self, sections: List[Section]) -> Iterator[ChunkPayload]:
        max_tokens = self.max_tokens
        overlap_tokens = max(1, int(max_tokens * self.overlap_ratio))

        for section in sections:
            lines = section.lines
            tokens_per_line = [self._count_tokens(line) for line in lines]
            total_lines = len(lines)
            cursor = 0
            while cursor < total_lines:
                token_sum = 0
                end_index = cursor
                while end_index < total_lines and token_sum < max_tokens:
                    token_sum += tokens_per_line[end_index]
                    end_index += 1
                chunk_lines = [line.rstrip() for line in lines[cursor:end_index] if line.strip()]
                if not chunk_lines:
                    cursor = end_index
                    continue
                text = "\n".join(chunk_lines).strip()
                start_line = section.start_line + cursor
                end_line = section.start_line + end_index - 1
                yield ChunkPayload(
                    text=text,
                    start_line=start_line,
                    end_line=end_line,
                    token_count=token_sum,
                )
                if end_index >= total_lines:
                    break
                overlap_lines = self._compute_overlap(tokens_per_line[cursor:end_index], overlap_tokens)
                next_cursor = end_index - overlap_lines
                cursor = max(cursor + 1, next_cursor)

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
