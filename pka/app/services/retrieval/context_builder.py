from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .orchestrator import RetrievalResult


@dataclass
class ContextSnippet:
    document_id: int
    chunk_id: int
    content: str
    citation: str
    rationale: str
    score_bm25: Optional[float] = None
    score_embed: Optional[float] = None


class ContextBuilder:
    """Compose retrieval results into concise context snippets for synthesis."""

    def __init__(self, max_length: int = 900) -> None:
        self.max_length = max_length

    def build(self, results: Iterable[RetrievalResult]) -> List[ContextSnippet]:
        snippets: List[ContextSnippet] = []
        for result in results:
            normalized = self._normalize_text(result.content)
            if not normalized:
                continue
            clipped = self._clip(normalized, self.max_length)
            citation = self._format_citation(result)
            rationale = self._compose_rationale(result)
            snippets.append(
                ContextSnippet(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    content=clipped,
                    citation=citation,
                    rationale=rationale,
                    score_bm25=result.score_bm25,
                    score_embed=result.score_embed,
                )
            )
        return snippets

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _clip(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return textwrap.shorten(text, width=max_length, placeholder=" ...")

    @staticmethod
    def _format_citation(result: RetrievalResult) -> str:
        path = Path(result.path)
        fragment = ""
        if result.start_line and result.end_line:
            fragment = f"L{result.start_line}-L{result.end_line}"
        elif result.page_no:
            fragment = f"p.{result.page_no}"
        return f"{path.name}:{fragment}" if fragment else path.name

    @staticmethod
    def _compose_rationale(result: RetrievalResult) -> str:
        parts: List[str] = []
        if result.score_bm25 is not None:
            parts.append(f"BM25={result.score_bm25:.3f}")
        if result.score_embed is not None:
            parts.append(f"Embed={result.score_embed:.3f}")
        if result.distance is not None and result.score_embed is None:
            parts.append(f"Dist={result.distance:.3f}")
        return ", ".join(parts) if parts else "Relevant snippet"
