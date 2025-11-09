from __future__ import annotations

from uuid import uuid4

from pka.app.models.schema import (
    ChatAnswer,
    ChatResponse,
    CitationSource,
    ConflictEntry,
    ContextSnippetModel,
)


def test_chat_response_includes_metadata() -> None:
    answer = ChatAnswer(
        abstain=False,
        answer="Answer text.",
        bullets=["Point"],
        conflicts=[ConflictEntry(claim="Claim", sources=[CitationSource(id="doc1", loc="L1-L3")])],
        sources=[CitationSource(id="doc1", loc="L1-L3")],
    )
    context = [
        ContextSnippetModel(
            chunk_id=1,
            document_id=2,
            citation="doc1:L1-L3",
            rationale="BM25=1.0",
            content="Snippet text",
            score_bm25=1.0,
            score_embed=0.9,
        )
    ]
    response = ChatResponse(
        run_id=uuid4(),
        latency_ms=123,
        answer=answer,
        context=context,
        question="Sample question",
        mode="synthesize",
        llm_version="llama3.1:8b",
        prompt_version="1.0.0",
        template_hash="deadbeef",
    )
    assert response.question == "Sample question"
    assert response.llm_version == "llama3.1:8b"
    assert response.context[0].citation == "doc1:L1-L3"
