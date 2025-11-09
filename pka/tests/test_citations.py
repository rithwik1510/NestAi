from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from pka.app.services.retrieval.context_builder import ContextBuilder, ContextSnippet
from pka.app.services.retrieval.orchestrator import RetrievalResult
from pka.app.services.synth.llama_local import ChatService, ChatServiceValidationError
from pka.app.services.synth.templates import PromptTemplate, PromptTemplateRegistry


def _retrieval_result(
    *,
    path: str = "/tmp/documents/example.md",
    start_line: int | None = 10,
    end_line: int | None = 20,
    page_no: int | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=42,
        document_id=7,
        path=path,
        title="Example Document",
        content="This is a relevant snippet of text that should be normalised.",
        start_line=start_line,
        end_line=end_line,
        page_no=page_no,
        token_count=120,
    )


def test_context_builder_line_citation() -> None:
    builder = ContextBuilder(max_length=200)
    snippets = builder.build([_retrieval_result()])
    assert len(snippets) == 1
    assert snippets[0].citation.endswith("example.md:L10-L20")


def test_context_builder_page_citation() -> None:
    builder = ContextBuilder(max_length=200)
    result = _retrieval_result(start_line=None, end_line=None, page_no=3)
    snippets = builder.build([result])
    assert snippets[0].citation.endswith("example.md:p.3")


def _write_schema(path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "abstain": {"type": "boolean"},
            "answer": {"type": "string"},
            "bullets": {"type": "array"},
            "conflicts": {"type": "array"},
            "sources": {"type": "array"},
        },
        "required": ["abstain", "answer", "bullets", "conflicts", "sources"],
    }
    path.write_text(json.dumps(schema), encoding="utf-8")


@pytest.mark.asyncio
async def test_chat_service_retries_on_invalid_json(tmp_path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_schema(schema_path)

    registry = PromptTemplateRegistry()
    registry.register(PromptTemplate(name="test", version="1", content="{question}\n{context}\n{schema_json}"))

    service = ChatService(
        base_url="http://localhost:11434",
        model="stub",
        temperature=0.0,
        seed=123,
        timeout=5,
        template_registry=registry,
        template_name="test",
        schema_path=schema_path,
        max_retries=1,
    )

    call_counter = {"count": 0}

    class _FakeResponse:
        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:  # pragma: no cover - behaviour is trivial
            return None

        def json(self) -> Dict[str, Any]:
            return self._payload

    async def _fake_post(url: str, json: Dict[str, Any]) -> _FakeResponse:  # type: ignore[override]
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            payload = {"message": {"content": "{not valid json"}}
        else:
            payload = {
                "message": {
                    "content": json.dumps(
                        {
                            "abstain": False,
                            "answer": "Valid answer",
                            "bullets": [],
                            "conflicts": [],
                            "sources": [{"id": "doc1", "loc": "L1-L5"}],
                        }
                    )
                }
            }
        return _FakeResponse(payload)

    service._client.post = _fake_post  # type: ignore[assignment]

    snippet = ContextSnippet(
        document_id=1,
        chunk_id=2,
        content="Content",
        citation="doc.md:L1-L2",
        rationale="BM25=1.0",
    )
    answer = await service.generate(question="What is up?", snippets=[snippet], mode="synthesize")
    await service.close()

    assert call_counter["count"] == 2  # one retry triggered
    assert answer.answer == "Valid answer"
    assert answer.sources[0].loc == "L1-L5"


@pytest.mark.asyncio
async def test_chat_service_raises_after_max_retries(tmp_path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_schema(schema_path)

    registry = PromptTemplateRegistry()
    registry.register(PromptTemplate(name="test", version="1", content="{question}\n{context}\n{schema_json}"))

    service = ChatService(
        base_url="http://localhost:11434",
        model="stub",
        temperature=0.0,
        seed=123,
        timeout=5,
        template_registry=registry,
        template_name="test",
        schema_path=schema_path,
        max_retries=1,
    )

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"message": {"content": "{still invalid"}}

    async def _fake_post(url: str, json: Dict[str, Any]) -> _FakeResponse:  # type: ignore[override]
        return _FakeResponse()

    service._client.post = _fake_post  # type: ignore[assignment]

    snippet = ContextSnippet(
        document_id=1,
        chunk_id=2,
        content="Content",
        citation="doc.md:L1-L2",
        rationale="BM25=1.0",
    )

    with pytest.raises(ChatServiceValidationError):
        await service.generate(question="Test failure", snippets=[snippet], mode="synthesize")

    await service.close()
