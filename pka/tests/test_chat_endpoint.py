import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from pka.app.main import create_app
from pka.app.models.schema import ChatAnswer, HealthStatus
from pka.app.routers.chat import get_chat_service, get_retrieval_service, get_retrieval_store
from pka.app.services.retrieval.context_builder import ContextSnippet
from pka.app.services.retrieval.orchestrator import RetrievalResult
from pka.app.services.retrieval.service import RetrievalOutcome
from pka.app.services.retrieval.store import ReplayRecord, RunSummary
from pka.app.services.synth.templates import PromptTemplate


@dataclass
class StubRetrievalService:
    outcome: RetrievalOutcome
    finalized: Optional[Dict[str, Any]] = None

    def retrieve(self, *args: Any, **kwargs: Any) -> RetrievalOutcome:
        return self.outcome

    def finalize(
        self,
        *_: Any,
        run_id: uuid.UUID,
        answer_json: Dict[str, Any],
        latency_ms: int,
        abstained: bool,
    ) -> None:
        self.finalized = {
            "run_id": run_id,
            "answer_json": answer_json,
            "latency_ms": latency_ms,
            "abstained": abstained,
        }


class StubChatService:
    def __init__(self) -> None:
        self.template = PromptTemplate(name="stub", content="{question}", version="stub-v1")

    async def generate(self, *, question: str, snippets: List[ContextSnippet], mode: str) -> ChatAnswer:
        return ChatAnswer(
            abstain=False,
            answer=f"Q: {question}",
            bullets=[f"Mode: {mode}", f"Snippets: {len(snippets)}"],
            conflicts=[],
            sources=[{"id": f"{snippets[0].document_id}:{snippets[0].chunk_id}", "loc": snippets[0].citation}],
        )

    async def close(self) -> None:  # match real service signature
        return None


class StubRetrievalStore:
    def __init__(self) -> None:
        self.record: Optional[ReplayRecord] = None
        self.summaries: List[RunSummary] = []

    def replay(self, run_id: uuid.UUID) -> Optional[ReplayRecord]:
        if self.record and self.record.run_id == run_id:
            return self.record
        return None

    def list_runs(self, limit: int = 20) -> List[RunSummary]:
        return self.summaries[:limit]


class DummyEmbeddingService:
    def close(self) -> None:
        return None


def _context_snippet() -> ContextSnippet:
    return ContextSnippet(
        document_id=1,
        chunk_id=10,
        content="Snippet text",
        citation="doc.md:L1-L4",
        rationale="BM25=1.0",
        score_bm25=1.0,
        score_embed=0.9,
    )


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=10,
        document_id=1,
        path="doc.md",
        title="Doc",
        content="Snippet text",
        start_line=1,
        end_line=4,
        page_no=None,
        token_count=120,
        score_bm25=1.0,
        score_embed=0.9,
        distance=0.1,
        rank_bm25=0,
        rank_embed=0,
        rationale="BM25=1.0",
    )


def _outcome() -> RetrievalOutcome:
    return RetrievalOutcome(run_id=uuid.uuid4(), latency_ms=50, contexts=[_retrieval_result()], snippets=[_context_snippet()])


@pytest.fixture()
def client() -> TestClient:
    app = create_app()

    app.state.readiness_service.run_checks = lambda: HealthStatus(status="pass", probes=[])
    app.state.embedding_service = DummyEmbeddingService()

    retrieval_stub = StubRetrievalService(_outcome())
    chat_stub = StubChatService()
    store_stub = StubRetrievalStore()

    def override_retrieval(request: Request) -> StubRetrievalService:  # type: ignore[override]
        return retrieval_stub

    def override_chat(request: Request) -> StubChatService:  # type: ignore[override]
        return chat_stub

    def override_store(request: Request) -> StubRetrievalStore:  # type: ignore[override]
        return store_stub

    app.dependency_overrides[get_retrieval_service] = override_retrieval
    app.dependency_overrides[get_chat_service] = override_chat
    app.dependency_overrides[get_retrieval_store] = override_store

    app.state.retrieval_service = retrieval_stub
    app.state.chat_service = chat_stub

    with TestClient(app) as test_client:
        test_client.app.state._test_retrieval_stub = retrieval_stub
        test_client.app.state._test_replay_store = store_stub
        yield test_client


def test_chat_endpoint_returns_llm_answer(client: TestClient) -> None:
    payload = {"question": "What is the snippet about?", "mode": "synthesize"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert uuid.UUID(data["run_id"])
    assert data["answer"]["abstain"] is False
    assert data["answer"]["answer"].startswith("Q: What is the snippet about?")
    assert data["context"][0]["citation"] == "doc.md:L1-L4"

    retrieval_stub = client.app.state._test_retrieval_stub
    assert retrieval_stub.finalized is not None
    assert retrieval_stub.finalized["latency_ms"] >= 50


def test_replay_endpoint_returns_record(client: TestClient) -> None:
    retrieval_stub = client.app.state._test_retrieval_stub
    store_stub = client.app.state._test_replay_store

    answer = ChatAnswer(
        abstain=False,
        answer="Replay answer",
        bullets=["Replay"],
        conflicts=[],
        sources=[{"id": "1:10", "loc": "doc.md:L1-L4"}],
    )
    record = ReplayRecord(
        run_id=retrieval_stub.outcome.run_id,
        question="What is the snippet about?",
        mode="synthesize",
        llm_version="stub-llm",
        prompt_version="stub-v1",
        template_hash="hash",
        latency_ms=123,
        abstained=False,
        answer=answer,
        snippets=[_context_snippet()],
    )
    store_stub.record = record

    response = client.get(f"/api/replay/{record.run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == str(record.run_id)
    assert data["answer"]["answer"] == "Replay answer"


def test_replay_endpoint_missing_returns_404(client: TestClient) -> None:
    store_stub = client.app.state._test_replay_store
    store_stub.record = None
    response = client.get(f"/api/replay/{uuid.uuid4()}")
    assert response.status_code == 404


def test_replay_list_endpoint_returns_summaries(client: TestClient) -> None:
    store_stub = client.app.state._test_replay_store
    summary = RunSummary(
        run_id=uuid.uuid4(),
        question="Sample question",
        mode="synthesize",
        started_at=datetime.utcnow(),
        latency_ms=250,
        abstained=False,
    )
    store_stub.summaries = [summary]

    response = client.get("/api/replay?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["run_id"] == str(summary.run_id)
    assert payload[0]["latency_ms"] == 250
