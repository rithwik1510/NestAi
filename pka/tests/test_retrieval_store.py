import uuid
from datetime import datetime
from unittest.mock import MagicMock

from pka.app.services.retrieval.orchestrator import RetrievalResult
from pka.app.services.retrieval.store import RetrievalStore


def _result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=1,
        document_id=2,
        path="doc.md",
        title="Doc",
        content="Sample chunk",
        start_line=1,
        end_line=5,
        page_no=None,
        token_count=100,
        score_bm25=0.9,
        score_embed=0.8,
        distance=0.2,
        rank_bm25=0,
        rank_embed=1,
        rationale="BM25=0.900, Embed=0.800",
    )


def test_retrieval_store_persists_contexts_and_finalizes() -> None:
    session = MagicMock()
    store = RetrievalStore(session)

    run_id = store.create_run(
        question="What is test?",
        mode="synthesize",
        llm_version="stub-llm",
        prompt_version="1.0.0",
        template_hash="hash",
    )

    assert isinstance(run_id, uuid.UUID)
    assert session.add.call_count == 1
    session.reset_mock()

    store.write_contexts(run_id, [_result()])
    # two adds: QAContext + QAAnswer later
    assert session.add.call_count == 1
    session.reset_mock()

    payload = {"abstain": True}
    store.write_answer(run_id, payload)
    session.merge.assert_called_once()
    session.reset_mock()

    mock_run = MagicMock()
    session.get.return_value = mock_run
    store.finalize_run(run_id, latency_ms=123, abstained=True)
    assert mock_run.latency_ms == 123
    assert mock_run.abstained is True
    session.add.assert_called_once_with(mock_run)


def test_retrieval_store_list_runs_returns_summaries() -> None:
    session = MagicMock()
    store = RetrievalStore(session)

    row = MagicMock()
    row.id = uuid.uuid4()
    row.question = "Question"
    row.mode = "synthesize"
    row.started_at = datetime.utcnow()
    row.latency_ms = 100
    row.abstained = False

    scalar_result = MagicMock()
    scalar_result.all.return_value = [row]
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalar_result
    session.execute.return_value = execute_result

    summaries = store.list_runs(limit=5)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.run_id == row.id
    assert summary.latency_ms == 100
