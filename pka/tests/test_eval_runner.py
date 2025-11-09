from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from pka.app.services.evals.scorer import EvaluationRunner


class DummyResponse:
    def __init__(self, data: Any, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def json(self) -> Any:
        return self._data

    @property
    def text(self) -> str:
        return json.dumps(self._data)


class DummyClient:
    def __init__(self, responses: List[DummyResponse]) -> None:
        self._responses = list(responses)
        self.requests: List[Any] = []
        self.closed = False

    def post(self, url: str, json: Any) -> DummyResponse:
        self.requests.append((url, json))
        if not self._responses:
            raise RuntimeError("No more responses configured")
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _dataset(tmp_path: Path, yaml_text: str) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


def test_evaluation_runner_no_examples(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        "metadata:\n  name: empty\nexamples: []\n",
    )
    client = DummyClient([])
    runner = EvaluationRunner(dataset, client=client)
    try:
        report = runner.run()
    finally:
        runner.close()
    assert report["summary"]["total_examples"] == 0
    assert client.closed is False  # external client is not closed by runner


def test_evaluation_runner_success(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        """
metadata:
  name: sample
examples:
  - question: "What is stored?"
    mode: synthesize
    expectations:
      min_sources: 1
""",
    )
    response_payload = {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "latency_ms": 120,
        "question": "What is stored?",
        "mode": "synthesize",
        "llm_version": "llama3.1:8b",
        "prompt_version": "1.0.0",
        "template_hash": "abc123",
        "answer": {
            "abstain": False,
            "answer": "Stored content.",
            "bullets": [],
            "conflicts": [],
            "sources": [{"id": "doc1", "loc": "L1-L10"}],
        },
        "context": [
            {
                "chunk_id": 1,
                "document_id": 1,
                "citation": "doc1:L1-L10",
                "rationale": "BM25=1.0",
                "content": "text",
                "score_bm25": 1.0,
                "score_embed": 0.5,
            }
        ],
    }
    client = DummyClient([DummyResponse(response_payload)])
    runner = EvaluationRunner(dataset, client=client)
    try:
        report = runner.run()
    finally:
        runner.close()
    summary = report["summary"]
    assert summary["total_examples"] == 1
    assert summary["completed"] == 1
    result = report["results"][0]
    assert result["status"] == "pass"
    assert result["source_count"] == 1
    assert result["latency_ms"] == 120


def test_evaluation_runner_flags_citation_issue(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        """
metadata: {}
examples:
  - question: "Test?"
    expectations:
      min_sources: 2
""",
    )
    response_payload = {
        "run_id": "00000000-0000-0000-0000-000000000002",
        "latency_ms": 50,
        "question": "Test?",
        "mode": "synthesize",
        "llm_version": "llama3.1:8b",
        "prompt_version": "1.0.0",
        "template_hash": "abc123",
        "answer": {
            "abstain": False,
            "answer": "Answer",
            "bullets": [],
            "conflicts": [],
            "sources": [],
        },
        "context": [],
    }
    client = DummyClient([DummyResponse(response_payload)])
    runner = EvaluationRunner(dataset, client=client)
    try:
        report = runner.run()
    finally:
        runner.close()
    result = report["results"][0]
    assert result["status"] == "fail"
    assert any("citations" in issue.lower() for issue in result.get("issues", []))
