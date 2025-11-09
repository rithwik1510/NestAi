from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import yaml

DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class ExampleExpectations:
    """Structured expectations for a single evaluation example."""

    min_sources: int = 1
    require_abstain: Optional[bool] = None
    required_source_ids: Tuple[str, ...] = ()
    max_latency_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ExampleExpectations":
        if not data:
            return cls()
        return cls(
            min_sources=max(0, int(data.get("min_sources", 1))),
            require_abstain=data.get("require_abstain"),
            required_source_ids=tuple(str(value) for value in data.get("required_sources", []) if value),
            max_latency_ms=int(data["max_latency_ms"]) if data.get("max_latency_ms") is not None else None,
        )


class EvaluationRunner:
    """Drive `/api/chat` requests and score results against deterministic expectations."""

    def __init__(
        self,
        dataset_path: Path,
        *,
        base_url: str = "http://localhost:8000",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._external_client = client is not None
        self._client = client or httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        """Release HTTP client resources."""

        if not self._external_client:
            self._client.close()

    def run(self, report_path: Path | None = None) -> Dict[str, Any]:
        """Execute the evaluation set and return a structured report."""

        dataset = self._load_dataset()
        examples = dataset.get("examples", [])

        results: List[Dict[str, Any]] = []
        latencies: List[int] = []
        completed = 0
        failures = 0

        for index, example in enumerate(examples, start=1):
            question = example.get("question")
            if not question:
                results.append(
                    {
                        "index": index,
                        "status": "error",
                        "issues": ["Missing question field."],
                    }
                )
                failures += 1
                continue

            mode = example.get("mode", "synthesize")
            expectations = ExampleExpectations.from_dict(example.get("expectations"))

            try:
                response = self._client.post("/api/chat", json={"question": question, "mode": mode})
            except Exception as exc:  # pragma: no cover - transport failure
                results.append(
                    {
                        "index": index,
                        "question": question,
                        "mode": mode,
                        "status": "error",
                        "issues": [f"Request failed: {exc}"],
                    }
                )
                failures += 1
                continue

            if response.status_code != 200:
                results.append(
                    {
                        "index": index,
                        "question": question,
                        "mode": mode,
                        "status": "fail",
                        "issues": [f"HTTP {response.status_code}: {response.text}"],
                    }
                )
                failures += 1
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                results.append(
                    {
                        "index": index,
                        "question": question,
                        "mode": mode,
                        "status": "fail",
                        "issues": [f"Invalid JSON response: {exc}"],
                    }
                )
                failures += 1
                continue

            evaluation = self._evaluate_example(payload, expectations)
            evaluation.update({"index": index, "question": question, "mode": mode})
            results.append(evaluation)

            if evaluation["status"] == "pass":
                completed += 1
            else:
                failures += 1

            latency = evaluation.get("latency_ms")
            if isinstance(latency, int):
                latencies.append(latency)

        summary = self._summarise(len(examples), completed, failures, latencies)
        report = {"summary": summary, "results": results}

        if report_path is not None:
            self._write_markdown_report(report_path, report)

        return report

    def _evaluate_example(self, payload: Dict[str, Any], expectations: ExampleExpectations) -> Dict[str, Any]:
        issues: List[str] = []

        answer = payload.get("answer") or {}
        sources = answer.get("sources") or []
        latency_ms = int(payload.get("latency_ms") or 0)

        source_ids = {str(item.get("id")) for item in sources if item and item.get("id")}
        source_count = len(source_ids)

        if expectations.require_abstain is not None:
            observed_abstain = bool(answer.get("abstain"))
            if observed_abstain != expectations.require_abstain:
                expectation = "abstain" if expectations.require_abstain else "provide an answer"
                issues.append(f"Expected model to {expectation}, received abstain={observed_abstain}.")

        if not answer.get("abstain"):
            if source_count < expectations.min_sources:
                issues.append(
                    f"Insufficient citations: expected ≥{expectations.min_sources}, found {source_count}."
                )
            missing_required = [sid for sid in expectations.required_source_ids if sid not in source_ids]
            if missing_required:
                issues.append(f"Missing required citations: {', '.join(missing_required)}.")

        if expectations.max_latency_ms is not None and latency_ms > expectations.max_latency_ms:
            issues.append(
                f"Latency {latency_ms}ms exceeds threshold of {expectations.max_latency_ms}ms."
            )

        status = "pass" if not issues else "fail"

        return {
            "status": status,
            "latency_ms": latency_ms,
            "source_count": source_count,
            "issues": issues,
            "abstain": bool(answer.get("abstain")),
        }

    def _summarise(
        self,
        total: int,
        completed: int,
        failures: int,
        latencies: Sequence[int],
    ) -> Dict[str, Any]:
        pending = total - (completed + failures)
        summary: Dict[str, Any] = {
            "total_examples": total,
            "completed": completed,
            "failed": failures,
            "pending": pending,
        }
        if latencies:
            summary["avg_latency_ms"] = int(statistics.mean(latencies))
            summary["p95_latency_ms"] = self._percentile(latencies, 95)
        return summary

    @staticmethod
    def _percentile(values: Sequence[int], percentile: int) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (percentile / 100) * (len(ordered) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        weight = rank - lower_index
        interpolated = ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight
        return int(round(interpolated))

    def _load_dataset(self) -> Dict[str, Any]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        return yaml.safe_load(self.dataset_path.read_text(encoding="utf-8")) or {}

    def _write_markdown_report(self, report_path: Path, report: Dict[str, Any]) -> None:
        summary = report["summary"]
        lines = [
            "# Evaluation Report",
            "",
            f"- Total examples: {summary.get('total_examples', 0)}",
            f"- Completed: {summary.get('completed', 0)}",
            f"- Failed: {summary.get('failed', 0)}",
            f"- Pending: {summary.get('pending', 0)}",
        ]
        if "avg_latency_ms" in summary:
            lines.append(f"- Average latency: {summary['avg_latency_ms']} ms")
        if "p95_latency_ms" in summary:
            lines.append(f"- P95 latency: {summary['p95_latency_ms']} ms")
        lines.extend(["", "## Result Breakdown"])
        for result in report["results"]:
            status = result.get("status", "unknown").upper()
            question = result.get("question", "Unknown question")
            lines.append(f"- [{status}] {question}")
            for issue in result.get("issues") or []:
                lines.append(f"  - {issue}")
        report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation over the golden dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to evaluation dataset YAML.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("eval_report.md"),
        help="Path to write the markdown report.",
    )
    parser.add_argument("--json", type=Path, help="Optional path to dump JSON summary.")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Root URL of the API server.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    runner = EvaluationRunner(
        args.config,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    try:
        report = runner.run(args.report)
    finally:
        runner.close()
    print(json.dumps(report["summary"], indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
