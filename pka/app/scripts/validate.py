from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

from ..core.settings import settings
from ..services.assistant import AssistantService, AssistantServiceError
from ..services.health import ReadinessService
from ..services.index.embed import EmbeddingService, EmbeddingServiceError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "validation_report.json"


def run_readiness() -> Dict[str, Any]:
    service = ReadinessService()
    try:
        status = service.run_checks()
        payload = status.model_dump()
        payload["passed"] = payload.get("status") == "pass"
        return payload
    finally:
        service.close()


def run_embedding() -> Dict[str, Any]:
    start = perf_counter()
    embedder = EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout=settings.ollama_timeout_seconds,
        expected_dim=settings.vector_dim,
        batch_size=3,
    )
    samples = [
        "NestAi validation ping 1",
        "NestAi validation ping 2",
        "Finance memo summary sample text.",
    ]
    try:
        vectors = embedder.embed_texts(samples)
        latency_ms = int((perf_counter() - start) * 1000)
        return {
            "passed": True,
            "latency_ms": latency_ms,
            "vector_dim": len(vectors[0]) if vectors else 0,
            "count": len(vectors),
        }
    except EmbeddingServiceError as exc:
        return {"passed": False, "error": str(exc)}
    finally:
        embedder.close()


async def run_chat(question: str) -> Dict[str, Any]:
    assistant = AssistantService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout=settings.ollama_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
    )
    start = perf_counter()
    try:
        answer = await assistant.generate(question)
        latency_ms = int((perf_counter() - start) * 1000)
        preview = (answer.answer or "").strip()
        return {
            "passed": True,
            "latency_ms": latency_ms,
            "abstained": answer.abstain,
            "answer_preview": preview[:180],
        }
    except AssistantServiceError as exc:
        return {"passed": False, "error": str(exc)}
    finally:
        await assistant.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-stack validation for NestAi.")
    parser.add_argument(
        "--question",
        default="Give a one-sentence confirmation that you are running locally with strict citations.",
        help="Prompt used for the validation chat turn.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Path to write the JSON report.")
    parser.add_argument("--skip-chat", action="store_true", help="Skip the chat validation step.")
    args = parser.parse_args()

    readiness = run_readiness()
    embedding = run_embedding()
    chat_result: Dict[str, Any] | None = None
    if not args.skip_chat:
        chat_result = asyncio.run(run_chat(args.question))

    passed = readiness.get("passed") and embedding.get("passed") and (
        args.skip_chat or (chat_result and chat_result.get("passed"))
    )

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "summary": {
            "passed": bool(passed),
            "ollama_base_url": settings.ollama_base_url,
            "chat_model": settings.ollama_chat_model,
            "embed_model": settings.ollama_embed_model,
        },
        "readiness": readiness,
        "embedding": embedding,
        "chat": chat_result,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"\nValidation report written to {args.report.resolve()}")


if __name__ == "__main__":
    main()
