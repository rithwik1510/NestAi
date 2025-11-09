from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from ..core.settings import settings
from ..services.assistant import AssistantService, AssistantServiceError
from ..services.health import ReadinessService


def _run_readiness() -> dict[str, Any]:
    service = ReadinessService()
    try:
        status = service.run_checks()
        return status.model_dump()
    finally:
        service.close()


async def _run_chat_question(question: str) -> dict[str, Any]:
    assistant = AssistantService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout=settings.ollama_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
    )
    try:
        answer = await assistant.generate(question)
        return answer.model_dump()
    finally:
        await assistant.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostics for the local Ollama-backed personal assistant.")
    parser.add_argument(
        "--question",
        default="Hello! Please confirm you are running locally.",
        help="Prompt used to validate the chat endpoint.",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip the chat round-trip test.",
    )
    args = parser.parse_args()

    print("==> Running readiness checks")
    readiness = _run_readiness()
    print(json.dumps(readiness, indent=2, default=str))

    if not args.skip_chat:
        print("\n==> Verifying chat service")
        try:
            response = asyncio.run(_run_chat_question(args.question))
        except AssistantServiceError as exc:
            print(f"Chat check failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(response, indent=2))

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
