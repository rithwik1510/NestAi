from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from pka.app.core.settings import settings  # type: ignore
    from pka.app.services.retrieval.context_builder import ContextSnippet  # type: ignore
    from pka.app.services.synth import (  # type: ignore
        ChatService,
        ChatServiceError,
        ChatServiceValidationError,
        PromptTemplate,
        PromptTemplateRegistry,
    )
else:
    from ..core.settings import settings
    from ..services.retrieval.context_builder import ContextSnippet
    from ..services.synth import (
        ChatService,
        ChatServiceError,
        ChatServiceValidationError,
        PromptTemplate,
        PromptTemplateRegistry,
    )


def _build_template_registry() -> PromptTemplateRegistry:
    registry = PromptTemplateRegistry()
    registry.register(
        PromptTemplate(
            name="cite_or_abstain_v1",
            version="1.0.0",
            content=textwrap.dedent(
                """\
                You must answer the user's question using ONLY these context snippets:

                {context}

                Question: {question}

                Return a JSON object matching this schema exactly:

                {schema_json}

                Rules:
                - Cite every claim with the provided citation identifiers.
                - If the context is insufficient, set "abstain": true and give actionable guidance.
                - Do not invent sources or information beyond the snippets."""
            ).strip(),
        )
    )
    return registry


def _build_chat_service(max_retries: int) -> ChatService:
    schema_path = Path(__file__).resolve().parents[1] / "services" / "synth" / "response_schema.json"
    registry = _build_template_registry()
    return ChatService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout=settings.ollama_timeout_seconds,
        template_registry=registry,
        template_name="cite_or_abstain_v1",
        schema_path=schema_path,
        max_retries=max(0, max_retries),
    )


def _default_snippets() -> List[ContextSnippet]:
    return [
        ContextSnippet(
            document_id=1,
            chunk_id=1,
            content="Deep work sessions were scheduled every morning to focus on high-impact tasks without interruptions.",
            citation="deep_work_notes.md:L12-L24",
            rationale="Manual test snippet",
        )
    ]


def _parse_contexts(values: Iterable[str] | None) -> List[ContextSnippet]:
    if not values:
        return _default_snippets()

    snippets: List[ContextSnippet] = []
    for idx, raw in enumerate(values, start=1):
        parts = raw.split("|", 1)
        if len(parts) != 2:
            raise ValueError(f"Context '{raw}' must use the format 'citation|text'.")
        citation, text = parts[0].strip(), parts[1].strip()
        if not citation or not text:
            raise ValueError(f"Context '{raw}' must include both citation and text.")
        snippets.append(
            ContextSnippet(
                document_id=idx,
                chunk_id=idx,
                content=text,
                citation=citation,
                rationale="Manual context",
            )
        )
    return snippets


async def _run(args: argparse.Namespace) -> int:
    chat = _build_chat_service(args.max_retries)
    snippets = _parse_contexts(args.contexts)

    try:
        answer = await chat.generate(question=args.question, snippets=snippets, mode=args.mode)
    except ChatServiceValidationError as exc:
        print(f"[validation-error] {exc}", file=sys.stderr)
        if chat.last_raw_response:
            print("--- Raw response ---", file=sys.stderr)
            print(chat.last_raw_response, file=sys.stderr)
        return 2
    except ChatServiceError as exc:
        print(f"[chat-error] {exc}", file=sys.stderr)
        if chat.last_raw_response:
            print("--- Raw response ---", file=sys.stderr)
            print(chat.last_raw_response, file=sys.stderr)
        return 3
    else:
        print(answer.model_dump_json(indent=2))
        if args.show_raw and chat.last_raw_response:
            print("\n--- Raw response ---")
            print(chat.last_raw_response)
        return 0
    finally:
        await chat.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a single question through ChatService to inspect the JSON contract emitted by the configured Ollama model.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What was the focus of my recent deep work sessions?",
        help="Question to send (defaults to a simple sanity-check prompt).",
    )
    parser.add_argument(
        "--context",
        "-c",
        dest="contexts",
        action="append",
        metavar="CITATION|TEXT",
        help="Manually supply a context snippet in the form 'citation|text'. May be passed multiple times.",
    )
    parser.add_argument(
        "--mode",
        choices=["synthesize", "lookup", "timeline", "flashcards"],
        default="synthesize",
        help="Chat mode to report to the service (defaults to synthesize).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="How many invalid-response retries ChatService should allow (defaults to 1).",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the raw LLM response in addition to the parsed JSON output.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except ValueError as exc:
        print(f"[input-error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
