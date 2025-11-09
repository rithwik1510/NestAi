from __future__ import annotations

import json
import logging
import textwrap
from copy import deepcopy
from json import JSONDecodeError
from pathlib import Path
from typing import List, Sequence

import httpx
from jsonschema import Draft7Validator, ValidationError

from ...models.schema import ChatAnswer
from ..retrieval.context_builder import ContextSnippet
from .templates import PromptTemplate
from .templates import PromptTemplateRegistry

logger = logging.getLogger(__name__)


class ChatServiceError(RuntimeError):
    """Base exception raised when synthesis fails."""


class ChatServiceValidationError(ChatServiceError):
    """Raised when the model output fails schema validation."""


class ChatService:
    """Deterministic Ollama-driven chat synthesis enforcing cite-or-abstain contract."""

    @property
    def template(self) -> PromptTemplate:
        return self.template_registry.get(self.template_name)

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        seed: int,
        *,
        timeout: int,
        template_registry: PromptTemplateRegistry,
        template_name: str,
        schema_path: Path,
        max_retries: int = 1,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.template_registry = template_registry
        self.template_name = template_name
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._system_prompt = textwrap.dedent(
            """\
            You are the Personal Knowledge Analyst. Use ONLY the provided context snippets.
            - If the snippets do not fully answer the question, you MUST abstain with actionable guidance.
            - Every claim must cite sources; provide citations using the supplied identifiers.
            - Respond with JSON only. No prose, no markdown, no commentary."""
        ).strip()
        schema_text = Path(schema_path).read_text(encoding="utf-8-sig")
        self._schema_prompt = schema_text.replace("{", "{{").replace("}", "}}")
        self._schema = json.loads(schema_text)
        self._validator = Draft7Validator(self._schema)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._last_raw_response: str | None = None

    @property
    def last_raw_response(self) -> str | None:
        """Return the raw JSON string returned by the LLM for the latest call."""

        return self._last_raw_response

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        *,
        question: str,
        snippets: Sequence[ContextSnippet],
        mode: str,
    ) -> ChatAnswer:
        template = self.template
        context_block = self._format_context(snippets)
        user_prompt = template.render(
            question=self._escape_braces(question.strip()),
            context=context_block,
            schema_json=self._schema_prompt,
            mode=mode,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response_json = await self._invoke_with_retries(messages=messages)
        except ChatServiceValidationError as exc:
            logger.error("Chat validation failed after retries: %s", exc)
            raise
        except ChatServiceError as exc:
            logger.error("Chat service error: %s", exc)
            raise

        return ChatAnswer.model_validate(response_json)

    async def _invoke_with_retries(self, messages: List[dict]) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._invoke(messages)
            except ChatServiceValidationError as exc:
                last_error = exc
                logger.debug("Validation failure (attempt %d): %s", attempt + 1, exc)
                correction = textwrap.dedent(
                    f"""\
                    The previous response was invalid: {exc}
                    Respond again with strictly valid JSON that satisfies the schema."""
                )
                messages.append({"role": "user", "content": correction})
            except ChatServiceError as exc:
                last_error = exc
                logger.debug("Chat service failure (attempt %d): %s", attempt + 1, exc)
                break
        if last_error is None:
            raise ChatServiceError("Chat generation failed for an unknown reason.")
        raise last_error

    async def _invoke(self, messages: List[dict]) -> dict:
        options = {
            "model": self.model,
        }
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        url = "/api/chat"
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatServiceError(
                "Ollama chat request timed out before completing."
            ) from exc
        except httpx.HTTPError as exc:
            raise ChatServiceError(f"Ollama chat request failed: {exc!s}") from exc

        data = response.json()
        try:
            content = data["message"]["content"]
        except KeyError as exc:
            raise ChatServiceError("Unexpected Ollama response structure.") from exc

        self._last_raw_response = content
        try:
            parsed = json.loads(content)
        except JSONDecodeError as exc:
            preview = content.strip()
            if len(preview) > 160:
                preview = f"{preview[:160]}…"
            raise ChatServiceValidationError(
                f"Response was not valid JSON ({exc.msg}). Preview: {preview}"
            ) from exc

        self._apply_schema_defaults(parsed)
        try:
            self._validator.validate(parsed)
        except ValidationError as exc:
            raise ChatServiceValidationError(f"Response failed schema validation: {exc.message}") from exc

        return parsed

    def _apply_schema_defaults(self, data: dict) -> None:
        """Populate any missing optional fields with defaults before validation."""

        properties = self._schema.get("properties", {})
        for key, definition in properties.items():
            if key not in data and "default" in definition:
                data[key] = deepcopy(definition["default"])

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    def _format_context(self, snippets: Sequence[ContextSnippet]) -> str:
        if not snippets:
            return "NO_SNIPPETS_AVAILABLE"
        blocks: List[str] = []
        for idx, snippet in enumerate(snippets, start=1):
            block = textwrap.dedent(
                f"""\
                SNIPPET {idx}:
                citation: {snippet.citation}
                rationale: {snippet.rationale}
                text: {snippet.content}"""
            ).strip()
            blocks.append(self._escape_braces(block))
        return "\n\n".join(blocks)


__all__ = ["ChatService", "ChatServiceError", "ChatServiceValidationError"]
