from __future__ import annotations

import logging
from typing import Optional

import httpx

from ..models.schema import ChatAnswer

logger = logging.getLogger(__name__)


class AssistantServiceError(RuntimeError):
    """Raised when the assistant cannot complete a request."""


class AssistantService:
    """Thin wrapper around the local Ollama chat API for personal assistant responses."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        timeout: int = 60,
        keep_alive: Optional[str] = None,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._keep_alive = keep_alive
        self._system_prompt = (
            "You are a privacy-preserving personal assistant running entirely on the user's machine. "
            "Provide concise, helpful answers. If you are unsure, say so clearly."
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, question: str) -> ChatAnswer:
        prompt = question.strip()
        if not prompt:
            raise AssistantServiceError("Question cannot be empty.")

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        if self._seed is not None:
            payload["options"]["seed"] = self._seed
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise AssistantServiceError("Timed out waiting for Ollama response.") from exc
        except httpx.HTTPError as exc:
            raise AssistantServiceError(f"Ollama request failed: {exc!s}") from exc
        except ValueError as exc:
            raise AssistantServiceError(f"Invalid response from Ollama: {exc!s}") from exc

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            logger.debug("Unexpected Ollama payload: %s", data)
            raise AssistantServiceError("Malformed response from Ollama.") from exc

        text = content.strip()
        return ChatAnswer(
            abstain=False,
            answer=text,
            bullets=[],
            conflicts=[],
            sources=[],
        )


__all__ = ["AssistantService", "AssistantServiceError"]
