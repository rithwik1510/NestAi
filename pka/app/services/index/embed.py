from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...core.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding service fails."""


class EmbeddingService:
    """Embedding client for Ollama models with deterministic validation."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 60,
        batch_size: int = 16,
        expected_dim: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.expected_dim = expected_dim or settings.vector_dim
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Return embeddings for the given texts."""

        inputs = [text for text in texts]
        if not inputs:
            return []

        vectors: List[List[float]] = []
        for offset in range(0, len(inputs), self.batch_size):
            batch = inputs[offset : offset + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        return vectors

    def embed_query(self, query: str) -> Sequence[float]:
        """Return embedding for a single query string."""

        [vector] = self.embed_texts([query])
        return vector

    @retry(
        retry=retry_if_exception_type(EmbeddingServiceError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        if not batch:
            return []

        payload = {
            "model": self.model,
            # Always send a list so Ollama returns a deterministic embeddings array.
            "input": list(batch),
        }

        try:
            response = self._client.post("/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            logger.debug("Embedding request timed out: %s", exc, exc_info=exc)
            raise EmbeddingServiceError("Ollama embedding request timed out.") from exc
        except httpx.HTTPError as exc:
            logger.debug("Embedding request failed: %s", exc, exc_info=exc)
            raise EmbeddingServiceError(f"Ollama embedding request failed: {exc!s}") from exc
        except ValueError as exc:
            logger.debug("Embedding response parsing failed: %s", exc, exc_info=exc)
            raise EmbeddingServiceError(f"Invalid embedding response: {exc!s}") from exc

        raw_embeddings: List[List[float]] | None = None
        if isinstance(data.get("embeddings"), list):
            raw_embeddings = data["embeddings"]
        elif isinstance(data.get("embedding"), list):
            raw_embeddings = [data["embedding"]]
        elif isinstance(data.get("data"), list):
            collected: List[List[float]] = []
            for item in data["data"]:
                if isinstance(item, list):
                    collected.append(item)
                elif isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    collected.append(item["embedding"])
            if collected:
                raw_embeddings = collected

        if raw_embeddings is None:
            raise EmbeddingServiceError("Unexpected embedding payload format.")

        vectors: List[List[float]] = []
        for vector in raw_embeddings:
            try:
                floats = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingServiceError("Embedding contains non-numeric values.") from exc
            if self.expected_dim and len(floats) != self.expected_dim:
                raise EmbeddingServiceError(
                    f"Embedding dimension mismatch: expected {self.expected_dim}, got {len(floats)}"
                )
            vectors.append(floats)

        if len(vectors) != len(batch):
            raise EmbeddingServiceError(
                f"Ollama returned {len(vectors)} embeddings for {len(batch)} inputs."
            )

        return vectors


__all__ = ["EmbeddingService", "EmbeddingServiceError"]
