from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import httpx

from ..core.settings import settings
from ..models.schema import HealthProbe, HealthStatus

logger = logging.getLogger(__name__)


class ReadinessService:
    """Lightweight readiness checks for the local assistant."""

    def __init__(self) -> None:
        self._http_client = httpx.Client(timeout=settings.ollama_timeout_seconds)

    def close(self) -> None:
        self._http_client.close()

    def run_checks(self) -> HealthStatus:
        tags_payload, tags_error = self._fetch_tags()
        probes: List[HealthProbe] = [
            self._check_ollama_daemon(tags_error),
            self._check_ollama_model(tags_payload, tags_error),
        ]
        status = "pass" if all(probe.healthy for probe in probes) else "fail"
        return HealthStatus(status=status, probes=probes)

    def _fetch_tags(self) -> Tuple[Dict[str, Any] | None, str | None]:
        url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
        try:
            response = self._http_client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            detail = f"Ollama unreachable: {exc!s}"
            logger.debug(detail, exc_info=exc)
            return None, detail
        try:
            payload: Dict[str, Any] = response.json()
        except ValueError as exc:
            detail = f"Invalid response from Ollama: {exc!s}"
            logger.debug(detail, exc_info=exc)
            return None, detail
        return payload, None

    def _check_ollama_daemon(self, tags_error: str | None) -> HealthProbe:
        if tags_error:
            return HealthProbe(name="ollama_daemon", healthy=False, detail=tags_error, checked_at=datetime.utcnow())
        return HealthProbe(name="ollama_daemon", healthy=True, detail="OK", checked_at=datetime.utcnow())

    def _check_ollama_model(
        self, payload: Dict[str, Any] | None, tags_error: str | None
    ) -> HealthProbe:
        if tags_error:
            return HealthProbe(name="ollama_chat_model", healthy=False, detail=tags_error, checked_at=datetime.utcnow())
        required = settings.ollama_chat_model
        models = {model.get("name") or model.get("model") for model in (payload or {}).get("models", [])}
        normalized = {item for item in models if item}
        normalized.update({item.split(":", 1)[0] for item in normalized if ":" in item})
        if required not in normalized:
            detail = f"Missing model: {required}"
            return HealthProbe(name="ollama_chat_model", healthy=False, detail=detail, checked_at=datetime.utcnow())
        return HealthProbe(name="ollama_chat_model", healthy=True, detail="OK", checked_at=datetime.utcnow())


__all__ = ["ReadinessService"]
