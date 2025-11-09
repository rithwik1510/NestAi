from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pka import __version__

from .core.logging import configure_logging
from .core.settings import settings
from .services.assistant import AssistantService
from .services.health import ReadinessService


def create_app() -> FastAPI:
    """Application factory for FastAPI initialization."""

    configure_logging()

    from .services.docs import DocumentService

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    readiness_service = ReadinessService()
    app.state.readiness_service = readiness_service

    static_dir = Path(__file__).resolve().parent / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    assistant_service = AssistantService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout=settings.ollama_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
    )

    app.state.assistant_service = assistant_service

    @app.on_event("startup")
    async def _run_startup_checks() -> None:
        if settings.skip_health_checks:
            logging.getLogger(__name__).warning("Skipping readiness checks (skip_health_checks=True)")
            return
        status = readiness_service.run_checks()
        if status.status != "pass":
            failed = ", ".join(probe.name for probe in status.probes if not probe.healthy)
            message = f"Readiness checks failed: {failed or 'unknown'}"
            logging.getLogger(__name__).error(message)
            raise RuntimeError(message)

    @app.on_event("shutdown")
    async def _cleanup_resources() -> None:
        readiness_service.close()
        await assistant_service.close()

    register_routers(app)
    return app


def register_routers(app: FastAPI) -> None:
    """Attach API routers to the FastAPI app instance."""

    from .routers import chat, health, web  # local import to avoid circular dependencies

    app.include_router(web.router)
    app.include_router(health.router)
    app.include_router(chat.router)


app = create_app()


__all__ = ["app", "create_app", "register_routers"]
