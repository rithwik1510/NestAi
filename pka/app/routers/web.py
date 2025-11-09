from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ... import __version__
from ..core.settings import settings
from ..services.health import ReadinessService

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "web" / "templates"))
templates.env.globals["static_version"] = __version__

router = APIRouter(tags=["web"])


def _load_validation_report() -> dict | None:
    report_path = Path(__file__).resolve().parents[2] / "validation_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@router.get("/", response_class=HTMLResponse)
async def chat_home(request: Request) -> HTMLResponse:
    """Render the chat interface shell."""

    context = {
        "request": request,
        "title": "NestAi",
        "active_page": "home",
        "assistant_model": settings.ollama_chat_model,
        "ollama_url": settings.ollama_base_url,
        "settings": settings,
    }
    return templates.TemplateResponse("chat.html", context)


@router.get("/library", response_class=HTMLResponse)
async def library_view(request: Request) -> HTMLResponse:
    """Render conversation library placeholder."""

    context = {
        "request": request,
        "title": "Library - NestAi",
        "active_page": "library",
        "assistant_model": settings.ollama_chat_model,
        "ollama_url": settings.ollama_base_url,
        "settings": settings,
    }
    return templates.TemplateResponse("library.html", context)


@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request) -> HTMLResponse:
    """Render settings placeholder."""

    context = {
        "request": request,
        "title": "Settings - NestAi",
        "active_page": "settings",
        "assistant_model": settings.ollama_chat_model,
        "ollama_url": settings.ollama_base_url,
        "settings": settings,
    }
    return templates.TemplateResponse("settings.html", context)


@router.get("/diagnostics", response_class=HTMLResponse)
async def diagnostics_view(request: Request) -> HTMLResponse:
    """Render diagnostics panel with live readiness and last validation report."""

    readiness_service: ReadinessService = request.app.state.readiness_service
    readiness = readiness_service.run_checks()
    validation_report = _load_validation_report()

    context = {
        "request": request,
        "title": "Diagnostics - NestAi",
        "active_page": "diagnostics",
        "assistant_model": settings.ollama_chat_model,
        "ollama_url": settings.ollama_base_url,
        "readiness": readiness,
        "validation_report": validation_report,
        "settings": settings,
    }
    return templates.TemplateResponse("diagnostics.html", context)


__all__ = ["router"]
