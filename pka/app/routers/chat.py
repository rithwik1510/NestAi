from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, Request

from ..core.settings import settings
from ..models.schema import ChatRequest, ChatResponse
from ..services.assistant import AssistantService, AssistantServiceError

router = APIRouter(prefix="/api", tags=["chat"])


def get_assistant_service(request: Request) -> AssistantService:
    return request.app.state.assistant_service


@router.post("/chat", response_model=ChatResponse, summary="Answer a question using the local assistant")
async def chat_endpoint(
    payload: ChatRequest,
    assistant: AssistantService = Depends(get_assistant_service),
) -> ChatResponse:
    started = time.perf_counter()
    try:
        answer = await assistant.generate(payload.question)
    except AssistantServiceError as exc:
        raise RuntimeError(str(exc)) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    return ChatResponse(
        run_id=uuid.uuid4(),
        latency_ms=latency_ms,
        answer=answer,
        context=[],
        question=payload.question,
        mode=payload.mode,
        llm_version=settings.ollama_chat_model,
        prompt_version="assistant-v1",
        template_hash="assistant-v1",
    )
