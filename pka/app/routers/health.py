from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..models.schema import HealthStatus
from ..services.health import ReadinessService

router = APIRouter(prefix="/health", tags=["health"])


def get_readiness_service(request: Request) -> ReadinessService:
    service: ReadinessService = request.app.state.readiness_service
    return service


@router.get("/", response_model=HealthStatus, summary="Application health status")
def health_check(service: ReadinessService = Depends(get_readiness_service)) -> HealthStatus:
    """Return current readiness state for core dependencies."""

    return service.run_checks()
