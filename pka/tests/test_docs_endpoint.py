from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pka.app.routers.docs import get_db_session, get_document_service, router as docs_router


class StubDocumentService:
    def __init__(self, document):
        self._document = document
        self.requested_ids = []

    def fetch_document(self, session, document_id: int):
        self.requested_ids.append(document_id)
        return self._document


def _app_with_overrides(service) -> TestClient:
    app = FastAPI()
    app.include_router(docs_router)

    def _override_session():
        yield None

    app.dependency_overrides[get_db_session] = _override_session
    app.dependency_overrides[get_document_service] = lambda: service
    return TestClient(app)


def test_fetch_document_success():
    chunk = SimpleNamespace(
        id=10,
        ordinal=1,
        text="This is a sample preview.",
        start_line=1,
        end_line=5,
        page_no=None,
        token_count=100,
    )
    document = SimpleNamespace(
        id=1,
        path="/tmp/doc.md",
        title="Doc",
        type="md",
        size=123,
        sha256="abc",
        created_at=None,
        updated_at=None,
        confidentiality_tag="private",
        meta={"tags": ["test"]},
        chunks=[chunk],
    )
    service = StubDocumentService(document)
    client = _app_with_overrides(service)

    resp = client.get("/api/docs/1")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] == 1
    assert payload["chunk_count"] == 1
    assert payload["chunks"][0]["preview"] == "This is a sample preview."
    assert service.requested_ids == [1]


def test_fetch_document_not_found():
    service = StubDocumentService(None)
    client = _app_with_overrides(service)

    resp = client.get("/api/docs/999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Document not found"
    assert service.requested_ids == [999]





