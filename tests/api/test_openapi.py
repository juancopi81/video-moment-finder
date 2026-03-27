from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_public_openapi_exposes_only_curated_routes() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]

    assert "/api/v1/keys" in paths
    assert "/api/v1/billing/units/summary" in paths
    assert "/api/v1/billing/units/usage" in paths
    assert "/api/v1/billing/units/checkout" in paths
    assert "/api/v1/videos/upload" in paths
    assert "/api/v1/videos/upload/init" in paths
    assert "/api/v1/videos/upload/complete" in paths
    assert "/api/v1/videos" in paths
    assert "/api/v1/videos/{video_id}" in paths
    assert "/api/v1/videos/{video_id}/search" in paths

    assert set(paths["/api/v1/videos"]) == {"get"}

    assert "/analytics/event" not in paths
    assert "/webhooks/lemonsqueezy" not in paths
    assert "/api/v1/billing/credits/summary" not in paths
    assert "/api/v1/billing/credits/checkout" not in paths
    assert "/api/v1/videos/{video_id}/search/image" not in paths


def test_docs_uses_public_openapi_schema() -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "/openapi.json" in response.text
