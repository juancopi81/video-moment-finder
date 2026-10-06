"""Search media must remain usable after disabling public bucket access."""
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth import AuthIdentity, get_current_user
from src.db.supabase import ApiUnitConsumeResult
from src.storage.config import R2Config, StorageConfigError
from src.storage.qdrant import SearchResult
from src.storage.r2 import R2StorageError
from tests.api.conftest import _authenticate, _video_record
from tests.api.test_app import QUERY_IMAGE_BYTES

VIDEO_ID = "00000000-0000-4000-8000-000000000456"


@pytest.fixture
def private_search(monkeypatch):
    _authenticate()
    monkeypatch.setenv("VIDEO_SOURCE_URL_TTL_S", "3600")
    monkeypatch.setattr("src.api.app.db_get_video",
                        lambda video_id, user_id: _video_record(video_id, status="ready"))
    config = R2Config(endpoint_url="https://storage.example.com", access_key_id="unit-test-key",
                      secret_access_key="unit-test-secret", bucket_name="private-media")
    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: config)
    results = [
        SearchResult(video_id=VIDEO_ID, frame_index=2, timestamp_s=4, score=.9,
                     thumbnail_url=f"https://public.example.com/thumb/{VIDEO_ID}/thumb_00002.jpg"),
        SearchResult(video_id=VIDEO_ID, frame_index=-1, timestamp_s=5, score=.8,
                     source="transcript", transcript_text="A spoken explanation."),
    ]
    monkeypatch.setattr("src.api.app.search_video_by_text_service", lambda **_: results)
    monkeypatch.setattr("src.api.app.search_video_by_image_service", lambda **_: results)
    return results


@pytest.mark.parametrize("auth_method,mode", [
    ("jwt", "text"), ("jwt", "image"), ("api_key", "text"), ("mcp_oauth", "text"),
])
def test_search_returns_temporary_owned_thumbnail_links(private_search, monkeypatch, auth_method, mode):
    app.dependency_overrides[get_current_user] = lambda: AuthIdentity(
        user_id="user_123", auth_method=auth_method,
        api_key_id="test-api-key" if auth_method == "api_key" else None,
    )
    monkeypatch.setattr("src.api.app.db_consume_api_units",
                        lambda **_: ApiUnitConsumeResult(allowed=True, remaining_balance=100))
    client = TestClient(app)
    if mode == "image":
        response = client.post(f"/api/v1/videos/{VIDEO_ID}/search/image",
                               files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")})
    else:
        response = client.post(f"/api/v1/videos/{VIDEO_ID}/search", json={"query_text": "explanation"})
    assert response.status_code == 200
    payload = response.json()
    url = urlsplit(payload["results"][0]["thumbnail_url"])
    assert url.hostname == "storage.example.com"
    assert url.path == f"/private-media/thumb/{VIDEO_ID}/thumb_00002.jpg"
    assert parse_qs(url.query)["X-Amz-Expires"] == ["3600"]
    assert "X-Amz-Signature" in parse_qs(url.query)
    assert payload["results"][1]["thumbnail_url"] is None
    assert payload["results"][1]["transcript_text"] == "A spoken explanation."


@pytest.mark.parametrize("path", ["/{video}/thumb_00002.jpg", "/private-media/{video}/thumb_00002.jpg"])
def test_search_preserves_legacy_owned_thumbnail_layout(private_search, path):
    private_search[0] = SearchResult(video_id=VIDEO_ID, frame_index=2, timestamp_s=4, score=.9,
                                    thumbnail_url="https://public.example.com" + path.format(video=VIDEO_ID))
    response = TestClient(app).post(f"/api/v1/videos/{VIDEO_ID}/search", json={"query_text": "explanation"})
    assert response.status_code == 200
    url = urlsplit(response.json()["results"][0]["thumbnail_url"])
    assert url.path == f"/private-media/{VIDEO_ID}/thumb_00002.jpg"


def test_search_never_signs_foreign_cached_object(private_search):
    private_search[0] = SearchResult(video_id=VIDEO_ID, frame_index=2, timestamp_s=4, score=.9,
                                    thumbnail_url="https://public.example.com/another-user/thumb_00002.jpg")
    response = TestClient(app).post(f"/api/v1/videos/{VIDEO_ID}/search", json={"query_text": "explanation"})
    assert response.status_code == 200
    url = urlsplit(response.json()["results"][0]["thumbnail_url"])
    assert url.path == f"/private-media/thumb/{VIDEO_ID}/thumb_00002.jpg"
    assert "another-user" not in response.text


@pytest.mark.parametrize("failure", ["config", "signing"])
def test_search_does_not_fall_back_to_public_url_on_storage_failure(private_search, monkeypatch, failure):
    def unavailable(*args, **kwargs):
        if failure == "config":
            raise StorageConfigError("unavailable")
        raise R2StorageError("unavailable")
    monkeypatch.setattr("src.api.app.R2Config.from_env" if failure == "config"
                        else "src.api.app.R2Store.generate_presigned_url", unavailable)
    response = TestClient(app).post(f"/api/v1/videos/{VIDEO_ID}/search", json={"query_text": "explanation"})
    assert response.status_code == 200
    assert response.json()["results"][0]["thumbnail_url"] is None
    assert response.json()["results"][1]["transcript_text"] == "A spoken explanation."
    assert "public.example.com" not in response.text


def test_foreign_video_is_rejected_before_any_thumbnail_signing(private_search, monkeypatch):
    monkeypatch.setattr("src.api.app.db_get_video", lambda *args, **kwargs: None)
    def unexpected_signing():
        pytest.fail("Storage must not be consulted for another user's video")
    monkeypatch.setattr("src.api.app.R2Config.from_env", unexpected_signing)
    response = TestClient(app).post(f"/api/v1/videos/{VIDEO_ID}/search", json={"query_text": "explanation"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found"
