from __future__ import annotations

import io
import json
from pathlib import Path
from urllib import error as urllib_error

import pytest

import src.cli as cli


def _read_stdout_json(capsys) -> object:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def test_auth_set_status_clear_and_precedence(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))

    rc = cli.main([
        "auth",
        "set",
        "--api-base-url",
        "https://api.example.com/",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload["saved"] is True
    assert payload["api_base_url"] == "https://api.example.com"
    assert config_path.exists()

    monkeypatch.setenv(cli.ENV_API_BASE_URL, "https://env.example.com")
    monkeypatch.setenv(cli.ENV_API_KEY, "vmf_feedface12345678feedface12345678")

    rc = cli.main(["auth", "status"])
    assert rc == 0
    status_payload = _read_stdout_json(capsys)
    assert status_payload["api_base_url"] == "https://env.example.com"
    assert status_payload["api_base_url_source"] == "env"
    assert status_payload["api_key_source"] == "env"
    assert status_payload["api_key_masked"] == "vmf_feed...5678"

    monkeypatch.delenv(cli.ENV_API_BASE_URL)
    monkeypatch.delenv(cli.ENV_API_KEY)

    rc = cli.main(["auth", "clear"])
    assert rc == 0
    clear_payload = _read_stdout_json(capsys)
    assert clear_payload["cleared"] is True
    assert not config_path.exists()


def test_keys_create_saves_config_by_default(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))

    def fake_json_request(method, url, *, headers=None, payload=None, timeout_s=30.0):
        assert method == "POST"
        assert url == "https://api.example.com/api/v1/keys"
        assert headers == {"Authorization": "Bearer jwt_token"}
        assert payload == {"name": "agent"}
        return {
            "id": "key_123",
            "name": "agent",
            "key_prefix": "vmf_dead",
            "key": "vmf_deadbeef12345678deadbeef12345678",
        }

    monkeypatch.setattr(cli, "_json_request", fake_json_request)

    rc = cli.main([
        "keys",
        "create",
        "--api-base-url",
        "https://api.example.com",
        "--bearer-token",
        "jwt_token",
        "--name",
        "agent",
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload["id"] == "key_123"
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    assert stored == {
        "api_base_url": "https://api.example.com",
        "api_key": "vmf_deadbeef12345678deadbeef12345678",
    }

    rc = cli.main(["auth", "status"])
    assert rc == 0
    status_payload = _read_stdout_json(capsys)
    assert status_payload["api_key_source"] == "config"
    assert status_payload["api_key_masked"] == "vmf_dead...5678"


def test_keys_create_no_save_leaves_config_unchanged(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))

    monkeypatch.setattr(
        cli,
        "_json_request",
        lambda *args, **kwargs: {
            "id": "key_123",
            "name": "",
            "key_prefix": "vmf_dead",
            "key": "vmf_deadbeef12345678deadbeef12345678",
        },
    )

    rc = cli.main([
        "keys",
        "create",
        "--api-base-url",
        "https://api.example.com",
        "--bearer-token",
        "jwt_token",
        "--no-save",
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload["key_prefix"] == "vmf_dead"
    assert not config_path.exists()


def test_keys_list_and_revoke_use_bearer_token(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str, dict | None]] = []

    def fake_json_request(method, url, *, headers=None, payload=None, timeout_s=30.0):
        calls.append((method, url, payload))
        assert headers == {"Authorization": "Bearer jwt_token"}
        if method == "GET":
            return [{"id": "key_1", "name": "first", "key_prefix": "vmf_abcd"}]
        return None

    monkeypatch.setattr(cli, "_json_request", fake_json_request)

    rc = cli.main([
        "keys",
        "list",
        "--api-base-url",
        "https://api.example.com",
        "--bearer-token",
        "jwt_token",
    ])
    assert rc == 0
    list_payload = _read_stdout_json(capsys)
    assert list_payload == [{"id": "key_1", "name": "first", "key_prefix": "vmf_abcd"}]

    rc = cli.main([
        "keys",
        "revoke",
        "--api-base-url",
        "https://api.example.com",
        "--bearer-token",
        "jwt_token",
        "key_1",
    ])
    assert rc == 0
    revoke_payload = _read_stdout_json(capsys)
    assert revoke_payload == {"key_id": "key_1", "revoked": True}

    assert calls == [
        ("GET", "https://api.example.com/api/v1/keys", None),
        ("DELETE", "https://api.example.com/api/v1/keys/key_1", None),
    ]


def test_videos_upload_uses_direct_upload_flow(monkeypatch, tmp_path, capsys) -> None:
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"video-bytes")
    requests: list[tuple[str, str, dict | None]] = []
    upload_calls: list[tuple[str, str, str | None]] = []

    def fake_json_request(method, url, *, headers=None, payload=None, timeout_s=30.0):
        requests.append((method, url, payload))
        assert headers == {"Authorization": "Bearer vmf_deadbeef12345678deadbeef12345678"}
        if url.endswith("/upload/init"):
            return {
                "video_id": "vid_123",
                "key": "source/vid_123/clip.mp4",
                "upload_url": "https://uploads.example.com/source/vid_123/clip.mp4",
                "expires_in": 900,
            }
        return {
            "id": "vid_123",
            "status": "queued",
            "source_type": "upload",
            "source_filename": "clip.mp4",
        }

    def fake_upload_put(upload_url: str, file_path: Path, *, content_type: str | None, timeout_s=300.0):
        upload_calls.append((upload_url, file_path.name, content_type))

    monkeypatch.setattr(cli, "_json_request", fake_json_request)
    monkeypatch.setattr(cli, "_stream_upload_put", fake_upload_put)

    rc = cli.main([
        "videos",
        "upload",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        str(video_file),
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload["id"] == "vid_123"
    assert upload_calls == [
        ("https://uploads.example.com/source/vid_123/clip.mp4", "clip.mp4", "video/mp4")
    ]
    assert requests == [
        (
            "POST",
            "https://api.example.com/api/v1/videos/upload/init",
            {"filename": "clip.mp4", "content_type": "video/mp4"},
        ),
        (
            "POST",
            "https://api.example.com/api/v1/videos/upload/complete",
            {"video_id": "vid_123", "filename": "clip.mp4"},
        ),
    ]


def test_videos_wait_returns_ready_video(monkeypatch, capsys) -> None:
    responses = [
        {"id": "vid_123", "status": "queued"},
        {"id": "vid_123", "status": "processing"},
        {"id": "vid_123", "status": "ready"},
    ]
    sleep_calls: list[float] = []

    def fake_json_request(method, url, *, headers=None, payload=None, timeout_s=30.0):
        assert method == "GET"
        return responses.pop(0)

    monkeypatch.setattr(cli, "_json_request", fake_json_request)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    rc = cli.main([
        "videos",
        "wait",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "vid_123",
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload["status"] == "ready"
    assert sleep_calls == [2.0, 2.0]


def test_videos_wait_failed_returns_payload_and_nonzero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_json_request",
        lambda *args, **kwargs: {"id": "vid_123", "status": "failed", "error_message": "boom"},
    )

    rc = cli.main([
        "videos",
        "wait",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "vid_123",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.out)["status"] == "failed"
    assert "Video processing failed" in captured.err


def test_videos_wait_timeout_returns_payload_and_nonzero(monkeypatch, capsys) -> None:
    times = iter([0.0, 1.0, 3.0])
    monkeypatch.setattr(
        cli,
        "_json_request",
        lambda *args, **kwargs: {"id": "vid_123", "status": "processing"},
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))

    rc = cli.main([
        "videos",
        "wait",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "vid_123",
        "--timeout-seconds",
        "2",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.out)["status"] == "processing"
    assert "Timed out waiting for video vid_123" in captured.err


def test_videos_search_returns_api_response(monkeypatch, capsys) -> None:
    seen_timeout: list[float] = []

    monkeypatch.setattr(
        cli,
        "_json_request",
        lambda *args, **kwargs: seen_timeout.append(kwargs["timeout_s"]) or {
            "video_id": "vid_123",
            "status": "ready",
            "results": [{"timestamp_s": 12.5, "score": 0.91, "source": "visual"}],
        },
    )

    rc = cli.main([
        "videos",
        "search",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "vid_123",
        "--query-text",
        "robot in blue hoodie",
        "--limit",
        "3",
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload["results"][0]["timestamp_s"] == 12.5
    assert seen_timeout == [cli.DEFAULT_SEARCH_TIMEOUT_S]


def test_json_request_formats_http_error_detail(monkeypatch) -> None:
    def fake_urlopen(req, timeout):
        raise urllib_error.HTTPError(
            url=req.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"Invalid authentication token"}'),
        )

    monkeypatch.setattr(cli.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(cli.CliError, match="HTTP 401: Invalid authentication token"):
        cli._json_request("GET", "https://api.example.com/api/v1/videos/vid_123")


def test_json_request_formats_timeout_error(monkeypatch) -> None:
    def fake_urlopen(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(cli.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(cli.CliError, match=r"Request timed out after 30s"):
        cli._json_request("GET", "https://api.example.com/api/v1/videos/vid_123")


def test_cli_happy_path_smoke(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"video-bytes")
    state = {"wait_calls": 0}

    def fake_json_request(method, url, *, headers=None, payload=None, timeout_s=30.0):
        if url.endswith("/keys") and method == "POST":
            return {
                "id": "key_123",
                "name": "agent",
                "key_prefix": "vmf_dead",
                "key": "vmf_deadbeef12345678deadbeef12345678",
            }
        if url.endswith("/upload/init") and method == "POST":
            return {
                "video_id": "vid_123",
                "key": "source/vid_123/clip.mp4",
                "upload_url": "https://uploads.example.com/source/vid_123/clip.mp4",
                "expires_in": 900,
            }
        if url.endswith("/upload/complete") and method == "POST":
            return {
                "id": "vid_123",
                "status": "queued",
                "source_type": "upload",
                "source_filename": "clip.mp4",
            }
        if url.endswith("/videos/vid_123") and method == "GET":
            state["wait_calls"] += 1
            if state["wait_calls"] < 2:
                return {"id": "vid_123", "status": "processing"}
            return {"id": "vid_123", "status": "ready"}
        if url.endswith("/videos/vid_123/search") and method == "POST":
            return {
                "video_id": "vid_123",
                "status": "ready",
                "results": [{"timestamp_s": 4.0, "score": 0.88, "source": "transcript"}],
            }
        raise AssertionError(f"Unexpected request: {method} {url}")

    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))
    monkeypatch.setattr(cli, "_json_request", fake_json_request)
    monkeypatch.setattr(cli, "_stream_upload_put", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    assert cli.main([
        "keys",
        "create",
        "--api-base-url",
        "https://api.example.com",
        "--bearer-token",
        "jwt_token",
        "--name",
        "agent",
    ]) == 0
    key_payload = _read_stdout_json(capsys)
    assert key_payload["key"].startswith("vmf_")

    assert cli.main(["videos", "upload", str(video_file)]) == 0
    upload_payload = _read_stdout_json(capsys)
    assert upload_payload["status"] == "queued"

    assert cli.main(["videos", "wait", "vid_123"]) == 0
    wait_payload = _read_stdout_json(capsys)
    assert wait_payload["status"] == "ready"

    assert cli.main(["videos", "search", "vid_123", "--query-text", "hello"]) == 0
    search_payload = _read_stdout_json(capsys)
    assert search_payload["results"][0]["source"] == "transcript"
