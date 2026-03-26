from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from urllib import error as urllib_error

import pytest

import src.cli as cli


class _FakeBinaryStdin:
    def __init__(self, data: bytes, *, is_tty: bool = False) -> None:
        self.buffer = io.BytesIO(data)
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty

    def readline(self) -> str:
        return ""

    def read(self) -> str:
        return self.buffer.getvalue().decode("utf-8")


def _read_stdout_json(capsys) -> object:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _render_help(argv: list[str], capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv + ["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


def test_root_help_uses_vmf_and_stays_shallow(capsys) -> None:
    help_text = _render_help([], capsys)

    assert help_text.startswith("usage: vmf")
    assert "auth" in help_text
    assert "keys" in help_text
    assert "videos" in help_text
    assert "Run `vmf <command> --help` for examples" in help_text
    assert "vmf videos upload ./sample.mp4" not in help_text


def test_version_flag_reports_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == "vmf 0.1.0"


def test_subcommand_help_includes_examples_and_defaults(capsys) -> None:
    wait_help = _render_help(["videos", "wait"], capsys)
    assert "Examples:" in wait_help
    assert "vmf videos wait 11111111-1111-4111-8111-111111111111" in wait_help
    assert "default:" in wait_help
    assert "2.0" in wait_help
    assert "1200.0" in wait_help

    clear_help = _render_help(["auth", "clear"], capsys)
    assert "--dry-run" in clear_help
    assert "--yes" in clear_help
    assert "vmf auth clear --dry-run" in clear_help

    upload_help = _render_help(["videos", "upload"], capsys)
    assert "cat sample.mp4 | vmf videos upload - --filename sample.mp4 --content-type video/mp4" in upload_help
    assert "--idempotency-key" in upload_help

    list_help = _render_help(["videos", "list"], capsys)
    assert "vmf videos list" in list_help


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

    rc = cli.main(["auth", "clear", "--yes"])
    assert rc == 0
    clear_payload = _read_stdout_json(capsys)
    assert clear_payload["cleared"] is True
    assert not config_path.exists()


def test_auth_clear_dry_run_is_non_mutating(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    config_path.write_text('{"api_base_url":"https://api.example.com","api_key":"vmf_dead"}\n', encoding="utf-8")
    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))

    rc = cli.main(["auth", "clear", "--dry-run"])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload == {
        "dry_run": True,
        "action": "auth_clear",
        "config_path": str(config_path),
        "config_exists": True,
    }
    assert config_path.exists()


def test_auth_clear_requires_yes_in_non_interactive_mode(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))

    rc = cli.main(["auth", "clear"])

    captured = capsys.readouterr()
    assert rc == 2
    assert config_path.exists()
    assert "Refusing to clear local CLI auth config in non-interactive mode without --yes." in captured.err
    assert "vmf auth clear --yes" in captured.err


def test_auth_clear_interactive_accepts_confirmation(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("yes\n"))

    rc = cli.main(["auth", "clear"])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["cleared"] is True
    assert "Clear local CLI auth config" in captured.err
    assert not config_path.exists()


def test_auth_clear_interactive_cancel_keeps_config(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "vmf-config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))

    rc = cli.main(["auth", "clear"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "Cancelled" in captured.err
    assert config_path.exists()


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
        "--yes",
        "key_1",
    ])
    assert rc == 0
    revoke_payload = _read_stdout_json(capsys)
    assert revoke_payload == {"key_id": "key_1", "revoked": True}

    assert calls == [
        ("GET", "https://api.example.com/api/v1/keys", None),
        ("DELETE", "https://api.example.com/api/v1/keys/key_1", None),
    ]


def test_keys_revoke_dry_run_is_non_mutating(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_json_request", lambda *args, **kwargs: pytest.fail("should not call API"))

    rc = cli.main([
        "keys",
        "revoke",
        "--api-base-url",
        "https://api.example.com",
        "key_1",
        "--dry-run",
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload == {
        "dry_run": True,
        "action": "keys_revoke",
        "key_id": "key_1",
        "api_base_url": "https://api.example.com",
        "api_base_url_source": "flag",
    }


def test_videos_upload_uses_multipart_endpoint(monkeypatch, tmp_path, capsys) -> None:
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"video-bytes")
    calls: list[dict[str, object]] = []

    def fake_multipart_request(method, url, *, headers=None, field_name, file_path, filename, content_type, timeout_s=300.0):
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "field_name": field_name,
                "file_path": str(file_path),
                "filename": filename,
                "content_type": content_type,
            }
        )
        return {
            "id": "vid_123",
            "status": "queued",
            "source_type": "upload",
            "source_filename": filename,
        }

    monkeypatch.setattr(cli, "_multipart_json_request", fake_multipart_request)

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
    assert calls == [
        {
            "method": "POST",
            "url": "https://api.example.com/api/v1/videos/upload",
            "headers": {"Authorization": "Bearer vmf_deadbeef12345678deadbeef12345678"},
            "field_name": "file",
            "file_path": str(video_file),
            "filename": "clip.mp4",
            "content_type": "video/mp4",
        }
    ]


def test_videos_upload_forwards_idempotency_key(monkeypatch, tmp_path, capsys) -> None:
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"video-bytes")
    seen_headers: list[dict[str, str]] = []

    monkeypatch.setattr(
        cli,
        "_multipart_json_request",
        lambda *args, **kwargs: seen_headers.append(kwargs["headers"]) or {"id": "vid_123", "status": "queued"},
    )

    rc = cli.main([
        "videos",
        "upload",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "--idempotency-key",
        "retry-1",
        str(video_file),
    ])

    assert rc == 0
    _read_stdout_json(capsys)
    assert seen_headers == [
        {
            "Authorization": "Bearer vmf_deadbeef12345678deadbeef12345678",
            "Idempotency-Key": "retry-1",
        }
    ]


def test_videos_upload_stdin_requires_filename_and_content_type(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeBinaryStdin(b"video-bytes"))

    rc = cli.main([
        "videos",
        "upload",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "-",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "stdin uploads require --filename" in captured.err


def test_videos_upload_reads_stdin_and_cleans_temp_file(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "stdin", _FakeBinaryStdin(b"video-bytes"))

    def fake_multipart_request(method, url, *, headers=None, field_name, file_path, filename, content_type, timeout_s=300.0):
        path = Path(file_path)
        captured["path"] = path
        captured["payload"] = path.read_bytes()
        captured["filename"] = filename
        captured["content_type"] = content_type
        return {"id": "vid_123", "status": "queued"}

    monkeypatch.setattr(cli, "_multipart_json_request", fake_multipart_request)

    rc = cli.main([
        "videos",
        "upload",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "--filename",
        "clip.mp4",
        "--content-type",
        "video/mp4",
        "-",
    ])

    assert rc == 0
    _read_stdout_json(capsys)
    assert captured["payload"] == b"video-bytes"
    assert captured["filename"] == "clip.mp4"
    assert captured["content_type"] == "video/mp4"
    assert not Path(captured["path"]).exists()


def test_videos_list_returns_api_response(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_json_request",
        lambda *args, **kwargs: [
            {"id": "vid_123", "status": "ready"},
            {"id": "vid_456", "status": "processing"},
        ],
    )

    rc = cli.main([
        "videos",
        "list",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
    ])

    assert rc == 0
    payload = _read_stdout_json(capsys)
    assert payload == [
        {"id": "vid_123", "status": "ready"},
        {"id": "vid_456", "status": "processing"},
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
    assert "Video processing failed: boom" in captured.err


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
    assert "Timed out waiting for video vid_123 (last status: processing)" in captured.err


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


def test_videos_search_reads_query_text_from_stdin(monkeypatch, capsys) -> None:
    seen_payloads: list[dict[str, object]] = []
    monkeypatch.setattr(sys, "stdin", io.StringIO("explain the model\n"))
    monkeypatch.setattr(
        cli,
        "_json_request",
        lambda *args, **kwargs: seen_payloads.append(kwargs["payload"]) or {
            "video_id": "vid_123",
            "status": "ready",
            "results": [],
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
        "-",
    ])

    assert rc == 0
    _read_stdout_json(capsys)
    assert seen_payloads == [{"query_text": "explain the model", "limit": 5}]


def test_videos_search_rejects_empty_query_text_from_stdin(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(" \n"))

    rc = cli.main([
        "videos",
        "search",
        "--api-base-url",
        "https://api.example.com",
        "--api-key",
        "vmf_deadbeef12345678deadbeef12345678",
        "vid_123",
        "--query-text",
        "-",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "No query text received on stdin" in captured.err


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
        if url.endswith("/videos/vid_123") and method == "GET":
            state["wait_calls"] += 1
            if state["wait_calls"] < 2:
                return {"id": "vid_123", "status": "processing"}
            return {"id": "vid_123", "status": "ready"}
        if url.endswith("/videos") and method == "GET":
            return [{"id": "vid_123", "status": "queued"}]
        if url.endswith("/videos/vid_123/search") and method == "POST":
            return {
                "video_id": "vid_123",
                "status": "ready",
                "results": [{"timestamp_s": 4.0, "score": 0.88, "source": "transcript"}],
            }
        raise AssertionError(f"Unexpected request: {method} {url}")

    def fake_multipart_request(method, url, *, headers=None, field_name, file_path, filename, content_type, timeout_s=300.0):
        assert method == "POST"
        assert url == "https://api.example.com/api/v1/videos/upload"
        assert filename == "clip.mp4"
        assert Path(file_path).name == "clip.mp4"
        assert headers == {"Authorization": "Bearer vmf_deadbeef12345678deadbeef12345678"}
        return {
            "id": "vid_123",
            "status": "queued",
            "source_type": "upload",
            "source_filename": "clip.mp4",
        }

    monkeypatch.setenv(cli.ENV_CONFIG_PATH, str(config_path))
    monkeypatch.setattr(cli, "_json_request", fake_json_request)
    monkeypatch.setattr(cli, "_multipart_json_request", fake_multipart_request)
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

    assert cli.main(["videos", "list"]) == 0
    list_payload = _read_stdout_json(capsys)
    assert list_payload == [{"id": "vid_123", "status": "queued"}]

    assert cli.main(["videos", "wait", "vid_123"]) == 0
    wait_payload = _read_stdout_json(capsys)
    assert wait_payload["status"] == "ready"

    assert cli.main(["videos", "search", "vid_123", "--query-text", "hello"]) == 0
    search_payload = _read_stdout_json(capsys)
    assert search_payload["results"][0]["source"] == "transcript"
