from __future__ import annotations

import importlib
import json
from urllib.error import URLError


def _reload_module():
    module = importlib.import_module("src.monitoring.sentry")
    return importlib.reload(module)


def test_init_sentry_returns_false_without_dsn(monkeypatch) -> None:
    module = _reload_module()

    monkeypatch.setenv("SENTRY_DSN", "")
    monkeypatch.setattr(module, "_SENTRY_CONFIG", None)

    assert module.init_sentry(service="api") is False


def test_init_sentry_builds_store_endpoint(monkeypatch) -> None:
    module = _reload_module()

    monkeypatch.setenv("SENTRY_DSN", "https://public@example.ingest.sentry.io/123")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "production")
    monkeypatch.setenv("SENTRY_RELEASE", "abc123")
    monkeypatch.setattr(module, "_SENTRY_CONFIG", None)

    assert module.init_sentry(service="api", include_fastapi_integration=True) is True
    config = module._SENTRY_CONFIG
    assert config is not None
    assert config.endpoint == "https://example.ingest.sentry.io/api/123/store/"
    assert config.environment == "production"
    assert config.release == "abc123"
    assert config.service == "api"


def test_capture_exception_posts_event(monkeypatch) -> None:
    module = _reload_module()
    sent_requests: list[tuple[str, dict, dict]] = []

    monkeypatch.setenv("SENTRY_DSN", "https://examplePublicKey@example.ingest.sentry.io/1")
    monkeypatch.setattr(module, "_SENTRY_CONFIG", None)

    def _fake_urlopen(request, timeout):
        sent_requests.append(
            (
                request.full_url,
                dict(request.headers),
                json.loads(request.data.decode("utf-8")),
            )
        )

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Response()

    monkeypatch.setattr(module, "urlopen", _fake_urlopen)

    assert module.init_sentry(service="worker")
    module.capture_exception(RuntimeError("boom"), context={"job_id": "job_1"})

    assert len(sent_requests) == 1
    url, headers, payload = sent_requests[0]
    assert url == "https://example.ingest.sentry.io/api/1/store/"
    assert headers["X-sentry-auth"].startswith("Sentry sentry_version=7")
    assert payload["exception"]["values"][0]["value"] == "boom"
    assert payload["extra"]["job_id"] == "job_1"


def test_capture_exception_swallow_send_failures(monkeypatch) -> None:
    module = _reload_module()

    monkeypatch.setenv("SENTRY_DSN", "https://examplePublicKey@example.ingest.sentry.io/1")
    monkeypatch.setattr(module, "_SENTRY_CONFIG", None)
    monkeypatch.setattr(module, "urlopen", lambda request, timeout: (_ for _ in ()).throw(URLError("nope")))

    assert module.init_sentry(service="worker")
    module.capture_exception(RuntimeError("boom"))
