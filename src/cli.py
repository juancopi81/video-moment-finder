"""Thin CLI for the external Video Moment Finder API."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import http.client
import json
import mimetypes
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, urlsplit

ENV_API_BASE_URL = "VMF_API_BASE_URL"
ENV_API_KEY = "VMF_API_KEY"
ENV_BEARER_TOKEN = "VMF_BEARER_TOKEN"
ENV_CONFIG_PATH = "VMF_CONFIG_PATH"

DEFAULT_API_TIMEOUT_S = 30.0
DEFAULT_UPLOAD_TIMEOUT_S = 300.0
DEFAULT_WAIT_INTERVAL_S = 2.0
DEFAULT_WAIT_TIMEOUT_S = 1200.0
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONFIG_DIR_NAME = "video-moment-finder"
CONFIG_FILE_NAME = "config.json"


class CliError(RuntimeError):
    """User-facing CLI error with exit code."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class LocalConfig:
    api_base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class ResolvedValue:
    value: str | None
    source: str


JsonRequestFn = Callable[..., Any]
UploadFn = Callable[..., None]


def _json_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: Any | None = None,
    timeout_s: float = DEFAULT_API_TIMEOUT_S,
) -> Any:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    req = urllib_request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as response:
            raw = response.read()
    except urllib_error.HTTPError as exc:
        detail = _extract_http_error_detail(exc)
        raise CliError(f"HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise CliError(f"Request failed: {reason}") from exc

    if not raw:
        return None

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("API returned a non-JSON response") from exc


def _stream_upload_put(
    upload_url: str,
    file_path: Path,
    *,
    content_type: str | None,
    timeout_s: float = DEFAULT_UPLOAD_TIMEOUT_S,
) -> None:
    parsed = urlsplit(upload_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CliError("Upload URL is invalid")

    connection_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"

    conn = connection_cls(parsed.hostname, port=parsed.port, timeout=timeout_s)
    try:
        conn.putrequest("PUT", target)
        conn.putheader("Content-Length", str(file_path.stat().st_size))
        if content_type:
            conn.putheader("Content-Type", content_type)
        conn.endheaders()

        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                conn.send(chunk)

        response = conn.getresponse()
        body = response.read()
    except OSError as exc:
        raise CliError(f"Upload failed: {exc}") from exc
    finally:
        conn.close()

    if 200 <= response.status < 300:
        return

    detail = _decode_text(body)
    if detail:
        raise CliError(f"Upload failed with status {response.status}: {detail}")
    raise CliError(f"Upload failed with status {response.status}")


def _decode_text(raw: bytes, *, limit: int = 240) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _extract_http_error_detail(exc: urllib_error.HTTPError) -> str:
    body = exc.read()
    if body:
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = _decode_text(body)
            if detail:
                return detail
        else:
            if isinstance(parsed, dict) and "detail" in parsed:
                return _format_detail(parsed["detail"])
            if isinstance(parsed, dict) and "message" in parsed:
                return _format_detail(parsed["message"])
            return _format_detail(parsed)
    return exc.reason or "Request failed"


def _format_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, separators=(",", ":"))


def _config_path() -> Path:
    override = os.environ.get(ENV_CONFIG_PATH, "").strip()
    if override:
        return Path(override).expanduser()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg_config_home:
        base = Path(xdg_config_home).expanduser()
    else:
        base = Path.home() / ".config"
    return base / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _load_local_config() -> LocalConfig:
    path = _config_path()
    if not path.exists():
        return LocalConfig()

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except OSError as exc:
        raise CliError(f"Failed to read config file {path}: {exc}", exit_code=2) from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"Config file {path} is not valid JSON", exit_code=2) from exc

    if not isinstance(parsed, dict):
        raise CliError(f"Config file {path} must contain a JSON object", exit_code=2)

    api_base_url = parsed.get("api_base_url")
    api_key = parsed.get("api_key")
    if api_base_url is not None and not isinstance(api_base_url, str):
        raise CliError(f"Config file {path} contains an invalid api_base_url", exit_code=2)
    if api_key is not None and not isinstance(api_key, str):
        raise CliError(f"Config file {path} contains an invalid api_key", exit_code=2)

    return LocalConfig(
        api_base_url=api_base_url.strip() or None if api_base_url else None,
        api_key=api_key.strip() or None if api_key else None,
    )


def _save_local_config(config: LocalConfig) -> Path:
    path = _config_path()
    data = {
        "api_base_url": config.api_base_url,
        "api_key": config.api_key,
    }
    payload = json.dumps(data, indent=2) + "\n"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_file.write(payload)
            temp_name = temp_file.name
        try:
            os.chmod(temp_name, 0o600)
        except OSError:
            pass
        os.replace(temp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError as exc:
        raise CliError(f"Failed to write config file {path}: {exc}", exit_code=2) from exc

    return path


def _clear_local_config() -> Path:
    path = _config_path()
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            raise CliError(f"Failed to clear config file {path}: {exc}", exit_code=2) from exc
    return path


def _normalize_api_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CliError("api_base_url must be a valid http(s) URL", exit_code=2)
    return cleaned


def _normalize_api_key(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise CliError("api_key must be non-empty", exit_code=2)
    if not cleaned.startswith("vmf_"):
        raise CliError("api_key must start with vmf_", exit_code=2)
    return cleaned


def _resolve_api_base_url(cli_value: str | None) -> ResolvedValue:
    if cli_value:
        return ResolvedValue(_normalize_api_base_url(cli_value), "flag")

    env_value = os.environ.get(ENV_API_BASE_URL, "").strip()
    if env_value:
        return ResolvedValue(_normalize_api_base_url(env_value), "env")

    config_value = _load_local_config().api_base_url
    if config_value:
        return ResolvedValue(_normalize_api_base_url(config_value), "config")

    return ResolvedValue(None, "missing")


def _resolve_api_key(cli_value: str | None) -> ResolvedValue:
    if cli_value:
        return ResolvedValue(_normalize_api_key(cli_value), "flag")

    env_value = os.environ.get(ENV_API_KEY, "").strip()
    if env_value:
        return ResolvedValue(_normalize_api_key(env_value), "env")

    config_value = _load_local_config().api_key
    if config_value:
        return ResolvedValue(_normalize_api_key(config_value), "config")

    return ResolvedValue(None, "missing")


def _resolve_bearer_token(cli_value: str | None) -> ResolvedValue:
    if cli_value and cli_value.strip():
        return ResolvedValue(cli_value.strip(), "flag")

    env_value = os.environ.get(ENV_BEARER_TOKEN, "").strip()
    if env_value:
        return ResolvedValue(env_value, "env")

    return ResolvedValue(None, "missing")


def _require_api_base_url(cli_value: str | None) -> str:
    resolved = _resolve_api_base_url(cli_value)
    if resolved.value is None:
        raise CliError(
            f"Missing API base URL. Provide --api-base-url, {ENV_API_BASE_URL}, or run `vmf auth set`.",
            exit_code=2,
        )
    return resolved.value


def _require_api_key(cli_value: str | None) -> str:
    resolved = _resolve_api_key(cli_value)
    if resolved.value is None:
        raise CliError(
            f"Missing API key. Provide --api-key, {ENV_API_KEY}, or run `vmf auth set`.",
            exit_code=2,
        )
    return resolved.value


def _require_bearer_token(cli_value: str | None) -> str:
    resolved = _resolve_bearer_token(cli_value)
    if resolved.value is None:
        raise CliError(
            f"Missing Clerk bearer token. Provide --bearer-token or {ENV_BEARER_TOKEN}.",
            exit_code=2,
        )
    return resolved.value


def _mask_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    if len(api_key) <= 12:
        return api_key[:4] + "..."
    return f"{api_key[:8]}...{api_key[-4:]}"


def _api_v1_url(base_url: str, path: str) -> str:
    return f"{base_url}/api/v1{path}"


def _authorization_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


def _ensure_dict(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CliError(f"{context} returned an unexpected response")
    return payload


def _ensure_file(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.exists():
        raise CliError(f"File not found: {path}", exit_code=2)
    if not path.is_file():
        raise CliError(f"Not a file: {path}", exit_code=2)
    return path


def _guess_video_content_type(path: Path) -> str | None:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("video/"):
        return guessed
    return None


def _quote_path_value(value: str) -> str:
    return quote(value, safe="")


def _maybe_print_payload_before_error(payload: Any, message: str) -> None:
    _print_json(payload)
    raise CliError(message)


def _cmd_auth_set(args: argparse.Namespace) -> Any:
    api_base_url = _normalize_api_base_url(args.api_base_url)
    api_key = _normalize_api_key(args.api_key)
    path = _save_local_config(LocalConfig(api_base_url=api_base_url, api_key=api_key))
    return {
        "saved": True,
        "config_path": str(path),
        "api_base_url": api_base_url,
        "api_key_masked": _mask_api_key(api_key),
    }


def _cmd_auth_status(args: argparse.Namespace) -> Any:
    config_path = _config_path()
    base_url = _resolve_api_base_url(args.api_base_url)
    api_key = _resolve_api_key(args.api_key)
    return {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "api_base_url": base_url.value,
        "api_base_url_source": base_url.source,
        "api_key_configured": api_key.value is not None,
        "api_key_masked": _mask_api_key(api_key.value),
        "api_key_source": api_key.source,
    }


def _cmd_auth_clear(_args: argparse.Namespace) -> Any:
    path = _clear_local_config()
    return {
        "cleared": True,
        "config_path": str(path),
    }


def _cmd_keys_create(
    args: argparse.Namespace,
    *,
    request_json: JsonRequestFn | None = None,
) -> Any:
    request_json = request_json or _json_request
    base_url = _require_api_base_url(args.api_base_url)
    bearer_token = _require_bearer_token(args.bearer_token)
    response = request_json(
        "POST",
        _api_v1_url(base_url, "/keys"),
        headers=_authorization_headers(bearer_token),
        payload={"name": args.name or ""},
    )
    data = _ensure_dict(response, context="Create API key")
    raw_key = data.get("key")
    if not isinstance(raw_key, str) or not raw_key.startswith("vmf_"):
        raise CliError("Create API key returned an invalid key")

    if args.no_save:
        return data

    try:
        _save_local_config(
            LocalConfig(
                api_base_url=base_url,
                api_key=_normalize_api_key(raw_key),
            )
        )
    except CliError as exc:
        _maybe_print_payload_before_error(
            data,
            f"API key created but failed to save config: {exc}",
        )

    return data


def _cmd_keys_list(
    args: argparse.Namespace,
    *,
    request_json: JsonRequestFn | None = None,
) -> Any:
    request_json = request_json or _json_request
    base_url = _require_api_base_url(args.api_base_url)
    bearer_token = _require_bearer_token(args.bearer_token)
    return request_json(
        "GET",
        _api_v1_url(base_url, "/keys"),
        headers=_authorization_headers(bearer_token),
    )


def _cmd_keys_revoke(
    args: argparse.Namespace,
    *,
    request_json: JsonRequestFn | None = None,
) -> Any:
    request_json = request_json or _json_request
    base_url = _require_api_base_url(args.api_base_url)
    bearer_token = _require_bearer_token(args.bearer_token)
    request_json(
        "DELETE",
        _api_v1_url(base_url, f"/keys/{_quote_path_value(args.key_id)}"),
        headers=_authorization_headers(bearer_token),
    )
    return {
        "key_id": args.key_id,
        "revoked": True,
    }


def _cmd_videos_upload(
    args: argparse.Namespace,
    *,
    request_json: JsonRequestFn | None = None,
    upload_put: UploadFn | None = None,
) -> Any:
    request_json = request_json or _json_request
    upload_put = upload_put or _stream_upload_put
    base_url = _require_api_base_url(args.api_base_url)
    api_key = _require_api_key(args.api_key)
    file_path = _ensure_file(args.file_path)
    content_type = _guess_video_content_type(file_path)

    init_payload: dict[str, Any] = {"filename": file_path.name}
    if content_type is not None:
        init_payload["content_type"] = content_type

    init_response = request_json(
        "POST",
        _api_v1_url(base_url, "/videos/upload/init"),
        headers=_authorization_headers(api_key),
        payload=init_payload,
    )
    init_data = _ensure_dict(init_response, context="Upload init")
    video_id = init_data.get("video_id")
    upload_url = init_data.get("upload_url")
    if not isinstance(video_id, str) or not video_id:
        raise CliError("Upload init response missing video_id")
    if not isinstance(upload_url, str) or not upload_url:
        raise CliError("Upload init response missing upload_url")

    upload_put(
        upload_url,
        file_path,
        content_type=content_type,
    )

    complete_response = request_json(
        "POST",
        _api_v1_url(base_url, "/videos/upload/complete"),
        headers=_authorization_headers(api_key),
        payload={
            "video_id": video_id,
            "filename": file_path.name,
        },
    )
    return complete_response


def _get_video_response(
    video_id: str,
    *,
    api_base_url: str,
    api_key: str,
    request_json: JsonRequestFn | None = None,
) -> dict[str, Any]:
    request_json = request_json or _json_request
    response = request_json(
        "GET",
        _api_v1_url(api_base_url, f"/videos/{_quote_path_value(video_id)}"),
        headers=_authorization_headers(api_key),
    )
    return _ensure_dict(response, context="Get video")


def _cmd_videos_get(
    args: argparse.Namespace,
    *,
    request_json: JsonRequestFn | None = None,
) -> Any:
    request_json = request_json or _json_request
    base_url = _require_api_base_url(args.api_base_url)
    api_key = _require_api_key(args.api_key)
    return _get_video_response(
        args.video_id,
        api_base_url=base_url,
        api_key=api_key,
        request_json=request_json,
    )


def _cmd_videos_wait(
    args: argparse.Namespace,
    *,
    request_json: JsonRequestFn | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> Any:
    request_json = request_json or _json_request
    sleep_fn = sleep_fn or time.sleep
    monotonic_fn = monotonic_fn or time.monotonic
    base_url = _require_api_base_url(args.api_base_url)
    api_key = _require_api_key(args.api_key)
    deadline = monotonic_fn() + args.timeout_seconds

    while True:
        response = _get_video_response(
            args.video_id,
            api_base_url=base_url,
            api_key=api_key,
            request_json=request_json,
        )
        status = response.get("status")
        if status == "ready":
            return response
        if status == "failed":
            _maybe_print_payload_before_error(response, "Video processing failed")
        if monotonic_fn() >= deadline:
            _maybe_print_payload_before_error(
                response,
                f"Timed out waiting for video {args.video_id}",
            )
        sleep_fn(args.interval_seconds)


def _cmd_videos_search(
    args: argparse.Namespace,
    *,
    request_json: JsonRequestFn | None = None,
) -> Any:
    request_json = request_json or _json_request
    base_url = _require_api_base_url(args.api_base_url)
    api_key = _require_api_key(args.api_key)
    return request_json(
        "POST",
        _api_v1_url(base_url, f"/videos/{_quote_path_value(args.video_id)}/search"),
        headers=_authorization_headers(api_key),
        payload={
            "query_text": args.query_text,
            "limit": args.limit,
        },
    )


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _search_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1 or parsed > 20:
        raise argparse.ArgumentTypeError("must be between 1 and 20")
    return parsed


def _add_api_base_url_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base-url", help="Override the API base URL")


def _add_api_key_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", help="Override the stored API key")


def _add_bearer_token_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bearer-token", help="Clerk bearer token for key bootstrap")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video Moment Finder CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_parser = subparsers.add_parser("auth", help="Manage local CLI auth config")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", required=True)

    auth_set = auth_subparsers.add_parser("set", help="Save API base URL and API key locally")
    auth_set.add_argument("--api-base-url", required=True)
    auth_set.add_argument("--api-key", required=True)
    auth_set.set_defaults(func=_cmd_auth_set)

    auth_status = auth_subparsers.add_parser("status", help="Show resolved auth settings")
    _add_api_base_url_arg(auth_status)
    _add_api_key_arg(auth_status)
    auth_status.set_defaults(func=_cmd_auth_status)

    auth_clear = auth_subparsers.add_parser("clear", help="Remove local CLI auth config")
    auth_clear.set_defaults(func=_cmd_auth_clear)

    keys_parser = subparsers.add_parser("keys", help="Bootstrap API keys with a Clerk token")
    keys_subparsers = keys_parser.add_subparsers(dest="keys_command", required=True)

    keys_create = keys_subparsers.add_parser("create", help="Create an API key")
    _add_api_base_url_arg(keys_create)
    _add_bearer_token_arg(keys_create)
    keys_create.add_argument("--name", default="")
    keys_create.add_argument("--no-save", action="store_true")
    keys_create.set_defaults(func=_cmd_keys_create)

    keys_list = keys_subparsers.add_parser("list", help="List API keys")
    _add_api_base_url_arg(keys_list)
    _add_bearer_token_arg(keys_list)
    keys_list.set_defaults(func=_cmd_keys_list)

    keys_revoke = keys_subparsers.add_parser("revoke", help="Revoke an API key")
    _add_api_base_url_arg(keys_revoke)
    _add_bearer_token_arg(keys_revoke)
    keys_revoke.add_argument("key_id")
    keys_revoke.set_defaults(func=_cmd_keys_revoke)

    videos_parser = subparsers.add_parser("videos", help="Upload, poll, and search videos")
    videos_subparsers = videos_parser.add_subparsers(dest="videos_command", required=True)

    videos_upload = videos_subparsers.add_parser("upload", help="Upload a video via the direct-upload API flow")
    _add_api_base_url_arg(videos_upload)
    _add_api_key_arg(videos_upload)
    videos_upload.add_argument("file_path")
    videos_upload.set_defaults(func=_cmd_videos_upload)

    videos_get = videos_subparsers.add_parser("get", help="Fetch video status")
    _add_api_base_url_arg(videos_get)
    _add_api_key_arg(videos_get)
    videos_get.add_argument("video_id")
    videos_get.set_defaults(func=_cmd_videos_get)

    videos_wait = videos_subparsers.add_parser("wait", help="Poll until a video is ready")
    _add_api_base_url_arg(videos_wait)
    _add_api_key_arg(videos_wait)
    videos_wait.add_argument("video_id")
    videos_wait.add_argument(
        "--interval-seconds",
        type=_positive_float,
        default=DEFAULT_WAIT_INTERVAL_S,
    )
    videos_wait.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=DEFAULT_WAIT_TIMEOUT_S,
    )
    videos_wait.set_defaults(func=_cmd_videos_wait)

    videos_search = videos_subparsers.add_parser("search", help="Run text search on a ready video")
    _add_api_base_url_arg(videos_search)
    _add_api_key_arg(videos_search)
    videos_search.add_argument("video_id")
    videos_search.add_argument("--query-text", required=True)
    videos_search.add_argument("--limit", type=_search_limit, default=5)
    videos_search.set_defaults(func=_cmd_videos_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        payload = args.func(args)
        if payload is not None:
            _print_json(payload)
        return 0
    except CliError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
