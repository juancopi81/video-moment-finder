"""Thin CLI for the external Video Moment Finder API."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import http.client
import io
import json
import mimetypes
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, urlsplit
from uuid import uuid4

CLI_VERSION = "0.1.0"
ENV_API_BASE_URL = "VMF_API_BASE_URL"
ENV_API_KEY = "VMF_API_KEY"
ENV_BEARER_TOKEN = "VMF_BEARER_TOKEN"
ENV_CONFIG_PATH = "VMF_CONFIG_PATH"

DEFAULT_API_TIMEOUT_S = 30.0
DEFAULT_SEARCH_TIMEOUT_S = 120.0
DEFAULT_UPLOAD_TIMEOUT_S = 300.0
DEFAULT_WAIT_INTERVAL_S = 2.0
DEFAULT_WAIT_TIMEOUT_S = 1200.0
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONFIG_DIR_NAME = "video-moment-finder"
CONFIG_FILE_NAME = "config.json"
STDIN_SENTINEL = "-"
REDACTED_FLAG_VALUES = {
    "--api-key": "<api-key>",
    "--bearer-token": "<bearer-token>",
}


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


@dataclass(frozen=True)
class UploadSource:
    file_path: Path
    filename: str
    content_type: str | None
    cleanup_path: Path | None = None


class CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Formatter that keeps examples readable and exposes defaults."""


def _parse_json_response(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("API returned a non-JSON response") from exc


def _extract_error_detail_from_bytes(body: bytes) -> str | None:
    if not body:
        return None
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        detail = _decode_text(body)
        return detail or None

    if isinstance(parsed, dict) and "detail" in parsed:
        return _format_detail(parsed["detail"])
    if isinstance(parsed, dict) and "message" in parsed:
        return _format_detail(parsed["message"])
    return _format_detail(parsed)


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
    except TimeoutError as exc:
        raise CliError(f"Request timed out after {timeout_s:g}s") from exc
    except urllib_error.HTTPError as exc:
        detail = _extract_http_error_detail(exc)
        raise CliError(f"HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise CliError(f"Request failed: {reason}") from exc

    return _parse_json_response(raw)


def _multipart_json_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    field_name: str,
    file_path: Path,
    filename: str,
    content_type: str | None,
    timeout_s: float = DEFAULT_UPLOAD_TIMEOUT_S,
) -> Any:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CliError("Request URL is invalid")

    boundary = f"vmf-{uuid4().hex}"
    preamble_lines = [
        f"--{boundary}\r\n",
        (
            "Content-Disposition: form-data; "
            f'name="{field_name}"; filename="{_escape_multipart_value(filename)}"\r\n'
        ),
    ]
    if content_type:
        preamble_lines.append(f"Content-Type: {content_type}\r\n")
    preamble_lines.append("\r\n")

    preamble = "".join(preamble_lines).encode("utf-8")
    epilogue = f"\r\n--{boundary}--\r\n".encode("utf-8")

    request_headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(preamble) + file_path.stat().st_size + len(epilogue)),
    }
    if headers:
        request_headers.update(headers)

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
        conn.putrequest(method, target)
        for header_name, header_value in request_headers.items():
            conn.putheader(header_name, header_value)
        conn.endheaders()
        conn.send(preamble)

        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                conn.send(chunk)

        conn.send(epilogue)
        response = conn.getresponse()
        raw = response.read()
    except TimeoutError as exc:
        raise CliError(f"Request timed out after {timeout_s:g}s") from exc
    except OSError as exc:
        raise CliError(f"Request failed: {exc}") from exc
    finally:
        conn.close()

    if 200 <= response.status < 300:
        return _parse_json_response(raw)

    detail = _extract_error_detail_from_bytes(raw) or response.reason or "Request failed"
    raise CliError(f"HTTP {response.status}: {detail}")


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
    detail = _extract_error_detail_from_bytes(body)
    if detail:
        return detail
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


def _normalize_non_empty_value(value: str, *, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise CliError(f"{label} must be non-empty", exit_code=2)
    return cleaned


def _resolve_api_base_url(
    cli_value: str | None,
    *,
    config: LocalConfig | None = None,
) -> ResolvedValue:
    if cli_value:
        return ResolvedValue(_normalize_api_base_url(cli_value), "flag")

    env_value = os.environ.get(ENV_API_BASE_URL, "").strip()
    if env_value:
        return ResolvedValue(_normalize_api_base_url(env_value), "env")

    config_value = (config or _load_local_config()).api_base_url
    if config_value:
        return ResolvedValue(_normalize_api_base_url(config_value), "config")

    return ResolvedValue(None, "missing")


def _resolve_api_key(
    cli_value: str | None,
    *,
    config: LocalConfig | None = None,
) -> ResolvedValue:
    if cli_value:
        return ResolvedValue(_normalize_api_key(cli_value), "flag")

    env_value = os.environ.get(ENV_API_KEY, "").strip()
    if env_value:
        return ResolvedValue(_normalize_api_key(env_value), "env")

    config_value = (config or _load_local_config()).api_key
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


def _resolve_api_credentials(
    cli_base_url: str | None,
    cli_api_key: str | None,
) -> tuple[ResolvedValue, ResolvedValue]:
    config = _load_local_config()
    return (
        _resolve_api_base_url(cli_base_url, config=config),
        _resolve_api_key(cli_api_key, config=config),
    )


def _require_api_base_url(cli_value: str | None) -> str:
    resolved = _resolve_api_base_url(cli_value)
    if resolved.value is None:
        raise CliError(
            f"Missing API base URL. Provide --api-base-url, {ENV_API_BASE_URL}, or run `vmf auth set`.",
            exit_code=2,
        )
    return resolved.value


def _require_api_credentials(
    cli_base_url: str | None,
    cli_api_key: str | None,
) -> tuple[str, str]:
    base_url, api_key = _resolve_api_credentials(cli_base_url, cli_api_key)
    if base_url.value is None:
        raise CliError(
            f"Missing API base URL. Provide --api-base-url, {ENV_API_BASE_URL}, or run `vmf auth set`.",
            exit_code=2,
        )
    if api_key.value is None:
        raise CliError(
            f"Missing API key. Provide --api-key, {ENV_API_KEY}, or run `vmf auth set`.",
            exit_code=2,
        )
    return base_url.value, api_key.value


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


def _stderr(message: str = "", *, end: str = "\n") -> None:
    print(message, file=sys.stderr, end=end, flush=True)


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


def _escape_multipart_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _stdin_is_tty() -> bool:
    isatty = getattr(sys.stdin, "isatty", None)
    return bool(isatty and isatty())


def _stderr_is_tty() -> bool:
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty and isatty())


def _is_interactive_session() -> bool:
    return _stdin_is_tty() and _stderr_is_tty()


def _build_invocation(argv: list[str] | None) -> list[str]:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    return ["vmf", *effective_argv]


def _redact_invocation(invocation: list[str]) -> list[str]:
    redacted: list[str] = []
    index = 0
    while index < len(invocation):
        token = invocation[index]
        replacement = REDACTED_FLAG_VALUES.get(token)
        if replacement is not None:
            redacted.append(token)
            if index + 1 < len(invocation):
                redacted.append(replacement)
                index += 2
                continue
            index += 1
            continue
        for flag, placeholder in REDACTED_FLAG_VALUES.items():
            prefix = f"{flag}="
            if token.startswith(prefix):
                redacted.append(f"{prefix}{placeholder}")
                break
        else:
            redacted.append(token)
        index += 1
    return redacted


def _with_yes_flag(invocation: list[str]) -> list[str]:
    if "--yes" in invocation:
        return invocation
    return [*invocation, "--yes"]


def _render_rerun_command(invocation: list[str]) -> str:
    return shlex.join(_redact_invocation(_with_yes_flag(invocation)))


def _confirm_destructive_action(
    *,
    prompt: str,
    action: str,
    dry_run_payload: dict[str, Any],
    yes: bool,
    dry_run: bool,
    invocation: list[str],
) -> dict[str, Any] | None:
    if dry_run:
        return dry_run_payload
    if yes:
        return None
    if not _is_interactive_session():
        raise CliError(
            (
                f"Refusing to {action} in non-interactive mode without --yes.\n"
                f"Re-run with: { _render_rerun_command(invocation) }"
            ),
            exit_code=2,
        )

    _stderr(f"{prompt} [y/N]: ", end="")
    response = sys.stdin.readline()
    if response.strip().lower() not in {"y", "yes"}:
        raise CliError("Cancelled", exit_code=1)
    return None


def _stdin_buffer() -> io.BufferedReader | io.BytesIO:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is not None:
        return buffer
    return io.BytesIO(sys.stdin.read().encode("utf-8"))


def _read_stdin_to_temp_file() -> Path:
    if _stdin_is_tty():
        raise CliError(
            "Reading upload bytes from stdin requires piped or redirected input.",
            exit_code=2,
        )

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        total_bytes = 0
        stdin_buffer = _stdin_buffer()
        while True:
            chunk = stdin_buffer.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            temp_file.write(chunk)
            total_bytes += len(chunk)

    if total_bytes == 0:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise CliError("No upload bytes received on stdin", exit_code=2)

    return temp_path


def _resolve_upload_source(
    file_path_value: str,
    *,
    filename_override: str | None,
    content_type_override: str | None,
) -> UploadSource:
    content_type = content_type_override.strip() if content_type_override else None
    filename = filename_override.strip() if filename_override else None

    if file_path_value == STDIN_SENTINEL:
        if not filename:
            raise CliError(
                "stdin uploads require --filename so the API can name the uploaded file.",
                exit_code=2,
            )
        if not content_type:
            raise CliError(
                "stdin uploads require --content-type, for example --content-type video/mp4.",
                exit_code=2,
            )
        temp_path = _read_stdin_to_temp_file()
        return UploadSource(
            file_path=temp_path,
            filename=filename,
            content_type=content_type,
            cleanup_path=temp_path,
        )

    path = _ensure_file(file_path_value)
    return UploadSource(
        file_path=path,
        filename=filename or path.name,
        content_type=content_type or _guess_video_content_type(path),
        cleanup_path=None,
    )


def _cleanup_upload_source(source: UploadSource) -> None:
    if source.cleanup_path is None:
        return
    try:
        source.cleanup_path.unlink()
    except OSError:
        pass


def _resolve_query_text(raw_value: str) -> str:
    if raw_value != STDIN_SENTINEL:
        query_text = raw_value.strip()
        if not query_text:
            raise CliError("query_text must be non-empty", exit_code=2)
        return query_text

    if _stdin_is_tty():
        raise CliError(
            "Reading query text from stdin requires piped or redirected input.",
            exit_code=2,
        )

    query_text = sys.stdin.read().strip()
    if not query_text:
        raise CliError("No query text received on stdin", exit_code=2)
    return query_text


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
    base_url, api_key = _resolve_api_credentials(args.api_base_url, args.api_key)
    return {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "api_base_url": base_url.value,
        "api_base_url_source": base_url.source,
        "api_key_configured": api_key.value is not None,
        "api_key_masked": _mask_api_key(api_key.value),
        "api_key_source": api_key.source,
    }


def _cmd_auth_clear(args: argparse.Namespace) -> Any:
    path = _config_path()
    preview = _confirm_destructive_action(
        prompt=f"Clear local CLI auth config at {path}?",
        action="clear local CLI auth config",
        dry_run_payload={
            "dry_run": True,
            "action": "auth_clear",
            "config_path": str(path),
            "config_exists": path.exists(),
        },
        yes=args.yes,
        dry_run=args.dry_run,
        invocation=args.invocation,
    )
    if preview is not None:
        return preview

    path = _clear_local_config()
    return {
        "cleared": True,
        "config_path": str(path),
    }


def _cmd_keys_create(args: argparse.Namespace) -> Any:
    base_url = _require_api_base_url(args.api_base_url)
    bearer_token = _require_bearer_token(args.bearer_token)
    response = _json_request(
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
        _print_json(data)
        raise CliError(f"API key created but failed to save config: {exc}") from exc

    return data


def _cmd_keys_list(args: argparse.Namespace) -> Any:
    base_url = _require_api_base_url(args.api_base_url)
    bearer_token = _require_bearer_token(args.bearer_token)
    return _json_request(
        "GET",
        _api_v1_url(base_url, "/keys"),
        headers=_authorization_headers(bearer_token),
    )


def _cmd_keys_revoke(args: argparse.Namespace) -> Any:
    resolved_base_url = _resolve_api_base_url(args.api_base_url)
    preview = _confirm_destructive_action(
        prompt=f"Revoke API key {args.key_id}?",
        action="revoke an API key",
        dry_run_payload={
            "dry_run": True,
            "action": "keys_revoke",
            "key_id": args.key_id,
            "api_base_url": resolved_base_url.value,
            "api_base_url_source": resolved_base_url.source,
        },
        yes=args.yes,
        dry_run=args.dry_run,
        invocation=args.invocation,
    )
    if preview is not None:
        return preview

    base_url = _require_api_base_url(args.api_base_url)
    bearer_token = _require_bearer_token(args.bearer_token)
    _json_request(
        "DELETE",
        _api_v1_url(base_url, f"/keys/{_quote_path_value(args.key_id)}"),
        headers=_authorization_headers(bearer_token),
    )
    return {
        "key_id": args.key_id,
        "revoked": True,
    }


def _cmd_videos_upload(args: argparse.Namespace) -> Any:
    base_url, api_key = _require_api_credentials(args.api_base_url, args.api_key)
    source = _resolve_upload_source(
        args.file_path,
        filename_override=args.filename,
        content_type_override=args.content_type,
    )
    headers = _authorization_headers(api_key)
    if args.idempotency_key is not None:
        headers["Idempotency-Key"] = _normalize_non_empty_value(
            args.idempotency_key,
            label="idempotency_key",
        )

    try:
        return _multipart_json_request(
            "POST",
            _api_v1_url(base_url, "/videos/upload"),
            headers=headers,
            field_name="file",
            file_path=source.file_path,
            filename=source.filename,
            content_type=source.content_type,
        )
    finally:
        _cleanup_upload_source(source)


def _cmd_videos_list(args: argparse.Namespace) -> Any:
    base_url, api_key = _require_api_credentials(args.api_base_url, args.api_key)
    return _json_request(
        "GET",
        _api_v1_url(base_url, "/videos"),
        headers=_authorization_headers(api_key),
    )


def _get_video_response(
    video_id: str,
    *,
    api_base_url: str,
    api_key: str,
) -> dict[str, Any]:
    response = _json_request(
        "GET",
        _api_v1_url(api_base_url, f"/videos/{_quote_path_value(video_id)}"),
        headers=_authorization_headers(api_key),
    )
    return _ensure_dict(response, context="Get video")


def _cmd_videos_get(args: argparse.Namespace) -> Any:
    base_url, api_key = _require_api_credentials(args.api_base_url, args.api_key)
    return _get_video_response(
        args.video_id,
        api_base_url=base_url,
        api_key=api_key,
    )


def _cmd_videos_wait(args: argparse.Namespace) -> Any:
    base_url, api_key = _require_api_credentials(args.api_base_url, args.api_key)
    deadline = time.monotonic() + args.timeout_seconds

    while True:
        response = _get_video_response(
            args.video_id,
            api_base_url=base_url,
            api_key=api_key,
        )
        status = response.get("status")
        if status == "ready":
            return response
        if status == "failed":
            _print_json(response)
            error_message = response.get("error_message")
            detail = str(error_message).strip() if error_message else "unknown error"
            raise CliError(f"Video processing failed: {detail}")
        if time.monotonic() >= deadline:
            _print_json(response)
            raise CliError(f"Timed out waiting for video {args.video_id} (last status: {status})")
        time.sleep(args.interval_seconds)


def _cmd_videos_search(args: argparse.Namespace) -> Any:
    base_url, api_key = _require_api_credentials(args.api_base_url, args.api_key)
    return _json_request(
        "POST",
        _api_v1_url(base_url, f"/videos/{_quote_path_value(args.video_id)}/search"),
        headers=_authorization_headers(api_key),
        payload={
            "query_text": _resolve_query_text(args.query_text),
            "limit": args.limit,
        },
        timeout_s=DEFAULT_SEARCH_TIMEOUT_S,
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


def _examples(*lines: str) -> str:
    rendered = ["Examples:"]
    rendered.extend(f"  {line}" for line in lines)
    return "\n".join(rendered)


def _add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help_text: str,
    description: str,
    epilog: str,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        name,
        help=help_text,
        description=description,
        epilog=epilog,
        formatter_class=CliHelpFormatter,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmf",
        description="Agent-friendly CLI for the Video Moment Finder /api/v1 happy path.",
        epilog="Run `vmf <command> --help` for examples and next steps.",
        formatter_class=CliHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {CLI_VERSION}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    auth_parser = _add_parser(
        subparsers,
        "auth",
        help_text="Manage local CLI auth config",
        description="Save, inspect, or remove the local API base URL and API key.",
        epilog=_examples(
            "vmf auth set --api-base-url https://api.videomomentfinder.com --api-key vmf_xxx",
            "vmf auth status",
            "vmf auth clear --dry-run",
        ),
    )
    auth_subparsers = auth_parser.add_subparsers(
        dest="auth_command",
        metavar="subcommand",
        required=True,
    )

    auth_set = _add_parser(
        auth_subparsers,
        "set",
        help_text="Save API base URL and API key locally",
        description="Write the API base URL and vmf_ API key to the local CLI config file.",
        epilog=_examples(
            "vmf auth set --api-base-url https://api.videomomentfinder.com --api-key vmf_xxx",
            "vmf auth set --api-base-url http://localhost:8000 --api-key vmf_local_key",
        ),
    )
    auth_set.add_argument(
        "--api-base-url",
        required=True,
        help="API base URL, for example https://api.videomomentfinder.com",
    )
    auth_set.add_argument(
        "--api-key",
        required=True,
        help="Video Moment Finder API key starting with vmf_",
    )
    auth_set.set_defaults(func=_cmd_auth_set)

    auth_status = _add_parser(
        auth_subparsers,
        "status",
        help_text="Show resolved auth settings",
        description="Show the resolved API base URL and API key source using flag > env > config precedence.",
        epilog=_examples(
            "vmf auth status",
            "vmf auth status --api-base-url https://api.videomomentfinder.com",
        ),
    )
    auth_status.add_argument(
        "--api-base-url",
        help="Override the resolved API base URL for this status check",
    )
    auth_status.add_argument(
        "--api-key",
        help="Override the resolved API key for this status check",
    )
    auth_status.set_defaults(func=_cmd_auth_status)

    auth_clear = _add_parser(
        auth_subparsers,
        "clear",
        help_text="Remove local CLI auth config",
        description="Delete the local CLI config file that stores the API base URL and API key.",
        epilog=_examples(
            "vmf auth clear --dry-run",
            "vmf auth clear --yes",
        ),
    )
    auth_clear.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the config file that would be removed without mutating anything",
    )
    auth_clear.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    auth_clear.set_defaults(func=_cmd_auth_clear)

    keys_parser = _add_parser(
        subparsers,
        "keys",
        help_text="Bootstrap API keys with a Clerk token",
        description="Create, list, or revoke API keys using a temporary Clerk bearer token.",
        epilog=_examples(
            "vmf keys create --name agent",
            "vmf keys list --bearer-token <clerk-token>",
            "vmf keys revoke key_123 --dry-run",
        ),
    )
    keys_subparsers = keys_parser.add_subparsers(
        dest="keys_command",
        metavar="subcommand",
        required=True,
    )

    keys_create = _add_parser(
        keys_subparsers,
        "create",
        help_text="Create an API key",
        description="Create a new API key and optionally save it to the local CLI config file.",
        epilog=_examples(
            "vmf keys create --name agent",
            "vmf keys create --bearer-token <clerk-token> --no-save",
        ),
    )
    keys_create.add_argument(
        "--api-base-url",
        help="Override the API base URL used to create the key",
    )
    keys_create.add_argument(
        "--bearer-token",
        help="Temporary Clerk bearer token used for key management",
    )
    keys_create.add_argument(
        "--name",
        default="",
        help="Optional display name for the created API key",
    )
    keys_create.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the returned API key to the local CLI config file",
    )
    keys_create.set_defaults(func=_cmd_keys_create)

    keys_list = _add_parser(
        keys_subparsers,
        "list",
        help_text="List API keys",
        description="List active API keys for the current account.",
        epilog=_examples(
            "vmf keys list --bearer-token <clerk-token>",
            "VMF_BEARER_TOKEN=<clerk-token> vmf keys list",
        ),
    )
    keys_list.add_argument(
        "--api-base-url",
        help="Override the API base URL used to list keys",
    )
    keys_list.add_argument(
        "--bearer-token",
        help="Temporary Clerk bearer token used for key management",
    )
    keys_list.set_defaults(func=_cmd_keys_list)

    keys_revoke = _add_parser(
        keys_subparsers,
        "revoke",
        help_text="Revoke an API key",
        description="Revoke one API key. In non-interactive mode this requires --yes unless you use --dry-run.",
        epilog=_examples(
            "vmf keys revoke key_123 --dry-run",
            "vmf keys revoke key_123 --yes --bearer-token <clerk-token>",
        ),
    )
    keys_revoke.add_argument(
        "--api-base-url",
        help="Override the API base URL used to revoke the key",
    )
    keys_revoke.add_argument(
        "--bearer-token",
        help="Temporary Clerk bearer token used for key management",
    )
    keys_revoke.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the revoke action without sending the DELETE request",
    )
    keys_revoke.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    keys_revoke.add_argument(
        "key_id",
        help="API key identifier to revoke",
    )
    keys_revoke.set_defaults(func=_cmd_keys_revoke)

    videos_parser = _add_parser(
        subparsers,
        "videos",
        help_text="Upload, list, poll, and search videos",
        description="Upload videos, list uploaded videos, poll processing status, and run text search against ready videos.",
        epilog=_examples(
            "vmf videos upload ./sample.mp4",
            "vmf videos list",
            "vmf videos wait 11111111-1111-4111-8111-111111111111",
            "vmf videos search 11111111-1111-4111-8111-111111111111 --query-text \"explain the model\"",
        ),
    )
    videos_subparsers = videos_parser.add_subparsers(
        dest="videos_command",
        metavar="subcommand",
        required=True,
    )

    videos_upload = _add_parser(
        videos_subparsers,
        "upload",
        help_text="Upload a video with the one-shot multipart API",
        description=(
            "Upload one video with the /api/v1/videos/upload endpoint. "
            "Pass a file path or '-' to read the raw video bytes from stdin."
        ),
        epilog=_examples(
            "vmf videos upload ./sample.mp4",
            "vmf videos upload ./sample.mp4 --idempotency-key sample-v1",
            "cat sample.mp4 | vmf videos upload - --filename sample.mp4 --content-type video/mp4",
        ),
    )
    videos_upload.add_argument(
        "--api-base-url",
        help="Override the API base URL used for upload",
    )
    videos_upload.add_argument(
        "--api-key",
        help="Override the stored API key used for upload",
    )
    videos_upload.add_argument(
        "--filename",
        help="Override the uploaded filename; required when file_path is '-'",
    )
    videos_upload.add_argument(
        "--content-type",
        help="Override the uploaded content type; required when file_path is '-'",
    )
    videos_upload.add_argument(
        "--idempotency-key",
        help="Retry-safe idempotency key forwarded as the Idempotency-Key header",
    )
    videos_upload.add_argument(
        "file_path",
        help="Local video file path or '-' to read raw bytes from stdin",
    )
    videos_upload.set_defaults(func=_cmd_videos_upload)

    videos_list = _add_parser(
        videos_subparsers,
        "list",
        help_text="List videos",
        description="List videos owned by the authenticated user.",
        epilog=_examples(
            "vmf videos list",
            "VMF_API_KEY=vmf_xxx vmf videos list",
        ),
    )
    videos_list.add_argument(
        "--api-base-url",
        help="Override the API base URL used to list videos",
    )
    videos_list.add_argument(
        "--api-key",
        help="Override the stored API key used to list videos",
    )
    videos_list.set_defaults(func=_cmd_videos_list)

    videos_get = _add_parser(
        videos_subparsers,
        "get",
        help_text="Fetch video status",
        description="Fetch the current processing state for one video.",
        epilog=_examples(
            "vmf videos get 11111111-1111-4111-8111-111111111111",
            "VMF_API_KEY=vmf_xxx vmf videos get 11111111-1111-4111-8111-111111111111",
        ),
    )
    videos_get.add_argument(
        "--api-base-url",
        help="Override the API base URL used to fetch video status",
    )
    videos_get.add_argument(
        "--api-key",
        help="Override the stored API key used to fetch video status",
    )
    videos_get.add_argument(
        "video_id",
        help="Video identifier returned by upload or submit commands",
    )
    videos_get.set_defaults(func=_cmd_videos_get)

    videos_wait = _add_parser(
        videos_subparsers,
        "wait",
        help_text="Poll until a video is ready",
        description="Poll a video until it reaches ready or failed status.",
        epilog=_examples(
            "vmf videos wait 11111111-1111-4111-8111-111111111111",
            "vmf videos wait 11111111-1111-4111-8111-111111111111 --interval-seconds 5 --timeout-seconds 600",
        ),
    )
    videos_wait.add_argument(
        "--api-base-url",
        help="Override the API base URL used to poll video status",
    )
    videos_wait.add_argument(
        "--api-key",
        help="Override the stored API key used to poll video status",
    )
    videos_wait.add_argument(
        "video_id",
        help="Video identifier returned by upload or submit commands",
    )
    videos_wait.add_argument(
        "--interval-seconds",
        type=_positive_float,
        default=DEFAULT_WAIT_INTERVAL_S,
        help="Seconds to sleep between status requests",
    )
    videos_wait.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=DEFAULT_WAIT_TIMEOUT_S,
        help="Maximum total seconds to wait before timing out",
    )
    videos_wait.set_defaults(func=_cmd_videos_wait)

    videos_search = _add_parser(
        videos_subparsers,
        "search",
        help_text="Run text search on a ready video",
        description=(
            "Run text search on a ready video. "
            "Use --query-text - to read the full query from stdin."
        ),
        epilog=_examples(
            "vmf videos search 11111111-1111-4111-8111-111111111111 --query-text \"explain the model\" --limit 3",
            "printf 'explain the model' | vmf videos search 11111111-1111-4111-8111-111111111111 --query-text -",
        ),
    )
    videos_search.add_argument(
        "--api-base-url",
        help="Override the API base URL used for search",
    )
    videos_search.add_argument(
        "--api-key",
        help="Override the stored API key used for search",
    )
    videos_search.add_argument(
        "video_id",
        help="Video identifier returned by upload or submit commands",
    )
    videos_search.add_argument(
        "--query-text",
        required=True,
        help="Search query text, or '-' to read the full query from stdin",
    )
    videos_search.add_argument(
        "--limit",
        type=_search_limit,
        default=5,
        help="Maximum number of search results to return",
    )
    videos_search.set_defaults(func=_cmd_videos_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.invocation = _build_invocation(argv)

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
