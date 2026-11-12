"""Keep backend diagnostics and validation inputs out of MCP error responses."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from src.api.frames import FrameRequestValidationError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PublicToolError(ValueError):
    """A fixed, user-facing message authored at the MCP boundary, never backend text."""


def _error_message(error: Exception) -> str:
    # FastMCP wraps exceptions in ToolError. Inspect types in its cause chain,
    # never parse or forward the wrapped message (which can contain credentials).
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, PublicToolError):
            return str(cause)
        if isinstance(cause, FrameRequestValidationError):
            return (
                "Invalid frame request. Use at most 8 timestamps for high resolution "
                "or 25 timestamps for thumbnails, all finite and non-negative."
            )
        if isinstance(cause, ValidationError):
            return "Invalid tool arguments. Check the required fields and allowed values."
        if isinstance(cause, HTTPException):
            if cause.status_code == 400 and cause.detail in (
                "Video not ready (status: queued)", "Video not ready (status: processing)",
                "Video not ready (status: failed)",
            ):
                return "Video not ready. Check its processing status before retrieving evidence."
            return {
                400: "Invalid request. Check the input and the tool's limits.",
                401: "Authentication expired. Reconnect Video Moment Finder and retry.",
                402: "Insufficient API units. Reuse existing evidence or choose an operation within your allowance.",
                403: "This operation is not permitted for the connected account.",
                404: "Video not found in the connected account. Refresh the library.",
                409: "Video not ready or upload conflicts with an existing video. Check its status before retrying.",
                413: "The upload exceeds the supported size limit.",
                422: "Invalid tool arguments. Check the required fields and allowed values.",
                429: "Rate limit exceeded. Wait before trying again.",
            }.get(cause.status_code, "VMF is temporarily unavailable. Check video status and allowance before retrying.")
        cause = cause.__cause__
    return "VMF could not complete this request. Check video status and allowance before retrying."


class PrivateErrorsFastMCP(FastMCP):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return await super().call_tool(name, arguments)
        except Exception as error:
            # Do not log arguments, exception text, account IDs or signed URLs.
            logger.warning("MCP tool call failed (%s)", type(error).__name__)
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=_error_message(error))],
            )
