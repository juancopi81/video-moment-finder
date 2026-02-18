from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_youtube_video_id(youtube_url: str) -> str | None:
    parsed = urlparse(youtube_url.strip())
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    video_id: str | None = None

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            video_id = query.get("v", [None])[0]
        elif path_parts and path_parts[0] in {"shorts", "live"}:
            video_id = path_parts[1] if len(path_parts) > 1 else None
    elif host == "youtu.be":
        video_id = path_parts[0] if path_parts else None

    if video_id is None:
        return None
    video_id = video_id.strip()
    if not YOUTUBE_VIDEO_ID_RE.fullmatch(video_id):
        return None
    return video_id


def normalize_youtube_url(youtube_url: str) -> str | None:
    video_id = extract_youtube_video_id(youtube_url)
    if video_id is None:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"
