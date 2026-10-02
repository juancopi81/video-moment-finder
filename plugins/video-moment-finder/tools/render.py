#!/usr/bin/env python3
"""Render an offline learning guide from typed, escaped evidence data (stdlib only)."""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
from pathlib import Path
import re
from uuid import UUID


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Timestamps must be numbers")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Timestamps must be finite and nonnegative")
    return result


def clock(value: object) -> str:
    total = int(seconds(value))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


def identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value):
        raise ValueError("IDs must be lowercase letters, digits, or hyphens, starting with a letter")
    return value


def validate(data: dict) -> None:
    if data.get("format_version") != 1:
        raise ValueError("Unsupported format_version")
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        raise ValueError("A title is required")
    lecture = data["lecture"]
    if lecture.get("video_id"):
        UUID(lecture["video_id"])
    coverage = data["coverage"]
    if coverage["kind"] not in ("excerpt", "full", "synthetic"):
        raise ValueError("Invalid coverage kind")
    if seconds(coverage["end_s"]) < seconds(coverage["start_s"]):
        raise ValueError("Coverage end precedes start")
    ids = set()
    for source in data["sources"]:
        sid = identifier(source["id"])
        if sid in ids:
            raise ValueError("Duplicate source ID")
        ids.add(sid)
        if source["kind"] not in ("transcript", "frame", "synthetic"):
            raise ValueError("Invalid source kind")
        if seconds(source["end_s"]) < seconds(source["start_s"]):
            raise ValueError("Source end precedes start")
    section_ids = set()
    for section in data["sections"]:
        sid = identifier(section["id"])
        if sid in section_ids or sid in {"source-notes", "practice", "takeaways", "experiment", "main"}:
            raise ValueError("Duplicate or reserved section ID")
        section_ids.add(sid)
        if section["origin"] not in ("lecture", "generated"):
            raise ValueError("Invalid section origin")
        if section["origin"] == "lecture" and not section.get("citations"):
            raise ValueError("Lecture sections require source citations")
    for item in [*data["sections"], *data.get("questions", [])]:
        if any(sid not in ids for sid in item.get("citations", [])):
            raise ValueError("Unknown citation ID")
    if data.get("interaction") not in (None, "contrastive-matrix"):
        raise ValueError("Unknown interaction")


def source_link(data: dict, source: dict) -> str:
    label = clock(source["start_s"])
    if source["end_s"] != source["start_s"]:
        label += "–" + clock(source["end_s"])
    video_id = data["lecture"].get("video_id")
    if not video_id:
        return f'<span class="timestamp">{label}</span>'
    safe_id = str(UUID(video_id))
    url = f"https://www.videomomentfinder.com/video/{safe_id}?t={int(source['start_s'])}"
    return f'<a class="timestamp" href="{url}" target="_blank" rel="noopener noreferrer">{label}<span class="sr-only"> in the source lecture, opens in a new tab</span></a>'


def citations(data: dict, ids: list[str]) -> str:
    sources = {s["id"]: s for s in data["sources"]}
    return '<div class="citations">' + " ".join(
        f'<a href="#source-{escape(sid)}" title="View evidence note">{escape(sid)}</a> {source_link(data, sources[sid])}'
        for sid in ids
    ) + "</div>" if ids else ""


def frame_image(source: dict, root: Path) -> str:
    relative = source.get("image_path")
    if not relative:
        return '<p class="image-fallback">No image is embedded. The evidence note and timestamp remain available.</p>'
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Image paths must stay within the evidence directory")
    if not path.is_file():
        return '<p class="image-fallback">The source image is unavailable in this copy. Use the evidence note and source timestamp.</p>'
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("Frame images must be at most 5 MiB")
    raw = path.read_bytes()
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ValueError("Only PNG or JPEG frame bytes are supported")
    encoded = base64.b64encode(raw).decode("ascii")
    return f'<img src="data:{mime};base64,{encoded}" alt="{escape(source.get("alt", source["summary"]))}" loading="lazy">'


def render(data: dict, evidence_dir: Path, template: Path | None = None) -> str:
    validate(data)
    template = template or Path(__file__).resolve().parents[1] / "templates" / "study-guide.html"
    text = template.read_text(encoding="utf-8")
    nav, sections = [], []
    for index, section in enumerate(data["sections"], 1):
        sid = section["id"]
        nav.append(f'<a href="#{sid}"><span>{index:02}</span>{escape(section["title"])}</a>')
        paragraphs = "".join(f"<p>{escape(p)}</p>" for p in section.get("paragraphs", []))
        bullets = "<ul>" + "".join(f"<li>{escape(p)}</li>" for p in section["bullets"]) + "</ul>" if section.get("bullets") else ""
        origin = "From the lecture" if section["origin"] == "lecture" else "Generated explanation · check the assumptions"
        sections.append(f'<section id="{sid}" class="lesson-section"><p class="eyebrow">{index:02} / {origin}</p><h2>{escape(section["title"])}</h2>{paragraphs}{bullets}{citations(data, section.get("citations", []))}</section>')
    source_items = []
    for source in data["sources"]:
        content = f'<p>{escape(source["summary"])}</p>'
        if source["kind"] == "frame":
            content = f'<figure class="source-frame">{frame_image(source, evidence_dir)}<figcaption>{escape(source.get("caption", source["summary"]))}</figcaption></figure>'
        source_items.append(f'<article id="source-{source["id"]}" class="source"><div class="source-heading"><strong>{escape(source["id"])} · {escape(source["kind"])}</strong>{source_link(data, source)}</div>{content}</article>')
    practice = []
    for index, q in enumerate(data.get("questions", []), 1):
        practice.append(f'<article class="question"><h3>{index:02}. {escape(q["prompt"])}</h3><details><summary>Get a hint</summary><p>{escape(q.get("hint", "Return to the cited source."))}</p></details><details><summary>Check your answer</summary><p>{escape(q["answer"])}</p>{citations(data, q.get("citations", []))}</details></article>')
    cov = data["coverage"]
    coverage = f'{cov["kind"].capitalize()} · {clock(cov["start_s"])}–{clock(cov["end_s"])}'
    gaps = " ".join(cov.get("gaps", [])) or "Transcript evidence and selected visuals; this is not a review of every video frame."
    interaction = "" if data.get("interaction") != "contrastive-matrix" else '''<section id="experiment" class="lesson-section experiment"><p class="eyebrow">Generated teaching model</p><h2>See how a batch becomes pairs</h2><p>Two views per image. Green means two different views of the same original; red means different originals. The pale diagonal is self-comparison. Colors show pair identities, not measured similarities.</p><label for="batch-size">Original images: <output id="batch-count" for="batch-size">3</output></label><input id="batch-size" type="range" min="2" max="6" value="3"><p id="matrix-description" aria-live="polite"></p><div id="pair-matrix" aria-hidden="true"></div><p>Before moving the slider, predict: if the number of originals doubles, what happens to the number of cells?</p><noscript><p>For N=3, there are 6 views, 36 matrix cells, and 6 directed positive-pair cells. Doubling N quadruples the matrix size.</p></noscript></section>'''
    values = {
        "TITLE": escape(data["title"]), "SUBTITLE": escape(data.get("subtitle", "")),
        "LECTURE": escape(data["lecture"]["title"]), "COVERAGE": escape(coverage),
        "GAPS": escape(gaps), "NAV": "".join(nav), "SECTIONS": "".join(sections),
        "SOURCES": "".join(source_items), "PRACTICE": "".join(practice),
        "TAKEAWAYS": "".join(f"<li>{escape(x)}</li>" for x in data.get("takeaways", [])),
        "INTERACTION": interaction,
    }
    # Single substitution pass: source text cannot introduce a second template expansion.
    return re.sub(r"\{\{([A-Z]+)\}\}", lambda m: values[m[1]], text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    output = render(data, args.input.parent, args.template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
