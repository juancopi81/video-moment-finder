#!/usr/bin/env python3
"""Render an offline flashcard deck and a spreadsheet-safe UTF-8 CSV (stdlib only)."""
from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import unicodedata
from uuid import UUID


# The guide and cards share timestamp, image-path, and escaping rules. Load by
# file location so this also works when an installed package is outside sys.path.
_SPEC = importlib.util.spec_from_file_location("vmf_learning_render", Path(__file__).with_name("render.py"))
shared = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shared)

CSV_FIELDS = ("Front", "Back", "Tags")
KINDS = ("recall", "concept", "application")


def require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if any(unicodedata.category(c) == "Cc" and c not in "\t\r\n" for c in value):
        raise ValueError(f"{name} contains unsupported control characters")
    return value


def validate(data: dict) -> None:
    if data.get("artifact") != "flashcards" or type(data.get("format_version")) is not int:
        raise ValueError("Expected a versioned flashcards artifact")
    shared.validate({**data, "sections": [], "questions": [], "interaction": None})
    require_text(data["lecture"]["title"], "lecture.title")
    for source in data["sources"]:
        require_text(source["summary"], "source.summary")
    cards = data.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError("At least one card is required")
    sources = {source["id"]: source for source in data["sources"]}
    ids, questions = set(), set()
    for card in cards:
        cid = shared.identifier(card["id"])
        if cid in ids:
            raise ValueError("Duplicate card ID")
        ids.add(cid)
        front = require_text(card["front"], "card.front")
        question = " ".join(front.split()).casefold()
        if question in questions:
            raise ValueError("Duplicate question; consolidate redundant cards")
        questions.add(question)
        require_text(card["back"], "card.back")
        if card["kind"] not in KINDS:
            raise ValueError("Invalid card kind")
        if card["origin"] not in ("lecture", "generated"):
            raise ValueError("Invalid card origin")
        refs = card.get("citations")
        if not isinstance(refs, list) or not refs or any(sid not in sources for sid in refs):
            raise ValueError("Every card requires known evidence citations")
        if len(refs) != len(set(refs)):
            raise ValueError("Duplicate card citation")
        if card["origin"] == "lecture" and all(sources[sid]["kind"] == "synthetic" for sid in refs):
            raise ValueError("Synthetic evidence cannot substantiate a lecture claim")
        if card["origin"] == "generated":
            require_text(card.get("rationale"), "generated card rationale")
        if card.get("hint") is not None:
            require_text(card["hint"], "card.hint")
        tags = card.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tag) for tag in tags):
            raise ValueError("Tags must be lowercase words joined by hyphens or underscores")


def scope_label(data: dict) -> str:
    cov = data["coverage"]
    return f'{cov["kind"].capitalize()} · {shared.clock(cov["start_s"])}–{shared.clock(cov["end_s"])}'


def origin_label(card: dict) -> str:
    return "Answer grounded in lecture evidence" if card["origin"] == "lecture" else "Generated practice · see the reasoning"


def spreadsheet_text(value: str) -> str:
    """Guard formula-leading cells, including whitespace/control obfuscation.

    CSV quoting alone does not stop spreadsheet formulas. The safety apostrophe
    can be visible in plain-text importers; exact original text remains in JSON.
    """
    significant = value
    while significant and (significant[0].isspace() or unicodedata.category(significant[0]) in ("Cc", "Cf")):
        significant = significant[1:]
    unsafe = value.startswith(("\t", "\r", "\n")) or significant.startswith(("=", "+", "-", "@"))
    return "'" + value if unsafe else value


def source_references(data: dict, card: dict) -> str:
    sources = {s["id"]: s for s in data["sources"]}
    refs = []
    for sid in card["citations"]:
        source = sources[sid]
        label = shared.clock(source["start_s"])
        if source["end_s"] != source["start_s"]:
            label += "–" + shared.clock(source["end_s"])
        label = f"{sid} [{source['kind']}] {label}"
        if data["lecture"].get("video_id"):
            video_id = str(UUID(data["lecture"]["video_id"]))
            label += f" https://www.videomomentfinder.com/video/{video_id}?t={int(source['start_s'])}"
        refs.append(label)
    return "\n".join(refs)


def export_csv(data: dict) -> str:
    validate(data)
    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(CSV_FIELDS)
    for card in data["cards"]:
        back = card["back"] + "\n\n" + origin_label(card)
        if card["origin"] == "generated":
            back += "\nReasoning: " + card["rationale"]
        back += f'\n{data["lecture"]["title"]}\n{scope_label(data)}\nSources:\n{source_references(data, card)}'
        tags = " ".join(dict.fromkeys(["vmf", card["kind"], *card.get("tags", [])]))
        writer.writerow([spreadsheet_text(value) for value in (card["front"], back, tags)])
    return output.getvalue()


def render(data: dict, evidence_dir: Path, template: Path | None = None) -> str:
    validate(data)
    template = template or Path(__file__).resolve().parents[1] / "templates" / "flashcards.html"
    text = template.read_text(encoding="utf-8")
    esc = shared.escape
    cards = []
    for index, card in enumerate(data["cards"], 1):
        hint = f'<details class="hint"><summary>Get a hint</summary><p>{esc(card["hint"])}</p></details>' if card.get("hint") else ""
        rationale = f'<p class="reasoning"><strong>Generated reasoning.</strong> {esc(card["rationale"])}</p>' if card["origin"] == "generated" else ""
        cards.append(f'<article class="flashcard" id="card-{card["id"]}" aria-labelledby="question-{card["id"]}"><p class="eyebrow">{index:02} / {esc(card["kind"])}</p><h2 id="question-{card["id"]}" tabindex="-1">{esc(card["front"])}</h2>{hint}<details class="answer"><summary>Reveal answer</summary><p class="answer-text">{esc(card["back"])}</p><p class="origin">{esc(origin_label(card))}</p>{rationale}{shared.citations(data, card["citations"])}</details></article>')
    sources = []
    for source in data["sources"]:
        content = f'<p>{esc(source["summary"])}</p>'
        if source["kind"] == "frame":
            content += f'<figure>{shared.frame_image(source, evidence_dir)}<figcaption>{esc(source.get("caption", source["summary"]))}</figcaption></figure>'
        sources.append(f'<article class="source" id="source-{source["id"]}"><h3>{esc(source["id"])} · {esc(source["kind"])}</h3>{shared.source_link(data, source)}{content}</article>')
    csv_url = "data:text/csv;charset=utf-8;base64," + base64.b64encode(export_csv(data).encode("utf-8")).decode("ascii")
    values = {
        "TITLE": esc(data["title"]), "SUBTITLE": esc(data.get("subtitle", "Practice a little. Revisit the evidence.")),
        "LECTURE": esc(data["lecture"]["title"]), "COVERAGE": esc(scope_label(data)),
        "COUNT": str(len(data["cards"])), "CARDS": "".join(cards), "SOURCES": "".join(sources),
        "GAPS": esc(" ".join(data["coverage"].get("gaps", [])) or "Only the cited evidence was checked; visual coverage may be incomplete."),
        "CSVURL": csv_url,
    }
    # A single pass prevents escaped source text from becoming template syntax.
    return re.sub(r"\{\{([A-Z]+)\}\}", lambda match: values[match[1]], text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="Offline review HTML output")
    parser.add_argument("--csv", type=Path, help="CSV output (defaults to the HTML path with .csv suffix)")
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    csv_path = args.csv or args.output.with_suffix(".csv")
    if args.output.resolve() in (args.input.resolve(), csv_path.resolve()) or csv_path.resolve() == args.input.resolve():
        parser.error("Input, HTML, and CSV paths must be distinct")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    html_output, csv_output = render(data, args.input.parent, args.template), export_csv(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_output, encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        output.write(csv_output)


if __name__ == "__main__":
    main()
