#!/usr/bin/env python3
"""Build a self-contained computed playground from bounded, cited evidence."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location("vmf_guide", Path(__file__).with_name("render.py"))
guide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guide)


def validate(data: dict) -> None:
    guide.validate({**data, "sections": [], "questions": [], "interaction": None})
    p = data["playground"]
    if p["model"] != "dot-product-2d":
        raise ValueError("This renderer supports dot-product-2d; author a verified model for another topic")
    for field in ("question", "model_limit", "extension_note"):
        if not isinstance(p.get(field), str) or not p[field].strip():
            raise ValueError(f"{field} must be nonempty text")
    ids = {s["id"] for s in data["sources"]}
    if not p.get("citations") or any(s not in ids for s in p["citations"]):
        raise ValueError("The model requires known evidence citations")
    for key in ("v", "w"):
        values = p["initial"][key]
        if not isinstance(values, list) or len(values) != 2 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > 5 for x in values
        ):
            raise ValueError("Initial vector coordinates must be finite and within [-5, 5]")


def render(data: dict, evidence_dir: Path) -> str:
    validate(data)
    p = data["playground"]
    sources = []
    for s in data["sources"]:
        content = f'<p>{guide.escape(s["summary"])}</p>'
        if s["kind"] == "frame":
            content += guide.frame_image(s, evidence_dir)
        sources.append(f'<article id="source-{s["id"]}"><h3>{guide.escape(s["id"])}</h3>{guide.source_link(data, s)}{content}</article>')
    # Explicit allowlist keeps unrelated account metadata out of the output.
    client = {"v": p["initial"]["v"], "w": p["initial"]["w"], "title": data["title"], "citations": p["citations"]}
    encoded = json.dumps(client).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    cov = data["coverage"]
    values = {
        "TITLE": guide.escape(data["title"]), "QUESTION": guide.escape(p["question"]),
        "LECTURE": guide.escape(data["lecture"]["title"]),
        "COVERAGE": guide.escape(f'{cov["kind"].capitalize()} · {guide.clock(cov["start_s"])}–{guide.clock(cov["end_s"])}'),
        "GAPS": guide.escape(" ".join(cov.get("gaps", []))),
        "LIMIT": guide.escape(p["model_limit"]), "EXTENSION": guide.escape(p["extension_note"]),
        "CITATIONS": guide.citations(data, p["citations"]), "SOURCES": "".join(sources), "DATA": encoded,
        "MATH": Path(__file__).with_name("playground_math.js").read_text(),
    }
    template = Path(__file__).resolve().parents[1] / "templates/playground.html"
    return re.sub(r"\{\{([A-Z]+)\}\}", lambda m: values[m[1]], template.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ")
    output = render(json.loads(args.input.read_text()), args.input.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
