#!/usr/bin/env python3
"""Render an offline, finite assumption lab from cited, reviewed scenario data."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import re


_spec = importlib.util.spec_from_file_location("vmf_guide_renderer", Path(__file__).with_name("render.py"))
_guide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guide)
escape = _guide.escape


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")


def _strings(value: object, field: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{field} must be a list of text")
    for item in value:
        _text(item, field)


def validate(data: dict) -> None:
    """Require a complete, bounded truth table; never evaluate source expressions."""
    _guide.validate({**data, "sections": [], "questions": [], "interaction": None})
    _text(data["lecture"]["title"], "lecture.title")
    _strings(data["coverage"].get("gaps", []), "coverage.gaps")
    source_ids = {s["id"] for s in data["sources"]}
    for source in data["sources"]:
        _text(source["summary"], "source.summary")
    lab = data["lab"]
    for field in ("question", "premise", "model_limit"):
        _text(lab[field], field)
    controls = lab["controls"]
    if not isinstance(controls, list) or not 1 <= len(controls) <= 3:
        raise ValueError("Use one to three controls")
    choices = {}
    for control in controls:
        cid = _guide.identifier(control["id"])
        if cid in choices:
            raise ValueError("Duplicate control ID")
        _text(control["label"], "control.label")
        options = control["options"]
        if not isinstance(options, list) or not 2 <= len(options) <= 3:
            raise ValueError("Each control needs two or three options")
        option_ids = []
        for option in options:
            oid = _guide.identifier(option["id"])
            if oid in option_ids:
                raise ValueError("Duplicate option ID")
            _text(option["label"], "option.label")
            option_ids.append(oid)
        choices[cid] = option_ids
    expected = set(itertools.product(*choices.values()))
    states = lab["states"]
    if not isinstance(states, list) or len(states) != len(expected):
        raise ValueError("Provide exactly one state for every control combination")
    seen, state_ids = set(), set()
    for state in states:
        sid = _guide.identifier(state["id"])
        if sid in state_ids:
            raise ValueError("Duplicate state ID")
        state_ids.add(sid)
        if not isinstance(state["when"], dict) or set(state["when"]) != set(choices):
            raise ValueError("State selections must name every control")
        key = tuple(state["when"][cid] for cid in choices)
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate control combination")
        seen.add(key)
        for field in ("title", "outcome", "explanation", "boundary_note", "next_step"):
            _text(state[field], field)
        for field in ("assumptions", "reasoning", "unknowns", "entities", "citations"):
            _strings(state[field], field, nonempty=True)
        if len(state["entities"]) != 2:
            raise ValueError("Each state needs two comparison entities")
        if state["boundary"] not in ("applies", "caution", "not-supported"):
            raise ValueError("Unknown boundary status")
        if any(sid not in source_ids for sid in state["citations"]):
            raise ValueError("Unknown citation ID")
    if seen != expected:
        raise ValueError("Incomplete control combinations")
    if lab["baseline"] not in state_ids:
        raise ValueError("Unknown baseline state")


def _list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def _state(data: dict, state: dict) -> str:
    boundary = {"applies": "Within the stated model", "caution": "A task assumption needs care", "not-supported": "The source does not settle this"}[state["boundary"]]
    entities = "".join(f'<div class="entity">{escape(entity)}</div>' for entity in state["entities"])
    return (
        f'<article class="scenario" data-state="{state["id"]}">'
        f'<p class="eyebrow">Generated what-if case</p><h3>{escape(state["title"])}</h3>'
        f'<div class="entities" aria-label="The two items being compared">{entities}</div>'
        f'<p class="outcome">{escape(state["outcome"])}</p><p>{escape(state["explanation"])}</p>'
        f'<p class="boundary {state["boundary"]}"><strong>{boundary}.</strong> {escape(state["boundary_note"])}</p>'
        f'<h4>Assumptions in this case</h4>{_list(state["assumptions"])}'
        f'<h4>Reasoning</h4>{_list(state["reasoning"])}'
        f'<h4>Still unknown</h4>{_list(state["unknowns"])}'
        f'{_guide.citations(data, state["citations"])}'
        f'<p class="next"><strong>Try next:</strong> {escape(state["next_step"])}</p></article>'
    )


def render(data: dict, evidence_dir: Path, template: Path | None = None) -> str:
    validate(data)
    lab = data["lab"]
    by_id = {s["id"]: s for s in lab["states"]}
    baseline = by_id[lab["baseline"]]
    controls = []
    for control in lab["controls"]:
        options = "".join(
            f'<option value="{option["id"]}"'
            + (" selected" if baseline["when"][control["id"]] == option["id"] else "")
            + f'>{escape(option["label"])}</option>'
            for option in control["options"]
        )
        controls.append(f'<label for="control-{control["id"]}">{escape(control["label"])}<select id="control-{control["id"]}" data-control="{control["id"]}">{options}</select></label>')
    sources = []
    for source in data["sources"]:
        content = f'<p>{escape(source["summary"])}</p>'
        if source["kind"] == "frame":
            content += f'<figure>{_guide.frame_image(source, evidence_dir)}<figcaption>{escape(source.get("caption", source["summary"]))}</figcaption></figure>'
        sources.append(f'<article class="source" id="source-{source["id"]}"><h3>{escape(source["id"])} · {escape(source["kind"])}</h3>{_guide.source_link(data, source)}{content}</article>')
    # The allowlist excludes unrelated or private metadata from saved artifacts.
    client = {
        "title": data["title"], "baseline": lab["baseline"],
        "controls": [{"id": c["id"], "label": c["label"], "options": [{"id": o["id"], "label": o["label"]} for o in c["options"]]} for c in lab["controls"]],
        "states": [{key: state[key] for key in ("id", "when", "title", "outcome", "boundary", "boundary_note", "citations")} for state in lab["states"]],
    }
    # JSON lives in a non-executable script element. Escape '<' to prevent a
    # source-supplied </script> ending that element in the HTML parser.
    encoded = json.dumps(client, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    cov = data["coverage"]
    values = {
        "TITLE": escape(data["title"]), "QUESTION": escape(lab["question"]),
        "PREMISE": escape(lab["premise"]), "LIMIT": escape(lab["model_limit"]),
        "LECTURE": escape(data["lecture"]["title"]),
        "COVERAGE": escape(f'{cov["kind"].capitalize()} · {_guide.clock(cov["start_s"])}–{_guide.clock(cov["end_s"])}'),
        "GAPS": escape(" ".join(cov.get("gaps", []))),
        "CONTROLS": "".join(controls), "BASELINE": _state(data, baseline),
        "STATES": "".join(_state(data, state) for state in lab["states"]),
        "SOURCES": "".join(sources), "DATA": encoded,
    }
    template = template or Path(__file__).resolve().parents[1] / "templates" / "assumption-lab.html"
    return re.sub(r"\{\{([A-Z]+)\}\}", lambda match: values[match[1]], template.read_text(encoding="utf-8"))


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
