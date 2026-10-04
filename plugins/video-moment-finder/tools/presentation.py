#!/usr/bin/env python3
"""Prepare evidence-backed slides and an offline preview (standard library)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location("vmf_guide", Path(__file__).with_name("render.py"))
guide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guide)


def _text(value, label, limit=1600):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be nonempty text up to {limit} characters")


def validate(data):
    guide.validate({**data, "sections": [], "questions": [], "interaction": None})
    for name in ("audience", "purpose"):
        _text(data[name], name)
    slides = data["slides"]
    if not isinstance(slides, list) or not 3 <= len(slides) <= 16:
        raise ValueError("Provide 3–16 slides")
    sources = {s["id"]: s for s in data["sources"]}
    for slide in slides:
        _text(slide["title"], "slide.title", 90)
        _text(slide["notes"], "slide.notes", 5000)
        if slide["layout"] not in {"cover", "statement", "evidence", "comparison", "exercise"}:
            raise ValueError("Unknown slide layout")
        if slide["origin"] not in {"source", "generated", "mixed"}:
            raise ValueError("Unknown slide origin")
        if not isinstance(slide["body"], list) or not 1 <= len(slide["body"]) <= 3:
            raise ValueError("Use one to three body paragraphs")
        for text in slide["body"]:
            _text(text, "slide.body", 240)
        if not isinstance(slide.get("citations"), list) or not slide["citations"] or any(s not in sources for s in slide["citations"]):
            raise ValueError("Every slide needs valid citations, including the premise for generated examples")
        if "headline" in slide:
            _text(slide["headline"], "slide.headline", 90)
            if len(slide["body"]) > 2:
                raise ValueError("A large headline leaves room for at most two body paragraphs")
        if "image_source" in slide:
            if slide["layout"] != "evidence" or len(slide["body"]) > 2:
                raise ValueError("A frame uses the evidence layout with at most two paragraphs")
            sid = slide["image_source"]
            if sid not in slide["citations"] or sources[sid]["kind"] != "frame":
                raise ValueError("Image must reference a cited frame")
        if slide["layout"] == "comparison":
            if len(slide["body"]) != 1 or len(slide["body"][0]) > 180:
                raise ValueError("Comparison uses one short introductory paragraph")
            if not isinstance(slide.get("columns"), list) or len(slide["columns"]) != 2:
                raise ValueError("Comparison needs two columns")
            for c in slide["columns"]:
                _text(c["title"], "column.title", 50)
                _text(c["body"], "column.body", 240)


def prepare(data: dict, evidence_dir: Path) -> dict:
    validate(data)
    cov = data["coverage"]
    coverage = f'{cov["kind"].capitalize()} · {guide.clock(cov["start_s"])}–{guide.clock(cov["end_s"])}'
    prepared = {"format": "vmf-presentation-1", "title": data["title"], "lecture": data["lecture"]["title"],
                "audience": data["audience"], "purpose": data["purpose"], "coverage": coverage,
                "gaps": list(cov.get("gaps", [])), "slides": []}
    sources = {s["id"]: s for s in data["sources"]}
    for index, original in enumerate(data["slides"]):
        s = {k: original[k] for k in ("title", "layout", "origin", "body", "notes")}
        for key in ("headline", "columns"):
            if key in original:
                s[key] = original[key] if key == "headline" else [{"title": c["title"], "body": c["body"]} for c in original[key]]
        evidence = []
        for sid in original["citations"]:
            source = sources[sid]
            item = {"id": sid, "time": guide.clock(source["start_s"])+"–"+guide.clock(source["end_s"]), "start_s": source["start_s"], "end_s": source["end_s"], "summary": source["summary"]}
            video_id = data["lecture"].get("video_id")
            if video_id:
                item["url"] = f"https://www.videomomentfinder.com/video/{video_id}?t={int(source['start_s'])}"
            evidence.append(item)
        s["evidence"] = evidence
        if original.get("image_source"):
            source = sources[original["image_source"]]
            markup = guide.frame_image(source, evidence_dir)
            image = re.search(r'src="(data:image/(?:png|jpeg);base64,[A-Za-z0-9+/=]+)"', markup)
            if image:
                s["image"] = image[1]
            s["image_alt"] = source.get("alt", source["summary"])
            s["image_caption"] = source.get("caption", "Actual source frame at " + guide.clock(source["start_s"]))
            if not image:
                s["image_missing"] = "Source frame unavailable in this copy; evidence notes remain available."
        provenance = {"source": "Source content", "generated": "Generated example", "mixed": "Source + generated explanation"}[s["origin"]]
        s["label"] = provenance
        notes = [s["notes"], "", provenance, f"Source: {prepared['lecture']}", "Coverage: " + coverage]
        if index == 0:
            notes.extend(prepared["gaps"])
        for e in evidence:
            notes.append(f"{e['id']} [{e['time']}; exact seconds {e['start_s']}–{e['end_s']}]: {e['summary']}" + ("\n" + e["url"] if "url" in e else ""))
        if s.get("image_caption"):
            notes.append(s["image_caption"])
        s["speaker_notes"] = "\n".join(notes)
        prepared["slides"].append(s)
    return prepared


def render(prepared):
    esc = guide.escape
    slides = []
    for index, s in enumerate(prepared["slides"], 1):
        body = "".join(f"<p>{esc(p)}</p>" for p in s["body"])
        headline = f'<p class="headline">{esc(s["headline"])}</p>' if s.get("headline") else ""
        image = ""
        if s.get("image"):
            image = f'<figure><img src="{s["image"]}" alt="{esc(s["image_alt"])}"><figcaption>{esc(s["image_caption"])}</figcaption></figure>'
        elif s.get("image_missing"):
            image = f'<p class="missing">{esc(s["image_missing"])}</p>'
        columns = '<div class="columns">'+"".join(f'<section><h3>{esc(c["title"])}</h3><p>{esc(c["body"])}</p></section>' for c in s.get("columns", []))+"</div>" if s.get("columns") else ""
        refs = " · ".join(
            f'<a href="{esc(e["url"])}" target="_blank" rel="noopener noreferrer">{esc(e["id"])} {esc(e["time"])}</a>'
            if e.get("url") else esc(f'{e["id"]} {e["time"]}') for e in s["evidence"]
        )
        slides.append(f'<article class="slide {s["layout"]}" id="slide-{index}" aria-label="Slide {index}: {esc(s["title"])}"><div class="slide-inner"><p class="label">{esc(s["label"])}</p><h2>{esc(s["title"])}</h2><div class="composition"><div class="copy">{headline}{body}</div>{image}{columns}</div><div class="slide-footer"><span>{refs}</span><span>{index:02}</span></div></div><details><summary>Speaker notes and sources</summary><pre>{esc(s["speaker_notes"])}</pre></details></article>')
    values = {"TITLE": esc(prepared["title"]), "AUDIENCE": esc(prepared["audience"]), "COVERAGE": esc(prepared["coverage"]), "COUNT": str(len(slides)), "SLIDES": "".join(slides), "GAPS": esc(" ".join(prepared["gaps"]))}
    template = Path(__file__).resolve().parents[1] / "templates/presentation.html"
    return re.sub(r"\{\{([A-Z]+)\}\}", lambda m: values[m[1]], template.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    paths = [args.input.resolve(), args.output.resolve()] + ([args.prepared.resolve()] if args.prepared else [])
    if len(paths) != len(set(paths)):
        parser.error("Input and output paths must differ")
    prepared = prepare(json.loads(args.input.read_text()), args.input.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(prepared), encoding="utf-8")
    if args.prepared:
        args.prepared.parent.mkdir(parents=True, exist_ok=True)
        args.prepared.write_text(json.dumps(prepared, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
