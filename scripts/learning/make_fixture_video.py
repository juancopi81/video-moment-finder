#!/usr/bin/env python3
"""Create an original narrated vector lesson locally; never uploads or indexes it."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCENES = [
    ("A DOT PRODUCT IS A SCALAR", "(3, 4) · (2, 0) = 6", "Multiply matching coordinates. Add the products.",
     "A dot product multiplies matching coordinates and adds the results. For vectors three comma four and two comma zero, the dot product is three times two, plus four times zero. That equals six. The result is a scalar."),
    ("PERPENDICULAR, NONZERO VECTORS", "(1, 0) · (0, 1) = 0", "A right angle gives a zero dot product.",
     "For nonzero vectors, the dot product equals the product of their lengths times the cosine of the angle between them. The vectors one comma zero and zero comma one are perpendicular. Their dot product is zero."),
    ("CHECK THE BOUNDARY CASE", "(0, 0) · any vector = 0", "The zero vector has no defined angle.",
     "Now check an assumption. A zero vector has a zero dot product with every vector, but its direction and angle are undefined. So zero dot product implies a right angle only when both vectors are nonzero. Always check the assumptions."),
]


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def choose_font(explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit else [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    raise ValueError("No supported font found. Supply --font with a local TrueType font.")


def choose_speech_engine(requested: str) -> str:
    for command in ("ffmpeg", "ffprobe"):
        if not shutil.which(command):
            raise ValueError(f"Install {command} before generating the fixture")
    if requested in ("auto", "flite"):
        filters = run(["ffmpeg", "-hide_banner", "-filters"]).stdout
        if any(len(line.split()) > 1 and line.split()[1] == "flite" for line in filters.splitlines()):
            return "flite"
        if requested == "flite":
            raise ValueError("This ffmpeg build has no flite filter; use --speech-engine say on macOS")
    if shutil.which("say"):
        return "say"
    raise ValueError("No local speech engine: use ffmpeg with libflite, or macOS say")


def audio_duration(path: Path) -> float:
    raw = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]).stdout.strip()
    try:
        duration = float(raw)
    except ValueError:
        duration = 0
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Speech generation produced no usable audio. Check local speech-engine permissions before retrying; no video has been uploaded.")
    return duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--font", type=Path, help="Local TrueType font; auto-detected on Linux/macOS")
    parser.add_argument("--speech-engine", choices=("auto", "flite", "say"), default="auto")
    parser.add_argument("--say-voice", default="Samantha", help="Installed English macOS voice; ignored for flite")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if any(c in str(root) for c in "'\\:\n"):
        raise ValueError("Use an output path without filter-special characters")
    font_path = choose_font(args.font)
    speech_engine = choose_speech_engine(args.speech_engine)
    root.mkdir(parents=True, exist_ok=True)
    title_font = ImageFont.truetype(str(font_path), 33)
    equation_font = ImageFont.truetype(str(font_path), 64)
    body_font = ImageFont.truetype(str(font_path), 30)
    timeline, elapsed = [], 0.0
    for index, (title, equation, caption, speech) in enumerate(SCENES):
        image = Image.new("RGB", (1280, 720), "#f7f5ee")
        draw = ImageDraw.Draw(image)
        draw.text((70, 65), "VMF  /  ORIGINAL VECTOR LAB", font=title_font, fill="#b94b27")
        draw.text((70, 205), title, font=title_font, fill="#183329")
        draw.text((70, 315), equation, font=equation_font, fill="#183329")
        draw.text((70, 460), caption, font=body_font, fill="#56665d")
        draw.text((70, 645), f"Scene {index + 1} of 3 · Original test material", font=body_font, fill="#56665d")
        png, text, wav, clip = [root / f"scene-{index}.{ext}" for ext in ("png", "txt", "wav", "mp4")]
        image.save(png)
        text.write_text(speech, encoding="utf-8")
        if speech_engine == "flite":
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"flite=textfile='{text}':voice=slt", "-ar", "24000", str(wav)])
        else:
            run(["say", "-v", args.say_voice, "-f", str(text), "-o", str(wav), "--file-format=WAVE", "--data-format=LEI16@24000"])
        duration = audio_duration(wav) + 1
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(png), "-i", str(wav), "-t", str(duration), "-vf", "format=yuv420p", "-af", "apad", "-c:v", "libx264", "-preset", "fast", "-crf", "26", "-r", "15", "-c:a", "aac", "-movflags", "+faststart", str(clip)])
        timeline.append({"scene": index + 1, "start_s": round(elapsed, 3), "end_s": round(elapsed + duration, 3), "narration": speech, "title": title})
        elapsed += duration
    listing = root / "concat.txt"
    listing.write_text("".join(f"file 'scene-{i}.mp4'\n" for i in range(len(SCENES))))
    output = root / "vmf-original-vector-lab.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "1", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(output)])
    (root / "evidence.json").write_text(json.dumps({"rights": "Original material created for VMF testing; repository license applies", "synthetic_voice": "local libflite slt" if speech_engine == "flite" else f"local macOS say {args.say_voice}", "video": output.name, "duration_s": elapsed, "scenes": timeline}, indent=2) + "\n", encoding="utf-8")
    logger.info("Created original fixture: %s", json.dumps({"video": str(output), "duration_s": round(elapsed, 3), "size_bytes": output.stat().st_size, "uploaded": False}))


if __name__ == "__main__":
    main()
