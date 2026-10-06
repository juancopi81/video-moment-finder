#!/usr/bin/env python3
"""Inspect the actual portable VMF ZIP without extracting or executing its contents.

This enforces the documented package contract. It is not the Developer Portal's
schema validator or a claim of submission eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from PIL import Image

PLUGIN_NAME = "video-moment-finder"
SEMVER = r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
CORE_SKILLS = {"study-guide", "flashcards", "tutor", "assumption-lab", "presentation"}
MAX_UNCOMPRESSED = 30 * 1024 * 1024
TOP_LEVEL = {"plugin.json", "mcp.json", "README.md", "LICENSE", "assets", "skills", "templates", "references", "tools", "examples", ".codex-plugin"}
FORBIDDEN_PARTS = {".git", ".env", ".app.json", "node_modules", ".venv", "__pycache__", "private", ".DS_Store"}
TEXT_SUFFIXES = {".json", ".md", ".html", ".css", ".js", ".mjs", ".py", ".tsv", ".csv", ".txt", ".yaml", ".yml", ".svg"}
SECRET_PATTERNS = [
    r"(?i)x-amz-(?:signature|credential)=",
    r"(?i)(?:access_token|refresh_token|client_secret)[\"']?\s*[:=]\s*[\"'][^\"'\s]{12,}",
    r"\bvmf_[A-Za-z0-9_-]{20,}",
    r"\bBearer\s+[A-Za-z0-9._~-]{24,}",
    r"\b(?:app_|plugin_)[0-9a-f]{24,}\b",
    r"/(?:Users|home)/[^\s/]+/",
]


def _json(data: bytes, label: str) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return value


def _https(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 1024:
        return False
    parsed = urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username and not parsed.password


def validate_archive(archive: Path, *, submission: bool = False) -> dict:
    errors: list[str] = []
    gaps: list[str] = []
    inventory: dict[str, bytes] = {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    try:
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            require(sum(item.file_size for item in entries) <= MAX_UNCOMPRESSED, "Archive exceeds 30 MiB uncompressed")
            if errors:
                return {"valid": False, "errors": errors, "submission_gaps": gaps}
            seen = set()
            for item in entries:
                path = PurePosixPath(item.filename)
                require(item.filename not in seen, f"Duplicate archive member: {item.filename}")
                seen.add(item.filename)
                safe = not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename and "\x00" not in item.filename
                require(safe, f"Unsafe archive path: {item.filename}")
                require(path.parts and path.parts[0] == PLUGIN_NAME, f"Expected single {PLUGIN_NAME}/ root: {item.filename}")
                require(not stat.S_ISLNK(item.external_attr >> 16), f"Symlink in archive: {item.filename}")
                require(not item.flag_bits & 1, f"Encrypted archive member: {item.filename}")
                require(not any(part in FORBIDDEN_PARTS or part.startswith(".env.") for part in path.parts), f"Private or build-only content: {item.filename}")
                if not safe or not path.parts or path.parts[0] != PLUGIN_NAME or item.is_dir():
                    continue
                relative = "/".join(path.parts[1:])
                require(bool(relative) and path.parts[1] in TOP_LEVEL, f"Unrecognized top-level package file: {item.filename}")
                require(not relative.endswith((".pyc", ".pyo", ".zip", ".pem", ".key")), f"Forbidden file type: {relative}")
                inventory[relative] = bundle.read(item)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        errors.append(f"Cannot read archive: {exc}")
        return {"valid": False, "errors": errors, "submission_gaps": gaps}

    for required in ("plugin.json", "mcp.json", "README.md", "LICENSE"):
        require(required in inventory, f"Missing {required}")
    if "plugin.json" not in inventory or "mcp.json" not in inventory:
        return {"valid": False, "errors": errors, "submission_gaps": gaps}
    try:
        manifest = _json(inventory["plugin.json"], "plugin.json")
        mcp = _json(inventory["mcp.json"], "mcp.json")
    except (ValueError, UnicodeDecodeError) as exc:
        errors.append(str(exc))
        return {"valid": False, "errors": errors, "submission_gaps": gaps}

    require(manifest.get("name") == PLUGIN_NAME, "Manifest name must match archive directory")
    require(bool(re.fullmatch(SEMVER, str(manifest.get("version", "")))), "version must be a semantic version")
    require(manifest.get("$schema") == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "Unexpected portable plugin schema")
    for name, data in inventory.items():
        if name.endswith("plugin.json"):
            try:
                overlay = _json(data, name)
                for field in ("skills", "mcpServers", "interface"):
                    require(field not in overlay, f"{name}: nonportable top-level {field}")
                require(overlay.get("apps") is None and overlay.get("extensions", {}).get("com.openai", {}).get("apps") is None, f"{name}: personal app binding is not allowed")
                if name != "plugin.json":
                    require(overlay.get("name") == manifest.get("name") and overlay.get("version") == manifest.get("version"), f"{name}: compatibility identity/version differ")
            except (ValueError, UnicodeDecodeError, AttributeError) as exc:
                errors.append(f"{name}: {exc}")

    extensions = manifest.get("extensions", {})
    if not isinstance(extensions, dict) or not isinstance(extensions.get("com.openai"), dict):
        errors.append("extensions.com.openai must be an object")
        return {"valid": False, "errors": errors, "submission_gaps": gaps}
    extension = extensions["com.openai"]
    interface = extension.get("interface", {})
    if not isinstance(interface, dict):
        errors.append("extensions.com.openai.interface must be an object")
        return {"valid": False, "errors": errors, "submission_gaps": gaps}
    for field, limit in (("displayName", 30), ("shortDescription", 30), ("longDescription", 4000)):
        value = interface.get(field)
        require(isinstance(value, str) and 0 < len(value.strip()) <= limit, f"{field} must be nonblank and at most {limit} characters")
    developer = interface.get("developerName")
    if developer is not None:
        require(isinstance(developer, str) and 0 < len(developer.strip()) <= 80, "developerName must be 1–80 characters")
    else:
        gaps.append("Verified publisher identity/developerName has not been supplied")
    prompts = interface.get("defaultPrompt", [])
    prompts = [prompts] if isinstance(prompts, str) else prompts
    require(isinstance(prompts, list) and 1 <= len(prompts) <= 3, "Provide 1–3 default prompts")
    if isinstance(prompts, list):
        require(all(isinstance(p, str) and 0 < len(p.strip()) <= 128 and "\n" not in p and "\r" not in p and "@" not in p for p in prompts), "Default prompts must be nonblank single lines of at most 128 characters without @mentions")
        require(len({" ".join(str(p).split()) for p in prompts}) == len(prompts), "Default prompts must be unique")
    for field in ("websiteURL", "supportURL", "privacyPolicyURL", "termsOfServiceURL"):
        require(_https(interface.get(field)), f"{field} needs an absolute HTTPS URL without credentials")
    gaps.append("Deployed-service compliance and release evidence require checks outside this archive; see docs/plugin/SUBMISSION.md")

    for field, minimum in (("logo", 256), ("composerIcon", 48), ("logoDark", 256), ("composerIconDark", 48)):
        if field.endswith("Dark") and field not in interface:
            continue
        value = interface.get(field, "")
        normalized = posixpath.normpath(value) if isinstance(value, str) else ""
        require(normalized.startswith("assets/") and normalized in inventory, f"{field} must refer to a contained asset")
        if normalized in inventory:
            payload = inventory[normalized]
            require(len(payload) <= 5 * 1024 * 1024, f"{field} exceeds 5 MiB")
            try:
                with Image.open(io.BytesIO(payload)) as icon:
                    require(icon.format == "PNG", f"{field} must use PNG for public submission")
                    require(icon.width == icon.height and minimum <= icon.width <= 4096, f"{field} must be square and {minimum}–4096 pixels")
                    icon.verify()
            except (OSError, ValueError) as exc:
                errors.append(f"Invalid {field}: {exc}")

    servers = mcp.get("mcpServers", {})
    require(mcp.get("$schema") == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json", "Unexpected portable MCP schema")
    require(isinstance(servers, dict) and set(servers) == {PLUGIN_NAME}, "Exactly one VMF MCP server is required")
    if isinstance(servers, dict):
        server = servers.get(PLUGIN_NAME, {})
        if not isinstance(server, dict):
            errors.append("VMF MCP server must be an object")
            return {"valid": False, "errors": errors, "submission_gaps": gaps}
        require(server.get("type") == "streamable-http", "MCP transport must be streamable-http")
        require(server.get("url") == "https://api.videomomentfinder.com/mcp", "MCP endpoint differs from the verified service contract")
        require(set(server) <= {"type", "url"}, "MCP config must use host-managed OAuth without headers, personal bindings or duplicate review metadata")

    skill_names = set()
    for path, data in inventory.items():
        if re.fullmatch(r"skills/[^/]+/SKILL\.md", path):
            skill_name = path.split("/")[1]
            skill_names.add(skill_name)
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"Skill must be UTF-8: {path}")
                continue
            require(bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill_name)) and len(skill_name) <= 64, f"{path}: invalid skill directory name")
            frontmatter = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", content, re.S)
            require(bool(frontmatter), f"{path}: missing YAML frontmatter")
            if frontmatter:
                name_match = re.search(r"^name:\s*(.*?)\s*$", frontmatter[1], re.M)
                require(bool(name_match) and name_match[1].strip("\"'") == skill_name, f"{path}: frontmatter name must match directory")
                require(bool(re.search(r"^description:\s*\S", frontmatter[1], re.M)), f"{path}: missing discoverability description")
    require(CORE_SKILLS <= skill_names, f"Missing core skills: {', '.join(sorted(CORE_SKILLS - skill_names))}")
    require(skill_names == CORE_SKILLS, "Expect the five documented VMF workflows")

    review = extension.get("review", {})
    if not isinstance(review, dict) or not isinstance(review.get("test_cases"), dict):
        errors.append("review.test_cases must be an object")
        return {"valid": False, "errors": errors, "submission_gaps": gaps}
    cases = review["test_cases"]
    for kind, count in (("positive", 5), ("negative", 3)):
        group = cases.get(kind, [])
        require(isinstance(group, list) and len(group) == count, f"Exactly {count} {kind} review cases are required")
        if isinstance(group, list):
            for index, case in enumerate(group, 1):
                fields = ("description", "prompt", "tools_triggered", "expected_behavior") if kind == "positive" else ("description", "prompt")
                require(isinstance(case, dict) and all(isinstance(case.get(field), str) and bool(case[field].strip()) for field in fields), f"{kind} case {index}: required fields must be nonblank strings")
    if not _https(review.get("demo_recording_url")):
        gaps.append("A verified reviewer-accessible demonstration recording URL is missing")
    if "commerce" not in review or "commerce_description" not in review:
        gaps.append("Commerce declarations await publisher confirmation and current-policy review")
    publication = extension.get("publication", {})
    if not isinstance(publication, dict):
        errors.append("publication must be an object")
        return {"valid": False, "errors": errors, "submission_gaps": gaps}
    require(isinstance(publication.get("release_notes"), str) and bool(publication["release_notes"].strip()), "Publication release notes are required")
    if "countries" not in publication:
        gaps.append("Supported-country targeting has not been selected by the publisher")
    gaps.append("Dedicated reviewer access, actual case runs, domain/developer verification and legal attestations remain portal gates")

    for name, data in inventory.items():
        if PurePosixPath(name).suffix not in TEXT_SUFFIXES:
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"Text file is not UTF-8: {name}")
            continue
        for pattern in SECRET_PATTERNS:
            require(not re.search(pattern, content), f"Potential secret, signed URL or personal binding in {name}")
        links = re.findall(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", content) if name.endswith(".md") else []
        for link in links:
            link = link.strip("<>")
            if urlsplit(link).scheme or link.startswith("#"):
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(urlsplit(link).path)))
            require(not target.startswith("../") and not target.startswith("/"), f"{name}: link escapes package: {link}")
            require(target in inventory, f"{name}: missing local link target: {link}")

    if submission:
        errors.extend(f"Submission gap: {gap}" for gap in gaps)
    return {
        "valid": not errors,
        "validation_scope": "VMF portable archive contract; not portal schema validation or live end-user validation",
        "submission_ready": False,
        "archive": str(archive),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "version": manifest.get("version"),
        "files": len(inventory),
        "skills": sorted(skill_names),
        "errors": errors,
        "submission_gaps": gaps,
        "inventory": [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for path, data in sorted(inventory.items())],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--submission", action="store_true", help="Fail while required public-review facts remain unverified")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_archive(args.archive, submission=args.submission)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
