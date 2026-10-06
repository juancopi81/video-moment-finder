"""CSV transport, source provenance, and offline HTML safety checks."""
import base64
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/video-moment-finder"
SPEC = importlib.util.spec_from_file_location("vmf_flashcards", PLUGIN / "tools/flashcards.py")
cards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cards)


@pytest.fixture
def deck():
    return json.loads((PLUGIN / "examples/flashcards.json").read_text(encoding="utf-8"))


def rows(output):
    return list(csv.reader(io.StringIO(output, newline="")))


def test_csv_roundtrip_keeps_multiline_quotes_unicode_tabs_and_evidence(deck):
    deck["cards"][0]["front"] = 'A "quote", a comma, and a\ttab\nUnicode: α × β'
    deck["cards"][0]["back"] = 'Line one\r\nLine two, "quoted"\n<script>text, not markup</script>'
    output = cards.export_csv(deck)
    result = rows(output)
    assert result[0] == ["Front", "Back", "Tags"]
    assert len(result) == len(deck["cards"]) + 1
    assert all(len(row) == 3 for row in result)
    assert result[1][0] == deck["cards"][0]["front"]
    assert result[1][1].startswith(deck["cards"][0]["back"])
    assert "s-definition [synthetic] 00:00–00:25" in result[1][1]
    assert "Synthetic · 00:00–01:30" in result[1][1]
    assert result[1][2] == "vmf recall dot-product definition"
    assert output == cards.export_csv(deck)
    assert output.startswith('"Front","Back","Tags"\r\n')


@pytest.mark.parametrize("hostile", ["=1+1", "+SUM(A1:A2)", "-1+1", "@A1", " \t=2", "\n=2", "\ufeff=2", "\u200b@A1", "\tordinary text"])
def test_csv_formula_safety_handles_obfuscated_leading_characters(deck, hostile):
    deck["cards"][0]["front"] = hostile
    deck["cards"][0]["back"] = hostile
    result = rows(cards.export_csv(deck))
    assert result[1][0] == "'" + hostile
    assert result[1][1].startswith("'" + hostile)
    assert deck["cards"][0]["front"] == hostile  # original JSON is not mutated


@pytest.mark.parametrize("text", ["1 + 1 = 2", "Why −1?", "café", "'already quoted"])
def test_safe_text_is_not_rewritten(text):
    assert cards.spreadsheet_text(text) == text


def test_html_escapes_source_markup_without_second_template_expansion(deck, tmp_path):
    hostile = '</script><img src=x onerror="alert(1)"> {{SOURCES}}'
    deck["cards"][0]["front"] = hostile
    deck["cards"][0]["back"] = hostile
    deck["sources"][0]["summary"] = hostile
    output = cards.render(deck, tmp_path)
    assert '<img src=x' not in output
    assert output.count('<script>') == 1
    assert '&lt;/script&gt;&lt;img' in output
    assert '{{SOURCES}}' in output
    assert 'default-src \'none\'' in output
    assert 'id="question-coordinate-rule" tabindex="-1"' in output
    assert 'id="review-controls"' in output
    assert '<details class="answer"><summary>Reveal answer</summary>' in output


def test_download_contains_exact_companion_csv(deck, tmp_path):
    output = cards.render(deck, tmp_path)
    match = re.search(r'href="data:text/csv;charset=utf-8;base64,([A-Za-z0-9+/=]+)"', output)
    assert match
    assert base64.b64decode(match[1]).decode('utf-8') == cards.export_csv(deck)


@pytest.mark.parametrize("change,match", [
    ({"citations": []}, "evidence citations"),
    ({"citations": ["s-fabricated"]}, "evidence citations"),
    ({"citations": ["s-definition", "s-definition"]}, "Duplicate card citation"),
    ({"rationale": ""}, "rationale"),
    ({"origin": "lecture"}, "Synthetic evidence"),
    ({"kind": "spaced-repetition"}, "card kind"),
    ({"tags": ["<b>HTML</b>"]}, "Tags"),
    ({"front": "abc\x00def"}, "control characters"),
])
def test_unsupported_cards_fail_before_output(deck, change, match):
    deck["cards"][0].update(change)
    with pytest.raises(ValueError, match=match):
        cards.export_csv(deck)


def test_redundant_questions_and_duplicate_ids_fail(deck):
    copy_card = copy.deepcopy(deck["cards"][0])
    copy_card["id"] = "new-id"
    copy_card["front"] = "  " + copy_card["front"].upper() + "  "
    deck["cards"].append(copy_card)
    with pytest.raises(ValueError, match="Duplicate question"):
        cards.validate(deck)
    copy_card["front"] = "A different question?"
    copy_card["id"] = deck["cards"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate card ID"):
        cards.validate(deck)


def test_real_lecture_references_use_only_valid_uuid_and_exact_start(deck):
    deck["lecture"]["video_id"] = "11111111-2222-4333-8444-555555555555"
    deck["sources"][0].update(kind="transcript", start_s=223.4, end_s=234.8)
    deck["cards"][0]["origin"] = "lecture"
    result = rows(cards.export_csv(deck))[1][1]
    assert "03:43–03:54" in result
    assert "https://www.videomomentfinder.com/video/11111111-2222-4333-8444-555555555555?t=223" in result
    deck["lecture"]["video_id"] = "javascript:alert(1)"
    with pytest.raises(ValueError):
        cards.export_csv(deck)


def test_missing_frame_keeps_sources_readable_and_path_escape_is_rejected(deck, tmp_path):
    deck["sources"][0].update(kind="frame", image_path="missing.jpg")
    output = cards.render(deck, tmp_path)
    assert "source image is unavailable" in output
    assert "s-definition" in output
    deck["sources"][0]["image_path"] = "../outside.jpg"
    with pytest.raises(ValueError, match="within"):
        cards.render(deck, tmp_path)


def test_cli_writes_matching_html_and_csv_without_overwriting_input(deck, tmp_path):
    source, output = tmp_path / 'deck.json', tmp_path / 'deck.html'
    source.write_text(json.dumps(deck), encoding='utf-8')
    subprocess.run([sys.executable, str(PLUGIN / 'tools/flashcards.py'), str(source), str(output)], check=True)
    assert output.read_text(encoding='utf-8') == cards.render(deck, tmp_path)
    assert output.with_suffix('.csv').read_bytes() == cards.export_csv(deck).encode('utf-8')
    before = source.read_bytes()
    result = subprocess.run([sys.executable, str(PLUGIN / 'tools/flashcards.py'), str(source), str(source)], capture_output=True)
    assert result.returncode != 0
    assert source.read_bytes() == before


@pytest.mark.parametrize("value", [float('nan'), float('inf'), -1, True, "223"])
def test_bad_timestamps_cannot_enter_cards_or_csv(deck, value):
    deck["sources"][0]["start_s"] = value
    with pytest.raises(ValueError):
        cards.export_csv(deck)


def test_review_script_navigation_boundaries_and_answer_reset(deck, tmp_path):
    """Execute the actual script against a small DOM double; this is not a browser/layout test."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the script-level interaction check')
    page = tmp_path / 'deck.html'
    page.write_text(cards.render(deck, tmp_path), encoding='utf-8')
    harness = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[1], 'utf8');
const count = (html.match(/class="flashcard"/g) || []).length;
const cardNodes = Array.from({length: count}, () => ({
  hidden: false, heading: {focused: false, focus() { this.focused = true; }},
  details: [{open: true}, {open: true}],
  querySelector(selector) { assert.equal(selector, 'h2'); return this.heading; },
  querySelectorAll(selector) { assert.equal(selector, 'details'); return this.details; }
}));
const names = ['review-controls','previous-card','next-card','toggle-list','review-status'];
const nodes = Object.fromEntries(names.map(id => {
  assert.ok(html.includes(`id="${id}"`), `Missing control ${id}`);
  return [id, {hidden: true, disabled: false, textContent: '', handlers: {}, attrs: {},
    addEventListener(event, handler) { this.handlers[event] = handler; },
    setAttribute(key, value) { this.attrs[key] = value; }
  }];
}));
const document = {
  querySelectorAll(selector) { assert.equal(selector, '.flashcard'); return cardNodes; },
  getElementById(id) { assert.ok(nodes[id], `Unexpected control ${id}`); return nodes[id]; }
};
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
vm.runInNewContext(script, {document});
const prev = nodes['previous-card'], next = nodes['next-card'], toggle = nodes['toggle-list'];
const status = () => nodes['review-status'].textContent;
assert.equal(nodes['review-controls'].hidden, false);
assert.equal(status(), `Card 1 of ${count}`);
assert.equal(prev.disabled, true);
assert.equal(cardNodes.filter(card => !card.hidden).length, 1);
prev.handlers.click();
assert.equal(status(), `Card 1 of ${count}`);
next.handlers.click();
assert.equal(status(), `Card 2 of ${count}`);
assert.equal(cardNodes[0].hidden, true);
assert.equal(cardNodes[1].hidden, false);
assert.equal(cardNodes[1].heading.focused, true);
assert.ok(cardNodes[1].details.every(detail => !detail.open));
for (let i = 1; i < count; i++) next.handlers.click();
assert.equal(status(), `Card ${count} of ${count}`);
assert.equal(next.disabled, true);
next.handlers.click();
assert.equal(status(), `Card ${count} of ${count}`);
toggle.handlers.click();
assert.equal(status(), `All ${count} cards`);
assert.ok(cardNodes.every(card => !card.hidden));
assert.equal(prev.disabled, true);
assert.equal(next.disabled, true);
assert.equal(toggle.attrs['aria-pressed'], 'true');
prev.handlers.click();
assert.equal(status(), `All ${count} cards`);
toggle.handlers.click();
assert.equal(status(), `Card ${count} of ${count}`);
assert.equal(toggle.attrs['aria-pressed'], 'false');
prev.handlers.click();
assert.equal(status(), `Card ${count - 1} of ${count}`);
'''
    subprocess.run([node, '-e', harness, str(page)], check=True, capture_output=True, text=True)
