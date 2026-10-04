"""Evidence, provenance, contained assets and audience-independent deck data."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT/'plugins/video-moment-finder'
SPEC = importlib.util.spec_from_file_location('presentation', PLUGIN/'tools/presentation.py')
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def fixture(name='presentation'):
    return json.loads((PLUGIN/f'examples/{name}.json').read_text())


@pytest.mark.parametrize('name', ['presentation', 'work-briefing'])
def test_each_slide_keeps_coverage_citations_and_delivery_notes(name, tmp_path):
    source = fixture(name)
    prepared = renderer.prepare(source, tmp_path)
    page = renderer.render(prepared)
    assert len(prepared['slides']) == len(source['slides'])
    for before, after in zip(source['slides'], prepared['slides'], strict=True):
        assert before['notes'] in after['speaker_notes']
        assert 'Synthetic' in after['speaker_notes']
        for cid in before['citations']:
            assert cid in after['speaker_notes']
            assert cid in page
    assert '<script src=' not in page
    assert 'Speaker notes and sources' in page


def test_no_unknown_source_or_uncited_image_can_reach_export(tmp_path):
    for change in [{'citations': []}, {'citations': ['missing']}, {'image_source': 's-coordinate'}]:
        data=fixture();data['slides'][0].update(change)
        with pytest.raises(ValueError):
            renderer.prepare(data,tmp_path)


def test_hostile_markup_and_unrelated_metadata_stay_out(tmp_path):
    data=fixture();data['slides'][0]['title']='<script>alert(1)</script>'
    data['slides'][0]['body'][0]='<img src=x onerror=alert(2)>'
    data['account_token']='PRIVATE_UNRELATED'
    data['slides'][0]['private']='PRIVATE_UNRELATED'
    prepared=renderer.prepare(data,tmp_path);page=renderer.render(prepared)
    assert '<script>alert(1)' not in page
    assert '<img src=x' not in page
    assert '&lt;script&gt;' in page
    assert 'PRIVATE_UNRELATED' not in json.dumps(prepared)


def test_missing_image_is_disclosed_and_outside_paths_are_rejected(tmp_path):
    data=fixture();data['sources'][0].update(kind='frame',image_path='missing.png')
    data['slides'][0]['image_source']='s-coordinate'
    data['slides'][0]['layout']='evidence'
    prepared=renderer.prepare(data,tmp_path)
    assert 'unavailable' in prepared['slides'][0]['image_missing']
    assert 'Source frame unavailable' in renderer.render(prepared)
    data['sources'][0]['image_path']='../secret.png'
    with pytest.raises(ValueError,match='within'):
        renderer.prepare(data,tmp_path)


def test_work_briefing_preserves_proposal_decision_and_unknowns(tmp_path):
    prepared=renderer.prepare(fixture('work-briefing'),tmp_path)
    text=json.dumps(prepared)
    assert 'Fictional work scenario' in text
    assert 'No owner or due date' in text
    assert 'Generated example' in prepared['slides'][4]['label']
    assert 'not decisions' in prepared['slides'][4]['speaker_notes']


def test_content_budget_prevents_silent_slide_overflow():
    data=fixture();data['slides'][0]['body'][0]='x'*241
    with pytest.raises(ValueError):
        renderer.validate(data)


def test_bundled_work_deck_is_editable_and_keeps_every_source_note():
    data = fixture('work-briefing')
    ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(PLUGIN/'examples/work-briefing.pptx') as deck:
        for i, slide in enumerate(data['slides'], 1):
            content = ET.fromstring(deck.read(f'ppt/slides/slide{i}.xml'))
            text = '\n'.join(n.text or '' for n in content.findall('.//a:t', ns))
            assert slide['title'] in text  # native editable text, not flattened images
            notes = ET.fromstring(deck.read(f'ppt/notesSlides/notesSlide{i}.xml'))
            note_text = '\n'.join(n.text or '' for n in notes.findall('.//a:t', ns))
            assert slide['notes'] in note_text
            for cid in slide['citations']:
                assert cid in note_text
        assert not any(name.endswith('vbaProject.bin') for name in deck.namelist())
