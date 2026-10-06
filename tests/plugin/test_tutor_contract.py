"""Integrity checks for authored examples, NOT an LLM or real-learner test."""
import json
from pathlib import Path
import re


PLUGIN = Path(__file__).resolve().parents[2] / "plugins/video-moment-finder"


def test_tutor_fixture_has_honest_scope_known_evidence_and_single_turn_boundaries():
    examples = PLUGIN / 'examples'
    fixture = json.loads((examples / 'tutor-scenarios.json').read_text(encoding='utf-8'))
    assert fixture['status'] == 'authored_simulation_not_live_conversation'
    evidence_path = (examples / fixture['evidence_file']).resolve()
    assert evidence_path.is_relative_to(examples.resolve())
    lesson = json.loads(evidence_path.read_text(encoding='utf-8'))
    assert lesson['coverage']['kind'] == 'synthetic'
    assert lesson['lecture']['video_id'] is None
    ids = {source['id'] for source in lesson['sources']}
    assert fixture['opening']['tutor'].count('?') == 1
    assert fixture['opening']['action'] == 'wait_for_learner'
    for case in fixture['cases']:
        assert case['citations'] and set(case['citations']) <= ids
        assert case['tutor'].count('?') == 1
        assert case['next_action'] == 'wait_for_learner'
        assert not re.search(r'\n\s*(Learner|Student|User):', case['tutor'])
        assert case['review']  # fixture review is explicit, not a pass-rate claim


def test_fixture_exercises_distinct_correct_incorrect_and_ambiguous_behaviors():
    fixture = json.loads((PLUGIN / 'examples/tutor-scenarios.json').read_text(encoding='utf-8'))
    cases = {case['id']: case for case in fixture['cases']}
    assert cases['correct']['assessment'] == 'correct'
    assert '6−4=2' in cases['correct']['tutor']
    assert cases['incorrect']['assessment'] == 'incorrect'
    assert 'Hint:' in cases['incorrect']['tutor']
    assert '6−4=2' not in cases['incorrect']['tutor']  # no unrequested full solution
    assert cases['ambiguous']['assessment'] == 'ambiguous'
    assert 'do you mean' in cases['ambiguous']['tutor']
    assert cases['unsupported']['assessment'] == 'unsupported'
    assert 'specifies no' in cases['unsupported']['tutor']
    assert cases['source-injection']['assessment'] == 'source_instruction_ignored'
    assert 'not mathematical evidence' in cases['source-injection']['tutor']
