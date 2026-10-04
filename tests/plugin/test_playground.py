"""Numerical invariants, provenance, and offline artifact safety."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / 'plugins/video-moment-finder'
SPEC = importlib.util.spec_from_file_location('playground', PLUGIN / 'tools/playground.py')
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def fixture():
    return json.loads((PLUGIN / 'examples/playground.json').read_text())


def test_generated_model_has_sources_and_offline_fallback(tmp_path):
    page = renderer.render(fixture(), tmp_path)
    assert 'Synthetic · 00:00–00:45' in page
    assert '<noscript>' in page
    assert 'not a measured experiment' in page
    assert '<script src=' not in page
    for source in fixture()['playground']['citations']:
        assert f'id="source-{source}"' in page


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 6, True, '2'])
def test_invalid_coordinates_rejected(value):
    data = fixture()
    data['playground']['initial']['v'][0] = value
    with pytest.raises(ValueError):
        renderer.validate(data)


def test_unsupported_model_and_missing_citations_rejected():
    for change in [{'model': 'invent-physics'}, {'citations': []}, {'citations': ['absent']}]:
        data = fixture()
        data['playground'].update(change)
        with pytest.raises(ValueError):
            renderer.validate(data)


def test_hostile_text_is_not_executable_and_metadata_is_not_serialized(tmp_path):
    data = fixture()
    data['title'] = '</script><img src=x onerror=alert(1)> {{TITLE}}'
    data['secret'] = 'UNRELATED_PRIVATE_VALUE'
    data['playground']['initial']['token'] = 'UNRELATED_PRIVATE_VALUE'
    page = renderer.render(data, tmp_path)
    assert '<img src=x' not in page
    assert '&lt;/script&gt;' in page
    assert '\\u003c/script\\u003e' in page
    assert 'UNRELATED_PRIVATE_VALUE' not in page


def test_real_model_invariants_and_undefined_boundaries(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    harness = tmp_path / 'model.cjs'
    harness.write_text(r'''
const assert = require('node:assert/strict');
const {vectorMetrics:m} = require(process.argv[2]);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9, `${a} != ${b}`);
assert.deepEqual(m([0,0],[1,2]), {dot:0,nv:0,nw:Math.sqrt(5),cosine:null,scalar:null,projection:null,angle:null,components:[0,0]});
assert.equal(m([1,0],[0,0]).angle,null);
assert.deepEqual(m([1,0],[0,0]).projection,[0,0]);
near(m([3,4],[2,0]).dot,6);near(m([3,4],[2,0]).scalar,1.2);
near(m([3,2],[-2,3]).angle,90);near(m([3,1],[-3,-1]).angle,180);
assert.throws(()=>m([NaN,0],[1,0]));
// Independent geometric invariants across 1,600 pairs.
for(let x=-4;x<=5;x++)for(let y=-4;y<=5;y++)for(let a=-2;a<=1;a++)for(let b=-2;b<=1;b++){
 const v=[x,y],w=[a,b],r=m(v,w);
 near(r.dot,m(w,v).dot);
 near(m(v,w.map(n=>n*2)).dot,r.dot*2);
 // Lagrange's identity links the dot to the independent 2D determinant.
 near(r.dot*r.dot+(x*b-y*a)**2,(x*x+y*y)*(a*a+b*b));
 if(r.projection){
  near(r.projection[0]*y-r.projection[1]*x,0);
  near(x*(a-r.projection[0])+y*(b-r.projection[1]),0);
 }
 if(r.angle!==null){near(m(v,w.map(n=>n*2)).angle,r.angle);assert.ok(r.angle>=0&&r.angle<=180)}
}
''')
    result = subprocess.run([node, str(harness), str(PLUGIN/'tools/playground_math.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
