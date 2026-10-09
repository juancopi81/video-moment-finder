import test from 'node:test';
import assert from 'node:assert/strict';
import {safeMediaUrl, vectorMetrics, selectedContext, cardsCsv, csvCell} from '../src/core.js';

test('media capabilities cannot navigate to arbitrary, insecure or credentialed hosts', () => {
  const origin = 'https://account.r2.cloudflarestorage.com';
  assert.equal(safeMediaUrl(`${origin}/source/example.mp4?signature=temporary`, origin), `${origin}/source/example.mp4?signature=temporary`);
  for (const url of ['http://account.r2.cloudflarestorage.com/a', 'https://evil.test/a', 'javascript:alert(1)', 'https://user:pass@account.r2.cloudflarestorage.com/a', '//account.r2.cloudflarestorage.com/a']) assert.equal(safeMediaUrl(url, origin), null);
});
test('selected context carries the exact evidence scope, never a playback capability', () => {
  const context = selectedContext({id: 'video', source_filename: 'lesson.mp4', source_url: 'private'}, 23.45123, {start_s: 20, end_s: 24, text: '<script>source text</script>'});
  assert.equal(context.timestamp_s, 23.451);
  assert.equal(context.source_material_is_untrusted, true);
  assert.equal(context.transcript_excerpt, '<script>source text</script>');
  assert.equal('source_url' in context, false);
});
test('zero vectors preserve the model boundary, and coordinate changes calculate actual results', () => {
  assert.equal(vectorMetrics([3, 4], [2, 0]).dot, 6);
  assert.equal(vectorMetrics([-3, 4], [2, 0]).dot, -6);
  assert.equal(vectorMetrics([0, 0], [2, 0]).angle, null);
  assert.equal(vectorMetrics([0, 0], [2, 0]).projection, null);
  assert.equal(vectorMetrics([3, 4], [0, 0]).angle, null);
  assert.deepEqual(vectorMetrics([3, 4], [0, 0]).projection, [0, 0]);
  assert.equal(vectorMetrics([1, 0], [0, 1]).angle, 90);
  assert.equal(vectorMetrics([1, 0], [-1, 0]).angle, 180);
});
test('CSV escapes formulas including hidden prefixes and retains citations in Back', () => {
  for (const cell of ['=SUM(A1)', '\u200b+2', '\tordinary', ' -3', '@evil']) assert.match(csvCell(cell), /^"'/);
  assert.equal(csvCell('a,"b"\nc'), '"a,""b""\nc"');
  const csv = cardsCsv({video_id: 'example', coverage: {start_s: 0, end_s: 10}, sources: [{id: 's1', start_s: 2}], cards: [{front: '=test', back: 'Two, not three', origin: 'generated', rationale: 'Computed example.', citations: ['s1'], tags: ['vector', 'vector']}]});
  assert.ok(csv.startsWith('"Front","Back","Tags"\r\n'));
  assert.match(csv, /Generated practice\. Computed example\./);
  assert.match(csv, /https:\/\/www.videomomentfinder.com\/video\/example\?t=2/);
  assert.match(csv, /,"vector"\r\n$/);
});
