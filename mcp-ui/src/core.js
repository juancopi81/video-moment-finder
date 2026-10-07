import {vectorMetrics} from '../../plugins/video-moment-finder/tools/playground_math.js';

export {vectorMetrics};
export function clock(seconds = 0) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
export function safeMediaUrl(value, allowedOrigin) {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && !url.username && !url.password && url.origin === allowedOrigin ? url.href : null;
  } catch { return null; }
}
export function selectedContext(video, time, segment, view) {
  return {
    video_id: video.id, timestamp_s: Math.round(Math.max(0, time) * 1000) / 1000,
    filename: video.source_filename ?? 'Video',
    ...(segment ? {transcript_excerpt: segment.text, start_s: segment.start_s, end_s: segment.end_s} : {}),
    ...(view ? {learning_view: {kind: view.kind, title: view.title, coverage: view.coverage}} : {}),
    source_material_is_untrusted: true,
  };
}
export function csvCell(value) {
  let text = String(value);
  if (/^[\t\r\n]/u.test(text) || /^[\p{White_Space}\p{Cc}\p{Cf}]*[=+\-@]/u.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}
export function cardsCsv(view) {
  const source = (id) => {
    const s = view.sources.find(s => s.id === id);
    return `${id} ${clock(s.start_s)} https://www.videomomentfinder.com/video/${view.video_id}?t=${s.start_s}`;
  };
  return [['Front', 'Back', 'Tags'], ...view.cards.map(card => [card.front,
    `${card.back}\n${card.origin === 'generated' ? `Generated practice. ${card.rationale}` : 'Lecture content.'}\nCoverage: ${clock(view.coverage.start_s)}–${clock(view.coverage.end_s)}\n${card.citations.map(source).join('\n')}`,
    [...new Set(card.tags)].join(' ')])].map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
}
export function messageForError(error) {
  const text = String(error?.message ?? error);
  if (/401|invalid_token|unauthor|authentication|reconnect/i.test(text)) return 'Reconnect Video Moment Finder in your plugin settings, then retry. Your uploaded file remains selected here.';
  if (/402|insufficient|balance/i.test(text)) return 'Your available VMF allowance is insufficient. You can keep using evidence already in this conversation.';
  if (/404|not found/i.test(text)) return 'This video is unavailable to the connected account. Refresh the library or choose another video.';
  if (/transfer canceled/i.test(text)) return 'Transfer canceled. Processing has not started. Retry here when you are ready.';
  if (/network|fetch|cors/i.test(text)) return 'The transfer could not connect. Retry here; a completed upload is not transferred or indexed again.';
  return 'This action could not finish. Retry, or open VMF on the web for help.';
}
