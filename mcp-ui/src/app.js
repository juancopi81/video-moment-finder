import {App} from '@modelcontextprotocol/ext-apps';
import {clock, safeMediaUrl, selectedContext, cardsCsv, vectorMetrics, messageForError} from './core.js';
import './style.css';

const $ = id => document.getElementById(id);
const el = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
const button = (text, handler, cls = 'quiet') => {
  const node = el('button', text, cls);
  node.addEventListener('click', () => Promise.resolve().then(handler).catch(showError));
  return node;
};
const state = {videos: [], allowance: null, config: {}, selected: null, segments: new Map(), pendingTranscripts: new Set(), frames: new Map(), pendingFrames: new Set(), views: new Map(), active: 'study-guide', time: 0, resumeTime: null, segment: null, file: null, upload: null, uploadNotice: null, xhr: null, poll: null, busy: false, frameUrl: null, selection: 0};
const labels = {'study-guide': 'Study guide', flashcards: 'Flashcards', playground: 'Playground', presentation: 'Presentation', tutor: 'Tutor'};
const app = new App({name: 'Video Moment Finder', version: '0.3.0'}, {availableDisplayModes: ['fullscreen']}, {autoResize: true});
let connected = false, libraryLoaded = false, libraryPending = false;
let currentExperiment = null;

function notice(text = '') { $('notice').textContent = text; }
function showError(error) { notice(messageForError(error)); }
async function call(name, args = {}) {
  const result = await app.callServerTool({name, arguments: args});
  if (result.isError) throw new Error(result.content?.find(c => c.type === 'text')?.text ?? 'Tool failed');
  return result;
}
function theme(context) {
  if (context?.theme === 'dark' || context?.theme === 'light') document.documentElement.dataset.theme = context.theme;
}
function receive(result) {
  if (result.isError) { showError(new Error(result.content?.find(c => c.type === 'text')?.text ?? 'Tool failed')); return; }
  const data = result.structuredContent ?? {};
  const privateData = result._meta?.vmf ?? {};
  if (privateData.config) state.config = privateData.config;
  if (data.videos) { state.videos = data.videos; libraryLoaded = true; }
  if (data.allowance) state.allowance = data.allowance;
  renderLibrary(); updateUpload();
  if (data.video) {
    const changed = state.selected?.id !== data.video.id;
    if (changed) { clearTimeout(state.poll); state.segment = null; state.time = 0; clearFrame(); $('player').removeAttribute('src'); }
    state.selected = data.video;
    if (!state.videos.some(v => v.id === data.video.id)) state.videos.push(data.video);
    if (data.view) {
      [...(data.companion_views ?? []), data.view].forEach(view => state.views.set(`${view.video_id}:${view.kind}`, view));
      state.active = data.view.kind;
    }
    renderVideo(privateData.source_url);
  }
  const balance = state.allowance?.api_units_balance;
  $('balance').textContent = balance === undefined ? '' : `${balance.toLocaleString()} units available`;
  $('welcome-access').textContent = balance === 0 ? 'This account has no available units. You can still browse existing videos and reuse evidence already in your conversation.' : '';
  if (connected && data.video && !libraryLoaded) ensureLibrary();
}
async function ensureLibrary() {
  if (libraryPending || libraryLoaded || !state.selected) return;
  libraryPending = true;
  try { await refreshLibrary(); } catch (error) { showError(error); }
  finally { libraryPending = false; }
}
function renderLibrary() {
  const list = $('videos'); list.replaceChildren();
  const filter = $('filter').value.toLocaleLowerCase();
  const videos = state.videos.filter(v => (v.source_filename ?? v.youtube_url ?? v.id).toLocaleLowerCase().includes(filter));
  if (!videos.length) list.append(el('p', state.videos.length ? 'No matching videos.' : 'Your library is empty. Upload your first video to begin.', 'model-note'));
  for (const video of videos) {
    const row = button('', () => selectVideo(video));
    row.append(el('span', video.source_filename ?? (video.source_type === 'youtube' ? 'YouTube video' : 'Uploaded video')),
      el('small', `${video.status === 'ready' ? 'Ready to explore' : video.status} · ${new Date(video.created_at).toLocaleDateString()}`));
    row.setAttribute('aria-pressed', String(state.selected?.id === video.id));
    list.append(row);
  }
}
async function refreshLibrary() { receive(await call('open_workspace')); }
async function selectVideo(video) {
  const selection = ++state.selection;
  const result = await call('get_workspace_video', {video_id: video.id});
  if (selection !== state.selection) return;
  receive(result);
  if (['queued', 'processing'].includes(state.selected.status)) pollVideo(video.id);
}
function clearFrame() {
  if (state.frameUrl) URL.revokeObjectURL(state.frameUrl);
  state.frameUrl = null; $('frame-preview').hidden = true; $('frame').removeAttribute('src');
}
function renderVideo(sourceUrl) {
  $('welcome').hidden = true; $('video-workbench').hidden = false;
  $('video-title').textContent = state.selected.source_filename ?? 'Your video';
  $('video-status').textContent = state.selected.status === 'ready' ? 'Ready to explore' : state.selected.status === 'failed' ? 'Processing failed. Choose another video or open VMF for help.' : 'Processing your video. You can leave this view and return later.';
  const player = $('player');
  const media = safeMediaUrl(sourceUrl, state.config.media_origin);
  if (media && player.getAttribute('src') !== media) { state.resumeTime = state.time; player.src = media; player.load(); }
  if (!media && sourceUrl !== undefined) { state.resumeTime = null; player.removeAttribute('src'); player.load(); }
  player.hidden = !player.hasAttribute('src');
  $('media-state').textContent = player.hidden ? 'Retained source playback is unavailable here. Timestamps and existing evidence remain usable; open the source on VMF when available.' : 'Select a transcript line or source timestamp to return to this player.';
  $('refresh-media').hidden = !media;
  const ready = state.selected.status === 'ready';
  if (state.uploadNotice === state.selected.id && ['ready', 'failed'].includes(state.selected.status)) {
    notice(ready ? 'Your video is ready to explore.' : 'Video processing failed. Open VMF for help before trying another upload.');
    state.uploadNotice = null;
  }
  for (const id of ['explain', 'quiz', 'visual', 'inspect-frame']) $(id).disabled = !ready;
  $('load-transcript').disabled = !ready || state.pendingTranscripts.has(state.selected.id);
  $('transcript-cost').textContent = `Load once: ${state.config.transcript_units ?? '?'} unit(s). Reused here for this conversation.`;
  updateFrameButton();
  renderTranscript(); renderLibrary(); renderLearning();
}
function seek(time, segment = null) {
  if (!Number.isFinite(time) || time < 0 || time > 86400) return;
  state.time = time; state.segment = segment;
  if (state.resumeTime !== null) state.resumeTime = time;
  $('moment-time').value = Math.round(time);
  if ($('player').readyState >= 1 && Math.abs($('player').currentTime - time) > .001) $('player').currentTime = Math.min(time, $('player').duration || time);
  renderTranscript(); updateFrameButton();
}
function renderTranscript() {
  const root = $('transcript'); root.replaceChildren();
  const segments = state.segments.get(state.selected?.id);
  $('load-transcript').hidden = Boolean(segments);
  if (!segments) { root.append(el('p', 'Load the transcript to select the exact passage you want to explore.', 'model-note')); return; }
  if (!segments.length) root.append(el('p', 'No spoken transcript was returned. Ask ChatGPT to inspect a frame or use material you supply.', 'model-note'));
  segments.forEach(segment => {
    const row = button('', () => seek(segment.start_s, segment));
    row.append(el('span', clock(segment.start_s)), el('span', segment.text));
    if (state.segment?.segment_index === segment.segment_index) row.className = 'selected';
    root.append(row);
  });
}
async function loadTranscript() {
  const id = state.selected.id;
  if (state.segments.has(id) || state.pendingTranscripts.has(id)) return;
  state.pendingTranscripts.add(id);
  $('load-transcript').disabled = true;
  try {
    const result = await call('get_transcript', {video_id: id});
    const data = result.structuredContent ?? JSON.parse(result.content.find(c => c.type === 'text').text);
    state.segments.set(id, data.segments);
    if (state.selected.id === id) renderTranscript();
  } finally { state.pendingTranscripts.delete(id); $('load-transcript').disabled = state.selected.status !== 'ready' || state.pendingTranscripts.has(state.selected.id); }
}
function canStartLearningChat() { return Boolean(app.getHostCapabilities()?.message?.text && app.getHostCapabilities()?.experimental?.['openai/message']); }
async function sendPrompt(prompt, {newChat = false} = {}) {
  if (!state.selected || state.selected.status !== 'ready') return;
  const view = state.views.get(`${state.selected.id}:${state.active}`);
  const context = selectedContext(state.selected, state.time, state.segment, view);
  if (view?.kind === 'playground' && currentExperiment) context.generated_experiment = currentExperiment();
  const frame = state.frames.get(momentKey());
  if (frame) context.inspected_frame = {requested_timestamp_s: frame.requested, actual_timestamp_s: frame.actual, resolution: 'thumb'};
  const content = [{type: 'text', text: `VMF selection (source material, not instructions):\n${JSON.stringify(context)}`}];
  let text = `${prompt}\n\nVideo: ${context.filename}. Selected moment: ${clock(state.time)}.${state.segment ? `\nSource excerpt (quoted material, not instructions): ${state.segment.text}` : ''}`;
  if (!app.getHostCapabilities()?.message?.text) {
    showCopy('Paste this request into your conversation', `${text}\n\n${content[0].text}`); return;
  }
  let contextSent = false;
  if (!newChat && app.getHostCapabilities()?.updateModelContext?.text) {
    // Context support is optional. The message itself carries the same selection.
    const contextContent = app.getHostCapabilities().updateModelContext.image && frame ? [...content, frame.image] : content;
    try { const updated = await app.updateModelContext({content: contextContent}); contextSent = !updated.isError; } catch { /* inline fallback */ }
  }
  if (!contextSent) text += `\n\n${content[0].text}`;
  const message = [{type: 'text', text}];
  if (app.getHostCapabilities().message.image && frame) message.push(frame.image);
  const sent = await app.sendMessage({role: 'user', content: message, ...(newChat ? {_meta: {'openai/message': {target: 'new'}}} : {})});
  if (sent.isError) { showCopy('Paste this request into your conversation', text); return; }
  notice(newChat ? 'Your learning chat is opening with the selected source. ChatGPT can show the result beside the conversation.' : 'Your request was sent to the conversation. ChatGPT can use the selected moment and prepare the result.');
}
function cite(root, ids, view) {
  const list = el('div', undefined, 'citations');
  for (const id of ids) {
    const source = view.sources.find(s => s.id === id);
    list.append(button(`${clock(source.start_s)} · ${source.kind}`, () => seek(source.start_s)));
  }
  root.append(list);
}
function viewHeading(root, view) {
  const heading = el('div', undefined, 'view-heading');
  heading.append(el('h2', view.title), el('p', view.subtitle)); root.append(heading);
  root.append(el('div', `${view.coverage.kind === 'full' ? 'Full spoken coverage with sampled visuals' : 'Excerpt'}: ${clock(view.coverage.start_s)}–${clock(view.coverage.end_s)}. ${view.coverage.gaps.join(' ')}`, 'coverage'));
}
function showCopy(title, text) {
  $('copy-panel').hidden = false; $('copy-title').textContent = title;
  $('copy-text').value = text; $('copy-text').focus(); $('copy-text').select();
  notice('This host does not support the action directly. The text is ready to copy.');
}
async function download(name, data, type = 'application/json') {
  if (!app.getHostCapabilities()?.downloadFile) { showCopy(`Copy ${name} and save it locally`, data); return; }
  const saved = await app.downloadFile({contents: [{type: 'resource', resource: {uri: `file:///${name}`, mimeType: type, text: data}}]});
  if (saved.isError) notice('The download was canceled or unavailable. You can try again.');
}
function renderLearning() {
  const nav = $('workflows'); nav.replaceChildren();
  for (const [kind, label] of Object.entries(labels)) {
    const tab = button(label, () => { state.active = kind; renderLearning(); });
    if (kind === state.active) { tab.classList.add('active'); tab.setAttribute('aria-current', 'page'); }
    nav.append(tab);
  }
  const root = $('learning'); root.replaceChildren();
  const view = state.views.get(`${state.selected.id}:${state.active}`);
  if (!view) {
    root.append(el('h2', state.active === 'tutor' ? 'Reason it through, one question at a time.' : `Make a ${labels[state.active].toLowerCase()}`));
    root.append(el('p', state.active === 'tutor' ? 'Your tutor stays in the conversation, with the source and your experiments beside it.' : 'ChatGPT will retrieve the evidence it needs and prepare this view. Existing evidence can be reused across all five workflows.', 'model-note'));
    const prompts = {
      'study-guide': 'Create a study guide for the selected video or passage. Inspect the source evidence and show the guide beside the video.',
      flashcards: 'Create flashcards for the selected video or passage. Reuse available evidence, include reasoning for generated answers, and show the cards beside the video.',
      playground: 'Create a Playground for a source-supported rule in this video. Use a checked mathematical model if justified; otherwise compare reviewed assumptions. Show it beside the source.',
      presentation: 'Create a presentation from this video or passage, with cited slides and speaker notes. Show it beside the source and offer editable PowerPoint export when available.',
      tutor: 'Tutor me on this video or selected passage. Ask one focused question and wait for my answer. Use the tutor skill.',
    };
    const newChat = state.views.size === 0 && canStartLearningChat();
    root.append(button(state.active === 'tutor' ? `Start tutoring in ${newChat ? 'a new chat' : 'chat'}` : `Create ${labels[state.active].toLowerCase()}${newChat ? ' in a new chat' : ''}`, () => sendPrompt(prompts[state.active], {newChat}), 'primary'));
    return;
  }
  viewHeading(root, view);
  if (view.kind === 'study-guide') renderGuide(root, view);
  if (view.kind === 'flashcards') renderCards(root, view);
  if (view.kind === 'presentation') renderSlides(root, view);
  if (view.kind === 'playground') renderPlayground(root, view);
  const sources = el('details'); sources.append(el('summary', 'Evidence and coverage'));
  view.sources.forEach(s => { const item = el('div'); item.append(el('p', `${s.id}: ${s.summary}`, 'model-note')); cite(item, [s.id], view); sources.append(item); });
  root.append(sources, button('Download view data', () => download(`${view.kind}.json`, JSON.stringify(view, null, 2))));
}
function renderGuide(root, view) {
  view.sections.forEach(section => {
    const article = el('article', undefined, 'guide-section'); article.append(el('h3', section.title), el('span', section.origin === 'generated' ? 'Generated explanation or practice' : 'From the lecture', 'origin'));
    section.paragraphs.forEach(text => article.append(el('p', text)));
    if (section.bullets.length) { const ul = el('ul'); section.bullets.forEach(text => ul.append(el('li', text))); article.append(ul); }
    cite(article, section.citations, view); root.append(article);
  });
  root.append(button('Prepare an offline HTML copy', () => sendPrompt('Export the prepared study guide as self-contained offline HTML using the study-guide skill. Preserve citations and inspected visuals.')));
}
function renderCards(root, view) {
  let index = 0, revealed = false;
  const area = el('div'); root.append(area);
  const draw = () => {
    area.replaceChildren(); const card = view.cards[index];
    area.append(el('p', `Card ${index + 1} of ${view.cards.length}`, 'counter'));
    const face = el('article', undefined, 'answer-card'); face.append(el('h3', card.front));
    if (card.hint) { const hint = el('details'); hint.append(el('summary', 'Hint'), el('p', card.hint)); face.append(hint); }
    face.append(button(revealed ? 'Hide answer' : 'Reveal answer', () => { revealed = !revealed; draw(); }, 'primary'));
    if (revealed) { const answer = el('div', undefined, 'answer'); answer.append(el('p', card.back), el('span', card.origin === 'generated' ? 'Generated practice' : 'From the lecture', 'origin')); if (card.rationale) answer.append(el('p', card.rationale)); cite(answer, card.citations, view); face.append(answer); }
    area.append(face); const controls = el('div', undefined, 'button-row');
    const previous = button('Previous card', () => { index--; revealed = false; draw(); }); previous.disabled = index === 0;
    const next = button('Next card', () => { index++; revealed = false; draw(); }); next.disabled = index === view.cards.length - 1;
    controls.append(previous, next); area.append(controls);
  };
  draw(); root.append(button('Download flashcards CSV', () => download('flashcards.csv', cardsCsv(view), 'text/csv;charset=utf-8')));
}
function renderSlides(root, view) {
  let index = 0; const area = el('div'); root.append(area);
  const draw = () => {
    area.replaceChildren(); const slide = view.slides[index]; area.append(el('p', `Slide ${index + 1} of ${view.slides.length}`, 'counter'));
    const page = el('article', undefined, 'slide'); page.append(el('h3', slide.title)); const bullets = el('ul'); slide.bullets.forEach(text => bullets.append(el('li', text))); page.append(bullets, el('span', slide.origin === 'generated' ? 'Generated explanation or example' : 'From the source', 'origin')); cite(page, slide.citations, view); area.append(page);
    const controls = el('div', undefined, 'button-row');
    const prev = button('Previous slide', () => { index--; draw(); }); prev.disabled = index === 0;
    const next = button('Next slide', () => { index++; draw(); }); next.disabled = index === view.slides.length - 1;
    controls.append(prev, next); area.append(controls);
    const notes = el('details'); notes.append(el('summary', 'Speaker notes'), el('p', slide.notes, 'notes')); area.append(notes);
  };
  draw(); root.append(button('Prepare editable PowerPoint', () => sendPrompt('Prepare an editable PowerPoint from these slides, with cited speaker notes. If export is unavailable, provide an HTML preview and outline.')));
}
function renderPlayground(root, view) {
  const p = view.playground;
  root.append(el('h3', p.question));
  if (p.model === 'dot-product-2d') renderVectors(root, view);
  else renderCases(root, view);
  root.append(button('Ask about this experiment', () => sendPrompt('Help me understand my current Playground inputs and prediction. Use the experiment as generated practice, separate from the lecture evidence. Start with one useful explanation or question.')));
  root.append(el('p', `${p.extension_note} ${p.model_limit}`, 'model-note'));
}
function renderVectors(root, view) {
  const p = view.playground; let v = [...p.initial.v], w = [...p.initial.w], pinned = null, drag = null;
  const controls = el('div', undefined, 'vector-controls'), inputs = [];
  ['vₓ', 'vᵧ', 'wₓ', 'wᵧ'].forEach((name, i) => {
    const label = el('label', name), input = el('input'); input.type = 'number'; input.min = '-5'; input.max = '5'; input.step = '.25';
    input.addEventListener('input', () => { const x = Number(input.value); if (!Number.isFinite(x) || x < -5 || x > 5 || input.value === '') return; (i < 2 ? v : w)[i % 2] = x; feedback.textContent = 'Inputs changed. Compare the updated result or test a new prediction.'; draw(); });
    label.append(input); controls.append(label); inputs.push(input);
  });
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('viewBox', '0 0 320 320'); svg.classList.add('vector-plot'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Two vectors. Use the coordinate controls to change them with the keyboard.');
  const shape = (tag, attrs) => { const node = document.createElementNS(svg.namespaceURI, tag); for (const [k, value] of Object.entries(attrs)) node.setAttribute(k, value); return node; };
  for (let n = 20; n <= 300; n += 28) svg.append(shape('line', {x1: n, y1: 20, x2: n, y2: 300, stroke: '#dce4ef'}), shape('line', {x1: 20, y1: n, x2: 300, y2: n, stroke: '#dce4ef'}));
  svg.append(shape('line', {x1: 160, y1: 20, x2: 160, y2: 300, stroke: '#7f90a8'}), shape('line', {x1: 20, y1: 160, x2: 300, y2: 160, stroke: '#7f90a8'}));
  const arrows = [v, w].map((_, i) => {
    const color = i ? '#dd7022' : '#1e55ce'; const line = shape('line', {x1: 160, y1: 160, stroke: color, 'stroke-width': 4}); const dot = shape('circle', {r: 9, fill: color, cursor: 'grab'});
    dot.addEventListener('pointerdown', e => { drag = i; svg.setPointerCapture(e.pointerId); e.preventDefault(); }); svg.append(line, dot); return [line, dot];
  });
  svg.addEventListener('pointermove', e => { if (drag === null) return; const rect = svg.getBoundingClientRect(); const x = Math.round(Math.max(-5, Math.min(5, ((e.clientX - rect.left) / rect.width * 320 - 160) / 28)) * 4) / 4; const y = Math.round(Math.max(-5, Math.min(5, (160 - (e.clientY - rect.top) / rect.height * 320) / 28)) * 4) / 4; if (drag === 0) v = [x, y]; else w = [x, y]; feedback.textContent = 'Inputs changed. Compare the updated result or test a new prediction.'; draw(); });
  svg.addEventListener('pointerup', () => { drag = null; }); svg.addEventListener('pointercancel', () => { drag = null; });
  const metrics = el('div', undefined, 'metrics'), comparison = el('div', undefined, 'comparison'); comparison.hidden = true;
  const prediction = el('label', 'Predict the dot product before revealing it', 'prediction'), input = el('input'), feedback = el('p'); input.type = 'number'; input.step = 'any'; prediction.append(input);
  let tested = false;
  currentExperiment = () => ({model: p.model, generated_model_not_observation: true, v: [...v], w: [...w], metrics: vectorMetrics(v, w), prediction: input.value, result_revealed: tested, pinned, model_limit: p.model_limit});
  const pretty = x => x === null ? 'Undefined' : Number(x.toFixed(3)).toString();
  const draw = () => {
    [...v, ...w].forEach((x, i) => { if (document.activeElement !== inputs[i]) inputs[i].value = x; });
    [v, w].forEach((xy, i) => { const [line, dot] = arrows[i]; const x = 160 + 28 * xy[0], y = 160 - 28 * xy[1]; line.setAttribute('x2', x); line.setAttribute('y2', y); dot.setAttribute('cx', x); dot.setAttribute('cy', y); });
    const m = vectorMetrics(v, w); metrics.replaceChildren();
    for (const [name, value] of [['Dot product', tested ? pretty(m.dot) : 'Predict first'], ['Angle', tested ? `${pretty(m.angle)}${m.angle === null ? '' : '°'}` : '—'], ['Projection on v', tested ? m.projection?.map(pretty).join(', ') ?? 'Undefined' : '—']]) { const item = el('div'); item.append(el('span', name), el('strong', value)); metrics.append(item); }
    if (pinned) { comparison.hidden = false; comparison.textContent = `Pinned: v=(${pinned.v}), w=(${pinned.w}), dot=${pretty(pinned.dot)}\nCurrent: v=(${v}), w=(${w})${tested ? `, dot=${pretty(m.dot)}` : ''}`; }
  };
  const actions = el('div', undefined, 'button-row');
  actions.append(button('Test prediction', () => { tested = true; const dot = vectorMetrics(v, w).dot; feedback.textContent = input.value === '' ? 'Result revealed. Change one coordinate and predict again.' : Math.abs(Number(input.value) - dot) < 1e-8 ? 'Your prediction matches this model.' : `The model gives ${pretty(dot)}. Compare the two coordinate products.`; draw(); }, 'primary'),
    button('Pin comparison', () => { pinned = {v: [...v], w: [...w], dot: vectorMetrics(v, w).dot}; draw(); }),
    button('Zero-vector example', () => { v = [0, 0]; tested = true; feedback.textContent = 'A zero dot product does not guarantee a right angle. With a zero vector, the angle is undefined.'; draw(); }),
    button('Reset', () => { v = [...p.initial.v]; w = [...p.initial.w]; tested = false; feedback.textContent = ''; draw(); }),
    button('Download observation', () => download('playground-observation.json', JSON.stringify({generated_model: p.model, v, w, metrics: vectorMetrics(v, w), prediction: input.value, pinned, model_limit: p.model_limit, sources: view.sources}, null, 2))));
  root.append(controls, metrics, svg, prediction, actions, feedback, comparison); cite(root, p.citations, view); draw();
}
function renderCases(root, view) {
  const p = view.playground; const chosen = Object.fromEntries(p.controls.map(c => [c.id, c.options[0].id])); let pinned = null, revealed = false;
  const controls = el('div', undefined, 'case-controls'), output = el('article', undefined, 'answer-card'), comparison = el('div', undefined, 'comparison'); comparison.hidden = true;
  const prediction = el('label', 'Predict what changes under these assumptions', 'prediction'), input = el('input'); prediction.append(input);
  const current = () => p.cases.find(c => Object.entries(chosen).every(([k, v]) => c.when[k] === v));
  currentExperiment = () => ({model: p.model, generated_model_not_observation: true, selected: {...chosen}, prediction: input.value, result_revealed: revealed, reviewed_case: current(), pinned, model_limit: p.model_limit});
  const draw = () => { const c = current(); output.replaceChildren(el('h3', c.title), el('p', `Assumptions: ${c.assumptions.join(' ')}`, 'model-note')); if (revealed) { output.append(el('p', c.outcome), el('p', c.explanation), el('p', `Unknowns: ${c.unknowns.join(' ') || 'None stated.'}`, 'model-note')); cite(output, c.citations, view); } else output.append(el('p', 'Make a prediction, then reveal the reviewed case.')); if (pinned) { comparison.hidden = false; comparison.textContent = `Pinned: ${pinned.title}\n${pinned.outcome}\nCurrent: ${c.title}${revealed ? `\n${c.outcome}` : ''}`; } };
  p.controls.forEach(control => { const label = el('label', control.label), select = el('select'); control.options.forEach(o => { const option = el('option', o.label); option.value = o.id; select.append(option); }); select.addEventListener('change', () => { chosen[control.id] = select.value; revealed = false; draw(); }); label.append(select); controls.append(label); });
  const actions = el('div', undefined, 'button-row'); actions.append(
    button('Test prediction', () => { revealed = true; draw(); }, 'primary'),
    button('Pin comparison', () => { pinned = structuredClone(current()); revealed = true; draw(); }),
    button('Reset', () => { p.controls.forEach((c, i) => { chosen[c.id] = c.options[0].id; controls.querySelectorAll('select')[i].value = c.options[0].id; }); revealed = false; draw(); }),
    button('Download observation', () => download('reviewed-case-observation.json', JSON.stringify({generated_model: p.model, selected: chosen, prediction: input.value, reviewed_case: current(), pinned, model_limit: p.model_limit, sources: view.sources}, null, 2))));
  root.append(controls, prediction, actions, output, comparison); draw();
}
function updateUpload() {
  const cost = state.allowance?.unit_cost_index_video;
  $('upload-cost').textContent = cost === undefined ? 'Refresh the library to check the current processing cost and available units.' : `Processing this video uses ${cost} units. Uploading the file alone does not start processing.`;
  $('start-upload').textContent = state.upload?.uploaded ? 'Retry processing this upload' : `Upload and process${cost === undefined ? '' : ` · ${cost} units`}`;
  $('start-upload').disabled = state.busy || !state.file || !$('rights').checked || cost === undefined || (!state.upload?.completionAttempted && state.allowance.api_units_balance < cost);
  $('file').disabled = state.busy; $('rights').disabled = state.busy; $('close-upload').disabled = state.busy;
}
function putFile(ticket, file) {
  const url = safeMediaUrl(ticket.upload_url, state.config.media_origin);
  if (!url) throw new Error('Upload storage origin is unavailable');
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); state.xhr = xhr;
    xhr.open('PUT', url); xhr.withCredentials = false; xhr.setRequestHeader('Content-Type', mimeType(file.name));
    xhr.upload.onprogress = e => { if (e.lengthComputable) $('upload-progress').value = e.loaded / e.total * 100; };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else { if ([401, 403].includes(xhr.status)) state.upload = null; reject(new Error('Network transfer failed')); }
    };
    xhr.onerror = () => reject(new Error('Network transfer failed'));
    xhr.onabort = () => reject(new Error('Transfer canceled'));
    xhr.send(file);
  });
}
function mimeType(name) { return {mp4: 'video/mp4', mov: 'video/quicktime', webm: 'video/webm'}[name.split('.').at(-1).toLowerCase()]; }
async function uploadVideo() {
  if (state.busy || !state.file || !$('rights').checked) return;
  state.busy = true; updateUpload(); notice('');
  const file = state.file;
  const approvedCost = state.allowance.unit_cost_index_video;
  try {
    // Refresh authoritative tariffs before starting, without indexing or retrieval.
    receive(await call('open_workspace'));
    if (state.allowance.unit_cost_index_video !== approvedCost) { notice('The processing cost changed. Review the updated cost, then choose Upload and process again.'); return; }
    if (state.upload?.completionAttempted) {
      // A lost response may have followed a successful charged completion.
      // Reconcile the existing UUID before retrying or rejecting a lower balance.
      const status = await app.callServerTool({name: 'get_video_status', arguments: {video_id: state.upload.video_id}});
      if (!status.isError && status.structuredContent) { await finishUpload(status.structuredContent); return; }
      if (status.isError && !/404|not found/i.test(status.content?.find(c => c.type === 'text')?.text ?? '')) throw new Error('Unable to reconcile upload. Reconnect and retry.');
    }
    if (state.allowance.api_units_balance < state.allowance.unit_cost_index_video) throw new Error('Insufficient allowance');
    if (!state.upload) {
      const result = await call('upload_video', {action: 'start', filename: file.name, content_type: mimeType(file.name)});
      const ticket = result.structuredContent; if (!ticket?.upload_url || !ticket?.video_id) throw new Error('Upload preparation failed');
      state.upload = {...ticket, uploaded: false};
    }
    if (!state.upload.uploaded) {
      $('upload-state').textContent = 'Transferring your file…'; $('upload-progress').hidden = false; $('cancel-upload').hidden = false;
      await putFile(state.upload, file); state.upload.uploaded = true;
    }
    $('cancel-upload').hidden = true; $('upload-state').textContent = 'Starting video processing…';
    state.upload.completionAttempted = true;
    const result = await call('upload_video', {action: 'complete', filename: file.name, video_id: state.upload.video_id});
    const video = result.structuredContent?.video;
    if (!video) throw new Error('Processing acknowledgment missing');
    await finishUpload(video);
  } finally { state.busy = false; state.xhr = null; $('cancel-upload').hidden = true; updateUpload(); }
}
async function finishUpload(video) {
  state.upload = null; state.file = null; $('file').value = ''; $('rights').checked = false; $('upload').hidden = true;
  state.uploadNotice = video.id;
  await refreshLibrary(); await selectVideo(video);
  notice(state.selected.status === 'ready' ? 'Your video is ready to explore.' : 'Your video is processing. Status checks use no units. You can return later.');
}
function pollVideo(id) {
  clearTimeout(state.poll);
  state.poll = setTimeout(async () => {
    if (document.hidden || state.selected?.id !== id) return;
    try {
      const result = await call('get_video_status', {video_id: id});
      const video = result.structuredContent;
      if (state.selected?.id !== id) return;
      state.selected = video; const i = state.videos.findIndex(v => v.id === id); if (i >= 0) state.videos[i] = video;
      if (['queued', 'processing'].includes(video.status)) { renderLibrary(); pollVideo(id); }
      else await selectVideo(video);
    } catch (error) { showError(error); }
  }, 15000);
}
function momentKey() { return `${state.selected?.id}:${Math.round(state.time * 1000)}`; }
function updateFrameButton() {
  $('inspect-frame').disabled = state.selected?.status !== 'ready' || state.pendingFrames.has(state.selected.id);
  $('inspect-frame').textContent = `Show this frame · ${state.frames.has(momentKey()) ? 'cached' : `${state.config.frame_thumb_units ?? '?'} unit(s)`}`;
}
async function inspectFrame() {
  const id = state.selected.id, time = state.time, key = momentKey();
  if (state.pendingFrames.has(id)) return;
  state.pendingFrames.add(id); updateFrameButton();
  try {
    let frame = state.frames.get(key);
    const cached = Boolean(frame);
    if (!frame) {
      const result = await call('get_frames', {video_id: id, timestamps: [time], resolution: 'thumb'});
      const image = result.content?.find(c => c.type === 'image');
      if (!image || !['image/png', 'image/jpeg'].includes(image.mimeType)) { notice('No usable image was returned for this moment.'); return; }
      let summary = {}; try { summary = JSON.parse(result.content.find(c => c.type === 'text').text); } catch { /* conservative requested-time label */ }
      frame = {image, requested: time, actual: summary.frames?.find(f => f.image_index === 0)?.actual_timestamp_s};
      if (state.frames.size >= 5) state.frames.delete(state.frames.keys().next().value);
      state.frames.set(key, frame);
    }
    if (state.selected.id !== id) return;
    clearFrame(); const bytes = Uint8Array.from(atob(frame.image.data), c => c.charCodeAt(0));
    state.frameUrl = URL.createObjectURL(new Blob([bytes], {type: frame.image.mimeType})); $('frame').src = state.frameUrl;
    $('frame-caption').textContent = `Source thumbnail${frame.actual === undefined ? ` requested near ${clock(time)}` : ` at ${clock(frame.actual)}`}. ${cached ? 'Reused from this view.' : `Frame retrieval uses ${state.config.frame_thumb_units} unit(s).`}`;
    $('frame-preview').hidden = false;
  } finally { state.pendingFrames.delete(id); updateFrameButton(); }
}
function on(id, handler, event = 'click') { $(id).addEventListener(event, () => Promise.resolve().then(handler).catch(showError)); }
on('refresh', refreshLibrary); on('filter', renderLibrary, 'input');
for (const id of ['add', 'welcome-upload']) on(id, () => { $('upload').hidden = false; $('upload').scrollIntoView({block: 'start'}); $('file').focus(); });
on('close-upload', () => { if (!state.busy) $('upload').hidden = true; });
on('file', () => {
  if (state.busy) return;
  const file = $('file').files[0]; state.upload = null; state.file = null;
  if (file && (!mimeType(file.name) || file.size === 0 || !state.config.max_upload_bytes || file.size > state.config.max_upload_bytes || file.name.length > 200)) { updateUpload(); notice('Choose a nonempty MP4, MOV or WebM within the VMF upload size limit. Use a filename of at most 200 characters.'); return; }
  state.file = file; $('file-details').textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : ''; $('upload-state').textContent = ''; $('upload-progress').value = 0; notice(''); updateUpload();
}, 'change');
on('rights', updateUpload, 'change'); on('start-upload', uploadVideo); on('cancel-upload', () => state.xhr?.abort());
on('close-copy', () => { $('copy-panel').hidden = true; });
on('youtube-help', () => app.openLink({url: 'https://support.google.com/youtube/answer/56100?hl=en'}));
on('web-source', () => app.openLink({url: `https://www.videomomentfinder.com/video/${state.selected.id}?t=${state.time}`}));
on('refresh-media', () => selectVideo(state.selected)); on('load-transcript', loadTranscript); on('seek', () => seek(Number($('moment-time').value)));
on('explain', () => sendPrompt('Explain the selected moment. Inspect the relevant source frame if it matters.'));
on('quiz', () => sendPrompt('Ask me one question about the selected moment and wait for my answer.'));
on('visual', () => sendPrompt('Make the selected idea visual. Choose a source-grounded study guide or Playground, and show it beside the video.'));
on('inspect-frame', inspectFrame);
$('player').addEventListener('loadedmetadata', () => { const time = state.resumeTime ?? state.time; state.resumeTime = null; seek(time, state.segment); });
$('player').addEventListener('timeupdate', () => { if (state.resumeTime !== null) return; state.time = $('player').currentTime; if (state.segment && (state.time < state.segment.start_s || state.time > state.segment.end_s)) { state.segment = null; renderTranscript(); } if (document.activeElement !== $('moment-time')) $('moment-time').value = Math.floor(state.time); updateFrameButton(); });
$('player').addEventListener('error', () => { $('media-state').textContent = 'Playback failed or its link expired. Refresh the playback link; evidence already retrieved remains usable.'; $('refresh-media').hidden = false; });
document.addEventListener('visibilitychange', () => { if (!document.hidden && ['queued', 'processing'].includes(state.selected?.status)) pollVideo(state.selected.id); });
window.addEventListener('pagehide', () => { clearTimeout(state.poll); state.xhr?.abort(); clearFrame(); });
app.ontoolresult = receive;
app.onhostcontextchanged = theme;
app.onerror = showError;
await app.connect();
connected = true;
theme(app.getHostContext());
if (state.selected && !libraryLoaded) ensureLibrary();
if (app.getHostContext()?.displayMode === 'inline' && app.getHostContext()?.availableDisplayModes?.includes('fullscreen')) await app.requestDisplayMode({mode: 'fullscreen'});
