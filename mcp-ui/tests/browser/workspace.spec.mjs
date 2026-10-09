import {test, expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {guide, reviewedCases, toolResult, video} from '../fixtures.js';

let media, referenceImage;
test.beforeAll(async () => { media = await readFile(new URL('../lesson.webm', import.meta.url)); referenceImage = await readFile(new URL('../reference-frame.png', import.meta.url)); });
async function open(page, query = '') {
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.route('https://storage.example.test/**', async route => {
    const headers = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'PUT, GET, HEAD, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type', 'Accept-Ranges': 'bytes'};
    if (route.request().url().includes('/reference-frame.png')) { await route.fulfill({status: 200, headers, contentType: 'image/png', body: referenceImage}); return; }
    const range = route.request().headers().range?.match(/^bytes=(\d+)-(\d*)$/);
    const isMedia = route.request().method() === 'GET';
    const start = isMedia && range ? Number(range[1]) : 0;
    const end = isMedia && range?.[2] ? Math.min(Number(range[2]), media.length - 1) : media.length - 1;
    if (isMedia && range) headers['Content-Range'] = `bytes ${start}-${end}/${media.length}`;
    await route.fulfill({status: isMedia && range ? 206 : 200, headers, contentType: isMedia ? 'video/webm' : 'text/plain', body: isMedia ? media.subarray(start, end + 1) : ''});
  });
  await page.goto('/' + query);
  const ui = page.frameLocator('#workspace');
  await expect(ui.getByRole('button', {name: 'Upload video', exact: true})).toBeVisible();
  await expect(ui.locator('#balance')).toContainText('units');
  return {ui, errors};
}
const calls = (page, name) => page.evaluate(name => window.fixtureHost.calls.filter(c => c.name === name), name);

test('text search finds spoken and visual candidates, seeks and hands off clean context; repeats are cached', async ({page}) => {
  const {ui, errors} = await open(page, '?injection');
  await ui.getByLabel('Describe what you want to find', {exact: true}).fill('the zero-vector example');
  await ui.locator('#moment-query').press('Enter');
  await expect(ui.locator('#search-results .search-match')).toHaveCount(2);
  expect(await calls(page, 'search_video')).toHaveLength(1);
  expect((await calls(page, 'search_video'))[0].args).toEqual({video_id: video.id, query_text: 'the zero-vector example', limit: 5});
  await expect(ui.locator('#balance')).toContainText('599 units');
  await expect(ui.locator('#search-results')).toContainText('<img src=x');
  expect(await ui.locator('body').evaluate(() => window.badSearch)).toBeUndefined();
  await ui.getByRole('button', {name: 'Go to 0:06 spoken match 1'}).click();
  await expect.poll(() => ui.locator('#player').evaluate(p => p.currentTime)).toBe(6);
  await ui.getByRole('button', {name: 'Explain this moment', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.contexts.length)).toBe(1);
  const context = await page.evaluate(() => window.fixtureHost.contexts[0].content[0].text);
  expect(context).toContain('"candidate_requires_source_verification":true');
  expect(context).toContain('Quoted spoken text.');
  expect(context).not.toContain('fixture=temporary');
  await ui.locator('#player').evaluate(p => { p.currentTime = 10; });
  await expect(ui.getByRole('button', {name: 'Go to 0:06 spoken match 1'})).toHaveAttribute('aria-pressed', 'false');
  await ui.getByRole('button', {name: 'Explain this moment', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.contexts.length)).toBe(2);
  expect(await page.evaluate(() => window.fixtureHost.contexts[1].content[0].text)).not.toContain('search_candidate');
  await ui.locator('#search-filter').selectOption('visual');
  await expect(ui.locator('#search-results .search-match')).toHaveCount(1);
  await expect(ui.locator('#search-results img')).toBeVisible();
  await ui.locator('#run-search').click();
  await expect(ui.locator('#search-state')).toContainText('No new search units');
  expect(await calls(page, 'search_video')).toHaveLength(1);
  expect(errors).toEqual([]);
});

test('image selection is local; explicit search sends a bounded JPEG and finds source frames', async ({page}) => {
  const {ui, errors} = await open(page);
  await ui.getByRole('button', {name: 'Image', exact: true}).click();
  await ui.locator('#search-image-file').setInputFiles({name: 'my-reference.png', mimeType: 'image/png', buffer: referenceImage});
  await expect(ui.locator('#search-state')).toContainText('Nothing has been sent');
  await expect(ui.locator('#search-image')).toBeVisible();
  expect(await calls(page, 'search_video_image')).toHaveLength(0);
  await expect(ui.locator('#run-search')).toContainText('1 unit');
  await ui.locator('#run-search').click();
  await expect(ui.locator('#search-results .search-match')).toHaveCount(1);
  const [request] = await calls(page, 'search_video_image');
  expect(request.args.video_id).toBe(video.id);
  const bytes = Buffer.from(request.args.image_base64, 'base64');
  expect(bytes.length).toBeLessThanOrEqual(524288);
  expect(bytes.subarray(0, 2).toString('hex')).toBe('ffd8');
  expect(request.args).not.toHaveProperty('image_url');
  await ui.getByRole('button', {name: 'Go to 0:06 visual match 1'}).click();
  await expect.poll(() => ui.locator('#player').evaluate(p => p.currentTime)).toBe(6);
  await ui.getByRole('button', {name: 'Quiz me', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.messages.length)).toBe(1);
  expect(await page.evaluate(() => window.fixtureHost.messages[0].content[0].text)).not.toContain('fixture=temporary');
  await ui.locator('#run-search').click();
  expect(await calls(page, 'search_video_image')).toHaveLength(1);
  await ui.getByRole('button', {name: 'Remove image'}).click();
  await expect(ui.locator('#run-search')).toBeDisabled();
  await expect(ui.locator('#search-image-preview')).toBeHidden();
  expect(errors).toEqual([]);
});

test('invalid or oversized image selection never runs a query', async ({page}) => {
  const {ui} = await open(page);
  await ui.getByRole('button', {name: 'Image', exact: true}).click();
  await ui.locator('#search-image-file').setInputFiles({name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('invalid')});
  await expect(ui.locator('#search-state')).toContainText('could not be opened');
  await expect(ui.locator('#run-search')).toBeDisabled();
  await ui.locator('#search-image-file').setInputFiles({name: 'large.png', mimeType: 'image/png', buffer: Buffer.alloc(10 * 1024 * 1024 + 1)});
  await expect(ui.locator('#search-state')).toContainText('at most 10 MB');
  expect(await calls(page, 'search_video_image')).toHaveLength(0);
});

test('empty results and failures are explicit; no automatic paid retry occurs', async ({page}) => {
  const {ui} = await open(page);
  await page.evaluate(() => { window.fixtureHost.noMatches = true; });
  await ui.locator('#moment-query').fill('a cat');
  await ui.locator('#run-search').click();
  await expect(ui.locator('#search-results')).toContainText('No matches returned');
  await page.evaluate(() => { window.fixtureHost.failures.search = 1; });
  await ui.locator('#moment-query').fill('a different scene');
  await ui.locator('#run-search').click();
  await expect(ui.locator('#search-state')).toContainText('No automatic retry');
  expect(await calls(page, 'search_video')).toHaveLength(2);
});

test('a changed tariff requires another click; zero allowance prevents a new query', async ({page}) => {
  const {ui} = await open(page);
  await page.evaluate(() => { window.fixtureHost.searchCost = 3; });
  await ui.locator('#moment-query').fill('zero-vector example');
  await ui.locator('#run-search').click();
  await expect(ui.locator('#search-state')).toContainText('cost changed');
  await expect(ui.locator('#run-search')).toContainText('3 units');
  expect(await calls(page, 'search_video')).toHaveLength(0);
  await page.evaluate(() => { window.fixtureHost.balance = 0; });
  await ui.locator('#run-search').click();
  await expect(ui.locator('#search-state')).toContainText('insufficient');
  expect(await calls(page, 'search_video')).toHaveLength(0);
});

test('a late query result never appears under a different selected video', async ({page}) => {
  const {ui} = await open(page);
  await page.evaluate(() => { window.fixtureHost.delays.search_video = 500; });
  await ui.locator('#moment-query').fill('zero-vector example');
  await ui.locator('#run-search').click();
  await expect.poll(() => calls(page, 'search_video')).toHaveLength(1);
  await ui.getByRole('button', {name: /Second lesson.mp4/}).click();
  await expect(ui.locator('#video-title')).toHaveText('Second lesson.mp4');
  await expect(ui.locator('#search-results-panel')).toBeHidden();
  await expect.poll(() => ui.locator('#run-search').textContent()).not.toContain('Finding moments');
  await expect(ui.locator('#search-results-panel')).toBeHidden();
  await expect(ui.locator('#moment-query')).toHaveValue('');
});

test('first learning request opens a full chat with its own source context; prepared views continue in place', async ({page}) => {
  const {ui, errors} = await open(page, '?library&new-chat');
  await ui.locator('#videos button').first().click();
  await ui.getByRole('button', {name: 'Load transcript', exact: true}).click();
  await ui.locator('#transcript button').nth(1).click();
  await ui.getByRole('button', {name: /Show this frame/}).click();
  await ui.getByRole('button', {name: 'Create study guide in a new chat', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.messages.length)).toBe(1);
  const first = await page.evaluate(() => window.fixtureHost.messages[0]);
  expect(first._meta['openai/message']).toEqual({target: 'new'});
  expect(first.content[0].text).toContain(`"video_id":"${video.id}"`);
  expect(first.content[0].text).toContain('"timestamp_s":6');
  expect(first.content[0].text).toContain('"actual_timestamp_s":6');
  expect(first.content[0].text).not.toContain('fixture=temporary');
  expect(first.content[1].type).toBe('image');
  // The old conversation's context must not be the only source of the new chat's selection.
  expect(await page.evaluate(() => window.fixtureHost.contexts)).toEqual([]);
  await page.evaluate(result => window.fixtureHost.push(result), toolResult({video, view: guide}));
  await ui.getByRole('button', {name: 'Tutor', exact: true}).click();
  await ui.getByRole('button', {name: 'Start tutoring in chat', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.messages.length)).toBe(2);
  expect(await page.evaluate(() => window.fixtureHost.messages[1]._meta)).toBeUndefined();
  expect(await page.evaluate(() => window.fixtureHost.contexts.length)).toBe(1);
  expect(errors).toEqual([]);
});

test('hosts without new-chat support keep the active conversation flow', async ({page}) => {
  const {ui, errors} = await open(page, '?library');
  await ui.locator('#videos button').first().click();
  await ui.getByRole('button', {name: 'Create study guide', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.messages.length)).toBe(1);
  expect(await page.evaluate(() => window.fixtureHost.messages[0]._meta)).toBeUndefined();
  expect(await page.evaluate(() => window.fixtureHost.contexts.length)).toBe(1);
  expect(errors).toEqual([]);
});

test('a delayed host module still receives the component initialization', async ({page}) => {
  let releaseModule;
  const moduleGate = new Promise(resolve => { releaseModule = resolve; });
  let moduleRequested;
  const requested = new Promise(resolve => { moduleRequested = resolve; });
  await page.route('**/fixtures.js', async route => {
    moduleRequested();
    await moduleGate;
    await route.continue();
  });
  await page.goto('/?no-media', {waitUntil: 'commit'});
  await requested;
  // Without the gate, the iframe boots before its parent can answer the SDK.
  await expect(page.locator('#workspace')).not.toHaveAttribute('src', '/workspace.html');
  releaseModule();
  const ui = page.frameLocator('#workspace');
  await expect(ui.locator('#balance')).toContainText('600 units available');
  expect(await calls(page, 'open_workspace')).toHaveLength(0);
});

test('opening a generated view directly enriches its missing library exactly once', async ({page}) => {
  const {ui, errors} = await open(page, '?render-only');
  await expect(ui.locator('#videos button')).toHaveCount(2);
  await expect(ui.getByRole('heading', {name: 'Dot products, made tangible'})).toBeVisible();
  expect(await calls(page, 'open_workspace')).toHaveLength(1);
  expect(errors).toEqual([]);
});

test('an initial authorization error explains reconnecting and never attempts ingestion', async ({page}) => {
  await page.goto('/?unauthorized');
  const ui = page.frameLocator('#workspace');
  await expect(ui.locator('#notice')).toContainText('Reconnect Video Moment Finder');
  expect(await calls(page, 'upload_video')).toHaveLength(0);
  expect(await calls(page, 'open_workspace')).toHaveLength(0);
});
async function chooseUpload(ui) {
  await ui.getByRole('button', {name: 'Upload video', exact: true}).click();
  await ui.locator('#file').setInputFiles({name: 'My lesson.mp4', mimeType: 'video/mp4', buffer: Buffer.from('fixture transfer bytes')});
  await expect(ui.locator('#start-upload')).toBeDisabled();
  await ui.getByRole('checkbox').check();
  await expect(ui.locator('#start-upload')).toBeEnabled();
}

test('initial result is reused; source markup is inert; exact selection reaches chat without media URLs', async ({page}) => {
  const {ui, errors} = await open(page, '?injection');
  await expect(ui.getByRole('heading', {name: 'Dot products, made tangible'})).toBeVisible();
  expect(await calls(page, 'open_workspace')).toHaveLength(0);
  await ui.getByRole('button', {name: 'Load transcript', exact: true}).click();
  await ui.locator('#transcript button').nth(1).click();
  await ui.getByRole('button', {name: 'Explain this moment'}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.messages.length)).toBe(1);
  const [context, message] = await page.evaluate(() => [window.fixtureHost.contexts[0], window.fixtureHost.messages[0]]);
  expect(context.content[0].text).toContain('"timestamp_s":6');
  expect(message.content[0].text).toContain('Source markup stays text.');
  expect(message.content[0].text).not.toContain('fixture=temporary');
  expect(await ui.locator('body').evaluate(() => window.badSource)).toBeUndefined();
  await ui.getByRole('button', {name: /Show this frame/}).click();
  await expect(ui.locator('#frame')).toBeVisible();
  await expect(ui.locator('#frame-caption')).toContainText('at 0:06');
  await ui.getByRole('button', {name: /Show this frame/}).click();
  expect(await calls(page, 'get_frames')).toHaveLength(1);
  await ui.getByRole('button', {name: 'Explain this moment'}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.messages.length)).toBe(2);
  expect(await page.evaluate(() => window.fixtureHost.messages[1].content[1].type)).toBe('image');
  expect(await calls(page, 'get_transcript')).toHaveLength(1);
  await ui.locator('#refresh-media').click();
  await expect.poll(() => ui.locator('#player').evaluate(player => player.currentTime)).toBe(6);
  expect(errors).toEqual([]);
});

test('cards reveal and advance; CSV goes through the host with cited answers', async ({page}) => {
  const {ui, errors} = await open(page);
  await ui.getByRole('button', {name: 'Flashcards', exact: true}).click();
  await expect(ui.locator('.answer')).toHaveCount(0);
  await ui.getByRole('button', {name: 'Reveal answer', exact: true}).click();
  await expect(ui.locator('.answer')).toContainText('3 × 2 + 4 × 0 = 6');
  await ui.getByRole('button', {name: 'Next card'}).click();
  await expect(ui.locator('.answer')).toHaveCount(0);
  await expect(ui.locator('.counter')).toHaveText('Card 2 of 2');
  await ui.getByRole('button', {name: 'Download flashcards CSV'}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.downloads.length)).toBe(1);
  const resource = await page.evaluate(() => window.fixtureHost.downloads[0].contents[0].resource);
  expect(resource.uri).toBe('file:///flashcards.csv');
  expect(resource.text).toContain('https://www.videomomentfinder.com/video/');
  expect(resource.text).toContain('Generated practice.');
  expect(resource.text).not.toContain('fixture=temporary');
  expect(errors).toEqual([]);
});

test('Playground computes changes, preserves comparisons and handles zero and finite-case boundaries', async ({page}) => {
  const {ui, errors} = await open(page);
  await ui.getByRole('button', {name: 'Playground', exact: true}).click();
  await ui.getByLabel('Predict the dot product').fill('6');
  await ui.getByRole('button', {name: 'Test prediction'}).click();
  await expect(ui.locator('.metrics')).toContainText('6');
  await ui.getByRole('button', {name: 'Pin comparison'}).click();
  await ui.getByLabel('vₓ', {exact: true}).fill('-3');
  await expect(ui.locator('.metrics')).toContainText('-6');
  await expect(ui.locator('#learning')).not.toContainText('Your prediction matches this model.');
  await expect(ui.locator('.comparison')).toContainText('dot=6');
  await ui.getByRole('button', {name: 'Ask about this experiment'}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.contexts.length)).toBe(1);
  const experiment = await page.evaluate(() => JSON.parse(window.fixtureHost.contexts[0].content[0].text.split('\n').slice(1).join('\n')).generated_experiment);
  expect(experiment.v).toEqual([-3, 4]);
  expect(experiment.metrics.dot).toBe(-6);
  expect(experiment.pinned.dot).toBe(6);
  expect(experiment.prediction).toBe('6');
  expect(experiment.generated_model_not_observation).toBe(true);
  await ui.getByRole('button', {name: 'Zero-vector example'}).click();
  await expect(ui.locator('.metrics')).toContainText('Undefined');
  await expect(ui.locator('#learning')).toContainText('does not guarantee a right angle');
  await page.evaluate(result => window.fixtureHost.push(result), toolResult({video, view: reviewedCases}));
  await ui.getByRole('button', {name: 'Test prediction'}).click();
  await expect(ui.locator('.answer-card')).toContainText('90 degrees');
  await ui.getByLabel('Both vectors nonzero').selectOption('no');
  await expect(ui.locator('.answer-card')).not.toContainText('The angle is undefined.');
  await ui.getByRole('button', {name: 'Test prediction'}).click();
  await expect(ui.locator('.answer-card')).toContainText('The angle is undefined.');
  expect(errors).toEqual([]);
});

test('slides and notes stay native; PowerPoint and tutoring request the selected source in chat', async ({page}) => {
  const {ui, errors} = await open(page);
  await ui.getByRole('button', {name: 'Presentation', exact: true}).click();
  await ui.getByText('Speaker notes', {exact: true}).click();
  await expect(ui.locator('.notes')).toContainText('gives 6');
  await ui.getByRole('button', {name: 'Next slide'}).click();
  await expect(ui.locator('.slide h3')).toHaveText('The useful exception');
  await ui.getByRole('button', {name: 'Prepare editable PowerPoint'}).click();
  await ui.getByRole('button', {name: 'Tutor', exact: true}).click();
  await ui.getByRole('button', {name: 'Start tutoring in chat'}).click();
  await expect.poll(() => page.evaluate(() => window.fixtureHost.messages.length)).toBe(2);
  expect(await page.evaluate(() => window.fixtureHost.messages[1].content[0].text)).toContain('Ask one focused question and wait');
  expect(errors).toEqual([]);
});

test('upload transfers original bytes without account credentials; free polling reaches ready', async ({page}) => {
  await page.clock.install();
  const {ui, errors} = await open(page, '?empty');
  await chooseUpload(ui);
  const transfer = page.waitForRequest(request => request.url().includes('/upload.mp4') && request.method() === 'PUT');
  await ui.locator('#start-upload').click();
  const request = await transfer;
  expect(request.postDataBuffer().toString()).toBe('fixture transfer bytes');
  expect(request.headers()).not.toHaveProperty('authorization');
  expect(request.headers()).not.toHaveProperty('cookie');
  await expect(ui.locator('#video-status')).toContainText('Processing your video');
  await page.evaluate(() => { window.fixtureHost.uploadStatus = 'ready'; });
  await page.clock.runFor(15001);
  await expect(ui.locator('#video-status')).toHaveText('Ready to explore');
  await expect(ui.locator('#notice')).toHaveText('Your video is ready to explore.');
  const uploads = await calls(page, 'upload_video');
  expect(uploads.map(c => c.args.action)).toEqual(['start', 'complete']);
  expect(uploads[0].args.filename).toBe('My lesson.mp4');
  expect(uploads[1].args.video_id).toBe('00000000-0000-4000-8000-000000000003');
  expect(errors).toEqual([]);
});

test('a lost charged completion can reconcile a lower balance without a second transfer or job', async ({page}) => {
  const {ui, errors} = await open(page, '?empty');
  await page.evaluate(() => { window.fixtureHost.failures.complete = 1; });
  let transfers = 0; page.on('request', r => { if (r.url().includes('/upload.mp4') && r.method() === 'PUT') transfers++; });
  await chooseUpload(ui); await ui.locator('#start-upload').click();
  await expect(ui.locator('#notice')).toContainText('could not finish');
  await expect(ui.locator('#start-upload')).toBeEnabled();
  await ui.locator('#start-upload').click();
  await expect(ui.locator('#upload')).toBeHidden();
  expect(transfers).toBe(1);
  expect((await calls(page, 'upload_video')).map(c => c.args.action)).toEqual(['start', 'complete']);
  expect(await page.evaluate(() => window.fixtureHost.balance)).toBe(100);
  expect(errors).toEqual([]);
});

test('a failed byte transfer can retry the same ticket', async ({page}) => {
  const {ui, errors} = await open(page, '?empty');
  let transfers = 0;
  await page.route('https://storage.example.test/upload.mp4**', async route => {
    if (route.request().method() === 'OPTIONS') { await route.fulfill({status: 200, headers: {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'PUT', 'Access-Control-Allow-Headers': 'Content-Type'}}); return; }
    transfers++;
    if (transfers === 1) await route.abort('failed');
    else await route.fulfill({status: 200, headers: {'Access-Control-Allow-Origin': '*'}});
  });
  await chooseUpload(ui); await ui.locator('#start-upload').click();
  await expect(ui.locator('#notice')).toContainText('could not connect');
  expect((await calls(page, 'upload_video')).map(c => c.args.action)).toEqual(['start']);
  await ui.locator('#start-upload').click();
  await expect(ui.locator('#upload')).toBeHidden();
  expect((await calls(page, 'upload_video')).map(c => c.args.action)).toEqual(['start', 'complete']);
  expect(transfers).toBe(2); expect(errors).toEqual([]);
});

test('canceling a transfer starts no processing and tariff changes require another user action', async ({page}) => {
  const {ui, errors} = await open(page, '?empty');
  await page.route('https://storage.example.test/upload.mp4**', async route => {
    if (route.request().method() === 'OPTIONS') await route.fulfill({status: 200, headers: {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'PUT', 'Access-Control-Allow-Headers': 'Content-Type'}});
    // Leave PUT pending until the user cancels it.
  });
  await chooseUpload(ui);
  await page.evaluate(() => { window.fixtureHost.cost = 501; });
  await ui.locator('#start-upload').click();
  await expect(ui.locator('#notice')).toContainText('processing cost changed');
  expect(await calls(page, 'upload_video')).toHaveLength(0);
  await ui.locator('#start-upload').click();
  await expect(ui.getByRole('button', {name: 'Cancel transfer'})).toBeVisible();
  await ui.getByRole('button', {name: 'Cancel transfer'}).click();
  await expect(ui.locator('#notice')).toContainText('Transfer canceled. Processing has not started');
  expect((await calls(page, 'upload_video')).map(c => c.args.action)).toEqual(['start']);
  expect(errors).toEqual([]);
});

test('rapid video selection ignores stale responses and transcript loads stay single while refreshing', async ({page}) => {
  const {ui, errors} = await open(page);
  await page.evaluate(() => { window.fixtureHost.delays['get_workspace_video:00000000-0000-4000-8000-000000000001'] = 250; });
  await ui.locator('#videos button').nth(0).click();
  await ui.locator('#videos button').nth(1).click();
  await expect(ui.locator('#video-title')).toHaveText('Second lesson.mp4');
  await page.waitForTimeout(300);
  await expect(ui.locator('#video-title')).toHaveText('Second lesson.mp4');
  await page.evaluate(() => { window.fixtureHost.delays.get_transcript = 250; });
  await ui.getByRole('button', {name: 'Load transcript', exact: true}).click();
  await ui.locator('#refresh-media').click();
  await expect(ui.locator('#load-transcript')).toBeDisabled();
  await expect(ui.locator('#transcript button')).toHaveCount(2);
  expect(await calls(page, 'get_transcript')).toHaveLength(1);
  expect(errors).toEqual([]);
});

test('optional host capabilities fall back to selectable text; no balance blocks processing', async ({page}) => {
  const {ui, errors} = await open(page, '?no-message&no-download&zero&no-media');
  await expect(ui.locator('#player')).toBeHidden();
  await ui.getByRole('button', {name: 'Explain this moment'}).click();
  await expect(ui.locator('#copy-text')).toHaveValue(/video_id/);
  await ui.getByRole('button', {name: 'Close text'}).click();
  await ui.getByRole('button', {name: 'Flashcards', exact: true}).click();
  await ui.getByRole('button', {name: 'Download flashcards CSV'}).click();
  await expect(ui.locator('#copy-text')).toHaveValue(/"Front","Back","Tags"/);
  await ui.getByRole('button', {name: 'Close text'}).click();
  await ui.getByRole('button', {name: 'Upload video', exact: true}).click();
  await ui.locator('#file').setInputFiles({name: 'own.mp4', mimeType: 'video/mp4', buffer: Buffer.from('bytes')});
  await ui.getByRole('checkbox').check();
  await expect(ui.locator('#start-upload')).toBeDisabled();
  expect(await calls(page, 'upload_video')).toHaveLength(0);
  expect(errors).toEqual([]);
});

test('dark theme, narrow layout and keyboard controls remain usable', async ({page}) => {
  const {ui, errors} = await open(page, '?theme=dark');
  await expect(ui.locator('html')).toHaveAttribute('data-theme', 'dark');
  await ui.getByRole('button', {name: 'Playground', exact: true}).click();
  await ui.getByLabel('vₓ', {exact: true}).focus();
  await page.keyboard.press('ArrowUp');
  await expect(ui.getByLabel('vₓ', {exact: true})).toHaveValue('3.25');
  await page.setViewportSize({width: 390, height: 844});
  expect(await ui.locator('body').evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await ui.getByRole('button', {name: 'Zero-vector example'}).click();
  await expect(ui.locator('.metrics')).toContainText('Undefined');
  await page.evaluate(() => window.fixtureHost.theme('light'));
  await expect(ui.locator('html')).toHaveAttribute('data-theme', 'light');
  expect(errors).toEqual([]);
});
