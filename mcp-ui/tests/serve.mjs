// Development-only host. All account data and tool responses are fixtures.
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';

const paths = {'/': new URL('harness.html', import.meta.url), '/fixtures.js': new URL('fixtures.js', import.meta.url), '/workspace.html': new URL('../../src/api/assets/workspace.html', import.meta.url)};
createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  if (!paths[path]) { res.writeHead(404); res.end(); return; }
  try {
    const headers = {'Content-Type': path.endsWith('.js') ? 'text/javascript' : 'text/html', 'Cache-Control': 'no-store'};
    if (path === '/workspace.html') headers['Content-Security-Policy'] = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src https://storage.example.test; media-src https://storage.example.test blob:; img-src https://storage.example.test data: blob:; frame-src 'none'; base-uri 'none'";
    res.writeHead(200, headers); res.end(await readFile(paths[path]));
  } catch { res.writeHead(500); res.end('Build the workspace first.'); }
}).listen(6285, '127.0.0.1', () => console.log('Fixture host: http://127.0.0.1:6285 (no live VMF calls)'));
