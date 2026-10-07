# VMF workspace component

The Python MCP server serves the tracked `src/api/assets/workspace.html` bundle.
Serving this component needs no Node build step at startup or CDN JavaScript. Node 20+ is
required for development. Dependencies are pinned in `package-lock.json`; license
notices travel inside the generated HTML.

```sh
cd mcp-ui
npm ci
npx playwright install chromium
npm run build
npm test
npm run check
npm run test:browser
```

`scripts/setup_local.sh` installs these development dependencies after its normal
database/backend/frontend setup. CI and `scripts/workflow/check_all.sh` reject a
stale bundle and exercise the actual SDK under a CSP without `unsafe-eval`.

For a local review, run `npm run preview`, then open
`http://127.0.0.1:6285`. This host supplies synthetic guides, cards, slides,
Playground data and tool responses. It has **no VMF authentication or charges**.
Use `?empty` for onboarding, `?theme=dark` for dark theme, and
`?no-message&no-download&zero&no-media` for capability/allowance fallbacks. The
automated tests intercept storage traffic and serve a tiny original geometric
test video; an ordinary manual preview does not have real source playback.

The fixture host is not proof of ChatGPT placement, picker permissions, actual R2
CORS or end-to-end indexing. See `docs/plugin/NATIVE_WORKSPACE.md` for the staged
host gate. Do not use fixture screenshots or fixtures as the public review demo.

`tests/lesson.webm` is an original, silent 12-second grid/shape fixture generated
with FFmpeg. No third-party transcript, lecture image or account data is included.
The browser tests use the real SDK and the same shared vector arithmetic as the
offline plugin tools. The optional `VMF_BROWSER_EXECUTABLE` test setting selects
an already installed development browser; CI uses the pinned Playwright browser.
