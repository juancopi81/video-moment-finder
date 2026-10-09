# Native workspace candidate: 0.3.0

This is a staging candidate. Production and the existing private plugin remain
on 0.2.0. Preserve that plugin identity and the original reviewer video.

## Experience and source ownership

Video Workspace opens from the sidebar or beside a conversation. Choose a ready
video or upload an authorized original file. Source playback and transcript
selection sit beside guides, cards, Playground and slides. Explain this moment,
Quiz me and Make it visual send the selected UUID, time and optional passage to
chat. Ask about this experiment includes the current inputs, prediction, pinned
comparison and checked model/case result, marked as generated practice. The tutor
remains in the conversation. Ordinary skills/exports remain the
fallback when the host does not support the native view.

The design uses a video library and two working panes, with system typography.
Colors: cloud `#f3f6fb`, paper `#ffffff`, ink `#19293f`, blue `#1e55ce`, slate
`#53657d` and moment marker `#e97924`, with host-aware dark colors. Source moments
and the vector plot carry the emphasis. Local browser checks include narrow
screens, keyboard controls and host-driven themes. Actual ChatGPT host checks
are still required before calling the integrated experience verified.

- `src/api/mcp.py`: three additive tools, entrypoints and HTML resource.
- `src/api/workspace.py`: strict cited-view schema and media-origin validation.
- `mcp-ui/src/`: browser source and shared vector arithmetic.
- `src/api/assets/workspace.html`: generated, self-contained component with
  third-party license notices; reproduced and checked from the lockfile.
- `plugins/video-moment-finder/skills/setup/SKILL.md`: first useful result.
- `plugins/video-moment-finder/references/native-views.md`: model delivery contract.

The six evidence tools and indexing pipeline remain the acquisition layer.
`open_workspace` returns an owned library and actual allowance/tariffs;
`get_workspace_video` refreshes owned playback; `render_learning_view` validates
prepared content and companion views from the same owned ready video. All three
use no API units. Rendering does not independently inspect images or verify that
an educational claim matches the lecture.

Ephemeral playback links stay in UI-only tool-result metadata and out of model
library/playback results and generated documents. Generated text is inserted as
DOM text, never executed HTML. The Playground reuses the checked offline vector
function, or a complete reviewed table; arbitrary generated code is not run.

Library ownership lives on the backend. Generated views are conversation
artifacts, not durable cross-chat learner profiles. Companion views keep prepared
outputs together and restore their contents from a tool result on remount without
repeating retrieval. Navigation and experiments live in the current component.
Transcript loads are reused per video; the last five requested thumbnails are
cached in memory and forwarded with the selection when the host supports image
messages. Retention and account access still determine playback availability.

## Upload and YouTube

Users select MP4, MOV or WebM, confirm permission and see the deployed processing
cost before Upload and process. Bytes go by credential-free PUT to the exact
configured R2 origin, followed by completion of the same UUID. Transfer progress
and processing status are distinct. A retry reconciles a possibly completed
request before repeating completion; it does not deliberately create a second
indexing job. Backend file limits do not establish host file-picker limits.

No YouTube URL ingestion is added. Owners use the original file or YouTube
Studio's download function, then upload. Content ownership and automated access
to YouTube are separate. No cookies or yt-dlp instructions are requested.

## Current verification and gate

October 7: **718 backend cases** passed; **17 PostgreSQL cases** skipped locally.
Four browser-core Node cases, 14 strict-CSP SDK browser cases and 14 archive cases
passed, along with seven frontend checks, lint and an 18-page production build.
The browser cases cover source seek/link refresh, inert source markup, exact
context/image handoff, cached retrieval, cards/CSV, slides/notes, computed and
reviewed Playgrounds, keyboard/theme/mobile use, cancellation, retries, tariff
changes and lost charged-completion reconciliation. The setup skill passed its
validator using a temporary PyYAML validation environment. These are local
checks; they do not establish the actual ChatGPT native host. CI repeats the
checks and enables isolated PostgreSQL tests.

The publisher approved the browser SDK and build dependency on October 7.
The component uses MCP Apps 1.7.5, MCP SDK 1.32.1 and Zod 4.2.0, with esbuild
0.25.12 and Playwright 1.56.1 for development. The SDK security patch is included;
the installed dependency audit reported zero vulnerabilities. No Python runtime
dependency changed. Build freshness, pure browser logic and strict-CSP fixture
host tests are part of CI and `scripts/workflow/check_all.sh`.

The fixture host exercises the real browser SDK and generated HTML with synthetic
content, intercepted storage requests and a tiny original test video. It does
not authenticate against VMF, debit units or establish ChatGPT host behavior.
After preview approval, Railway exposed a failed push-triggered browser run even
though the separate PR run passed. The fixture host could load its iframe before
its module had registered the SDK listener. Deferring iframe startup until the
listener exists fixes that test-host race; a controlled delayed-module case was
added. All 14 browser scenarios passed three consecutive repetitions (42 cases).
Both push and PR CI runs passed at `300297f`, including isolated PostgreSQL
checks. Railway deployed that commit successfully to the existing isolated
staging API. Fifteen free protocol checks passed: OAuth discovery, old-consent
rejection, nine tools, global/thread entrypoints, byte-identical 0.3.0 HTML
resource, empty owned library, zero allowance and denied foreign-video access.
Those checks used short-lived synthetic OAuth fixtures; cleanup was verified.
They do not establish real account consent or ChatGPT rendering.

The first real staging consent attempt exposed the legacy positive-balance gate
in both the frontend and approval endpoint. Consent now works at zero balance
without billing lookup or trial enrollment; each metered operation continues to
enforce its own allowance. Both CI runs passed at `9ec44e5`, including isolated
PostgreSQL checks, and Railway deployed that fix successfully. The user completed
real staging OAuth consent at zero balance; the custom plugin shows the connected
account.

A separate USER-private **VMF Staging** package was saved at 0.3.0 with six
skills and the staging MCP URL. The web page exposes its portable MCP declaration
but no account-connection control. The separate **VMF Staging Tools** custom MCP
connection completed the supported ChatGPT test flow. Its actual app identity,
verified from the plugin and live workspace URLs, is now bound to a private
0.3.1 update of the same staging package. Read-back preserved all six skills,
starter prompts, assets and staging server configuration; the package page shows
the app as Connected. This personal app binding is absent from the portable
public candidate.

The actual ChatGPT global entrypoint opened the native workspace in its sandbox.
The empty owned library and zero allowance rendered, free library refresh
returned without an error, and the upload dialog displayed ownership confirmation,
the actual 500-unit indexing tariff and instructions for downloading one's own
YouTube upload. No file was selected or transferred. The observed component
origin is `https://api-staging-staging-72b8-up-railway-app.web-sandbox.oaiusercontent.com`.
This establishes the global empty-workspace flow, not conversation-side rendering,
media playback or indexing. On October 8 the user sent the combined staging
package's setup prompt and checked its response. The workflow used the staging
integration, explained the empty library, zero units, disabled trial and actual
tariffs, and offered one text-grounded next step without a purchase promotion.
Ready-video onboarding and exact opener call count remain unverified.

On October 8, the publisher approved a 30-day Object Read & Write credential
restricted to a new private staging bucket, exact sandbox-origin CORS for
GET/HEAD/PUT and Content-Type, and staging service configuration. The four R2
variables were installed on Railway's staging API and the deployment became
active. Nine storage checks passed: bucket listing, exact-origin PUT preflight,
the permitted header, denial of another origin, presigned upload, object size,
signed GET/HEAD and denial of anonymous retrieval. One named 56-byte non-video
fixture remains in the private staging bucket. Seventeen free deployed protocol
checks passed, including the exact account S3 origin in the component CSP and
workspace configuration, unchanged UI bundle, access restrictions and trial-off
behavior; disposable authentication fixtures were removed and cleanup verified.

The staging processing worker is configured with the isolated staging database,
bucket-scoped storage, collection-scoped vector write and approved inference
credentials. Actual ChatGPT file transfer and playback remain gates; backend
HTTP checks and idle worker startup do not establish them. No indexing, API
units, migration or trial grant occurred during these checks.

On October 8, after the publisher's scoped approval, `video_frames_staging` was
created on the existing free Qdrant cluster with a default 2048-dimensional
cosine vector and keyword indexes on `video_id` and `source`. Two collection-only
keys expire November 7: read-only for the staging API and read/write for the
staging worker. The API's three vector variables were deployed with commit
`ccdca31`; the write key was subsequently installed on the approved worker.
Eighteen hosted checks passed: exact scope and expiry, filtered collection
listing, default
vector configuration, indexes, repository storage initialization, a synthetic
upsert and filtered queries, denial of API writes, and denial of production
collection access for both keys. One clearly labeled synthetic point remains;
it is not an indexed lecture. Production vector data was not read or changed.
CPU, RAM and capacity remain shared with production.

`QDRANT_COLLECTION_NAME` selects the same explicit collection for search and
processing, preserves `video_frames` when unset, and rejects blank configuration.
Nineteen focused storage tests and full validation passed: 722 backend tests,
17 PostgreSQL cases skipped locally, 14 archive cases, four core browser cases,
14 SDK browser scenarios, seven frontend checks, lint and build. All three CI
jobs passed for `ccdca31`. Production deployment remains separately gated; the
portable 0.3.0 ZIP still references the production endpoint.

Seventeen free protocol checks passed again after the vector configuration
redeploy, with the same reviewed UI bundle, exact storage-origin CSP, zero
allowance and disabled trial; disposable authentication cleanup was verified.

After the publisher approved staging inference access, a dedicated Modal token
was created with a 30-day lifetime ending November 7. On Starter it has
workspace-wide permissions; its configured use is the existing VMF inference
app in `main`. The three Modal variables were installed on the staging API.
Five metadata-only function/class hydration checks passed without invoking GPU
inference or redeploying the production model app. Seventeen free protocol checks
passed after that deployment, including verified disposable-auth cleanup. The
staging queue and video table were empty at the worker preflight. This verifies
access, not execution of the processing pipeline.

After separate approval, `Worker-staging` was created in Railway's existing
staging environment. It uses `Dockerfile.worker` from
`codex/native-vmf-workspace`, one replica and no public endpoint. Thirteen
variables bind it to the isolated staging database, private bucket and vector
collection, and the existing VMF inference app in Modal's `main` environment.
Wait for CI is enabled. CI passed for `fb83158`; that deployment became active,
and its startup log showed the worker loop with the documented queue defaults.
The queue and video table were empty immediately before startup and after idle
verification; no processing job or GPU invocation was run. Normal Railway
runtime usage applies. This establishes idle infrastructure readiness; actual
ingestion and the resulting charges remain separate release checks.

The publisher then approved one 600-unit private staging test allowance and one
indexing attempt for the original 41.8-second vector lesson. The audited grant
was applied once to the connected ChatGPT account; no trial was enrolled and
other balances were unchanged. The worker was redeployed with
`VIDEO_JOB_MAX_ATTEMPTS=1`, and its startup log confirmed the bound. ChatGPT's
workspace showed 600 units. Automated file-chooser attempts in the in-app
browser did not produce a chooser or select a file. The user then successfully
selected the original sample manually, and the native Upload and process action
reached transfer but reported a network failure. A read-only ledger check
confirmed the full 600-unit balance, zero usage events, zero videos and zero jobs;
no indexing attempt was spent. The exact sandbox-origin video PUT preflight
still passed. Browser transfer, rather than file selection, is now the blocker.

The component was first loaded before staging storage was configured. Resource
URIs are host cache keys, so a cached pre-storage policy is a possible cause;
it is not confirmed by browser diagnostics. The component now uses
`ui://vmf/workspace/0.3.1.html`, and both `resources/list` and `resources/read`
declare the same exact storage policy. This preserves the scoped origin without
wildcards. Full local validation passed: 724 backend cases (17 PostgreSQL
cases skipped locally), archive and browser checks, frontend lint and build.
Both CI runs passed, and Railway deployed `2c9cebf` to staging. Eighteen free
protocol checks passed, including descriptor/content policy equality and the
new URI; disposable authentication cleanup was verified. After connector refresh,
workspace reload and manual reselection, the actual native upload completed.
One indexing job completed on its first attempt in approximately 71 seconds.
The worker's normal three-attempt default was then restored and verified in its
deployed startup log. There was no second indexing job or allowance grant.

The October 8 actual-host test used the original lesson, not the synthetic
harness. High-resolution frames at 10, 22 and 37 seconds were inspected, the
transcript and one search were retrieved, and authenticated playback and seeking
passed. The ledger recorded 510 units: 500 indexing plus 10 evidence calls,
leaving 90 of the approved 600 units. All four prepared learning views rendered
beside a full conversation. Cards revealed and advanced, slide notes opened,
and the Playground preserved a pinned comparison, computed a negative dot
product and kept zero-vector angles undefined. Its experiment handoff and tutor
question used the selected context. Direct CSV/JSON downloads were unavailable
in this web host; the selectable-text exports worked. Editable PowerPoint
delivery remains unverified because the test chat's output folder was read-only.

The global workspace's floating conversation accepted a render request without
showing the result. The first learning button now explicitly opens a new chat
when the host advertises `openai/message`; its source context travels in the
message, rather than being attached only to the previous conversation. Prepared
views keep follow-ups in place, and unsupported hosts retain their fallback.
The resource cache revision is `ui://vmf/workspace/0.3.2.html`; the portable
candidate version remains 0.3.0. Sixteen SDK browser cases passed, including
new-chat context, active-chat fallback and clearing the completed-upload notice.
Deployment, actual new-chat routing and reconnect verification of this revision
remain pending.

Portable candidate: 46 files and six skills; archive SHA256
`4165b93edb382ad9271c930cf8156a0a58150c0291301adbec09ed685771ca19`. This is not portal or live-host approval.

## Release validation

Review the local fixture experience using `mcp-ui/README.md`. Staging transport
and private-package saving passed on October 7. Continue with the following
checks in ChatGPT before requesting production rollout:

1. Real staging consent, global empty workspace, initial combined setup and
   full-conversation learning views passed. Deploy and verify the first-request
   new-chat routing, ready-video onboarding and reconnect on the same account.
2. One original-video job, its first-attempt completion and ledger passed.
   Playback, seeking, guide/cards/slides/Playground and context handoff passed.
   Native CSV/JSON copy fallback passed; editable PowerPoint delivery remains
   host-dependent and must not be claimed as verified.
3. Exact sandbox-origin CORS and storage-origin CSP passed hosted checks;
   actual browser transfer and playback now passed with those restrictions.
4. Reuse the indexed sample within the remaining approved allowance. Do not
   create another job or grant. Cancellation and transfer-recovery regressions
   passed in the harness; live retry of a failed byte transfer did not create
   a duplicate job. Confirm the final ledger after reconnect and new-chat checks.
5. Record five positive and three negative saved-release review cases. Only then
   request production rollout and update the existing private plugin identity
   with a matching production package. Recording follows the new real version.

No new database migration is required. A synthetic fixture does not satisfy any
live account, storage-transfer, charge or saved-release review requirement.

After building, check the local UI harness and intended ChatGPT host: placement,
initial-result reuse, themes, narrow layouts, keyboard controls, real file
selection/transfer, processing state, owned playback, expired links, reconnect,
zero allowance, source-text injection, exact context handoff, cards/CSV,
slides/notes, vector limits and finite-case completeness. A harness does not
replace a host test. Refresh the five positive and three negative cases and
record the actual new version only after those checks.

The tool consent version is 3, so older connections need one reconnect. Plan
that transition before rollout. No database migration is needed and the trial
stays disabled. A new metered upload test needs a separate bounded allowance;
use the already-indexed reviewer sample for free checks first. Production,
trial activation, recording, portal upload and human attestations are separate
release gates.
