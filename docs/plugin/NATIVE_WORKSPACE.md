# Native workspace: 0.4.2

## Moment search update (2026-10-09)

The release adds Find a moment for the selected owned ready video. Text uses
the existing semantic search across visual and transcript vectors; results
label Visual or Spoken and seek the source player. Image accepts JPEG, PNG or
WebP up to 10 MB, normalizes locally to a JPEG at most 1024 pixels per side and
512 KiB, previews it without a network call, then submits only on Find moments.
The app-only `search_video_image` tool reuses the existing Modal image embedder
and owned Qdrant visual retrieval. It validates bytes/format/dimensions before
billing, uses `image_query` with the atomic debit/refund path, and keeps signed
thumbnail URLs in UI-only metadata. No query image is added to the video library.

Both controls show the effective tariff and refresh allowance/cost freely before
submission. A changed tariff needs a second click. Repeated queries reuse a
bounded in-memory cache without charging. Failed searches never retry
automatically. Late results cannot appear under another selected video.
Explain/Quiz me receives the selected candidate without a signed URL; a candidate
requires source verification. Similarity rank is not certainty or person identity.

Local reference and retrieved-frame previews draw their decoded pixels on a
canvas. They do not require blob/data image URLs or additional CSP origins.
The 0.4.1 patch fixes the broken reference preview observed in actual ChatGPT;
the browser fixture now blocks both URL schemes and verifies rendered pixels.

Resource cache revision is `ui://vmf/workspace/0.4.2.html`. Consent version 4 adds
the tenth tool and requires one reconnect. Apply
`20261009120000_image_search_api_usage.sql` before deploying this API: the older
ledger constraint rejects the new event type. This migration preserves existing
events and changes no balance, grant or retention setting. The approved staging
migration, renewed consent and actual text/image inference passed on 0.4.0;
both retrieved the original lesson's 36-second zero-vector slide, with cached
repeats. The broken local preview prompted the canvas patch. The publisher and
agent verified the actual 0.4.1 reference preview and source frame. A later query returned successfully in the backend at the SDK's 60-second
response deadline but timed out in the component. Search-only bridge calls now
allow at most three minutes; no compute scaling or tariff changes are involved.
Timeouts disclose uncertain billing and do not retry automatically. Both CI runs
passed at `1c7a89c`, and Railway staging deployed that head. Twenty free protocol
checks passed for the exact 0.4.2 bundle, resource URI, unchanged CSP/tariffs and
version-4 consent, with disposable authentication cleanup. ChatGPT tool metadata
was refreshed and the new workspace loaded with the existing lesson and balance.
After manual reference reselection, actual 0.4.2 image search returned five
candidates; selecting the 36-second zero-vector slide sought the source player
correctly. An inspected thumbnail rendered on canvas. Quiz me sent that moment
to the constrained test conversation, which asked one correct zero-vector/angle
question and waited without more metered retrieval. The final ledger records six
of ten approved additional units used, a balance of 75, one unchanged original
job/attempt/private grant and no trial. The documentation head `f4dd2b3` also
passed both CI runs and deployed the identical bundle. The subsequent production
rollout is recorded below; production image search remains a separate host check.

## Production search checkpoint (2026-10-09)

The publisher explicitly approved the search rollout. The single
`image_query` constraint migration was applied to production before PR #97
merged at `7737afd4bb19fcf6fcee4ab9c99a9a17989bab83`. Transaction snapshots
verified unchanged data, balances, grants, jobs, billing functions and RLS.
Main backend/workspace/frontend CI passed, followed by the production Railway
API and worker and Vercel frontend deployments. Ten free public HTTP checks
passed against the active release.

The same USER-private plugin was updated to 0.4.2 with its existing release
guard. Read-back verified all 48 files, six skills, matching portable/legacy
manifests, onboarding, starter prompts, publisher and MCP configuration.
Unchanged binary assets were preserved. The portable submission archive remains
46 files; private compatibility files are not its public packaging contract.

The connection requested and accepted OAuth renewal. Its free library result
matched the dedicated reviewer's existing original video; the database records
current consent version 4. Refreshed ChatGPT tool metadata loaded the new native
search controls and owned 1280×720 playback. One semantic text query returned
five visual and two spoken candidates. The spoken passage states that the zero
vector's angle is undefined. Selecting the fourth visual candidate sought the
actual player to 37 seconds, and a retrieved thumbnail rendered the same
zero-vector slide on canvas. Repeating the text query reused its cache.

The ledger records exactly two of ten approved existing reviewer units used:
one text query and one thumbnail. Balance is 2,072; videos, indexing jobs,
grants and trial enrollment are unchanged. Eight units remain for this smoke
test. Production reference-image selection/inference, cache reuse and the
moment-specific chat handoff remain to check. This rollout does not establish
saved public-review cases, trial activation, recording, submission or publication.

## Historical 0.3.0 production checkpoint (2026-10-09)

The publisher reviewed the four native views and authorized production rollout.
PR #95 was squash merged at `f1d04aac469d5ca97aab54504044d7d422a10575`;
main CI and the API, worker and frontend deployments passed. Railway API and
worker now wait for CI. The same USER-private production plugin was updated to
0.3.0, preserving its identity, starter prompts and connection configuration.
Read-back verified all six skills, including setup.

Current nine-tool consent, same-account reconnect, free owned-library retrieval
and the actual production workspace passed. The existing original reviewer
video loaded at 1280×720 with no media error. The exact observed ChatGPT origin
was added to private R2 CORS without changing website uploads or retention;
eight read-only storage checks passed, including signed source range/HEAD,
GET/HEAD/PUT preflights and unrelated-origin/anonymous-read denial.

A once-only 600-unit test allowance increased the reviewer's balance from 1,480
to 2,080. This checkpoint used zero new API units and created no indexing job.
Reserve the approved single new job and at most 600 new units for the saved
review-release ingestion and remaining checks. Public trial activation, public
portal upload, complete saved-release cases, recording, human attestations,
submission and publication are not established by this rollout.

Preserve the production plugin identity and the original reviewer video. The
dated staging sections below describe the development checks that preceded
production rollout; they do not establish public-review acceptance.

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

- `src/api/mcp.py`: workspace tools, app-only image search, entrypoints and HTML resource.
- `src/api/workspace.py`: strict cited-view schema and media-origin validation.
- `mcp-ui/src/`: browser source and shared vector arithmetic.
- `src/api/assets/workspace.html`: generated, self-contained component with
  third-party license notices; reproduced and checked from the lockfile.
- `plugins/video-moment-finder/skills/setup/SKILL.md`: first useful result.
- `plugins/video-moment-finder/references/native-views.md`: model delivery contract.

The existing evidence tools and indexing pipeline remain the acquisition layer;
app-only reference-image search adds one metered retrieval primitive.
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

October 9, 0.4.2 cold-response patch: virtual-clock SDK browser checks reproduced
the old timeout for both search modes, then verified delivery at 120 seconds
and a bounded 180-second failure without automatic retry or late result display.
The actual reference preview passed on 0.4.1. Full local checks passed: 736
backend cases, 18 isolated PostgreSQL skipped locally, 14 archive, four core,
25 SDK browser and seven frontend cases, lint and an 18-page production build.
The portable archive has 46 files and six skills; SHA256
`40cd438ca4cbfca15ee0c8e77de90296d4d68cf746cb14ca778660bbd2add91a`.
The corrected clock fixture isolates the delayed response from cross-frame timer
jumps; 24 repeated focused cases, the full browser suite and final error-notice
assertions passed. Both CI runs and 20 deployed staging protocol checks passed
at `1c7a89c`. The actual 0.4.2 image-search/source/tutor handoff subsequently
passed after manual reference reselection, using two more staging units for one
query and one thumbnail. Production approval remains required.

October 9, 0.4.1 preview patch: full local validation passed with **736 backend
cases**, **18 isolated PostgreSQL cases skipped locally**, **14 archive**, **four
core**, **22 SDK browser** and **seven frontend** cases, lint and an 18-page
production build. Browser checks now verify actual canvas pixels for references
and retrieved frames under an image policy that permits neither blob nor data
URLs. The portable archive has 46 files and six skills; SHA256
`7206c8fe7ed8f899c4105bc1651bb1d859a5219dacf9258d100502bf6ca9f31e`.
The API resource revision changes to bypass the host's old component cache;
OAuth remains version 4, with no additional consent, tariff or storage change.
Its CI/staging deployment and actual source/reference canvas previews passed.
The subsequent cold-response failure is recorded in the 0.4.2 entry above.

Before the patch, the actual ChatGPT 0.4.0 text query returned five visual and
two spoken candidates; the image query returned five visual candidates. Both
included the inspected zero-vector slide at 36 seconds. Same-query repeats used
the cache. The staging ledger records exactly two new units, balance 79, one
original job/attempt, one private grant and no trial. This single original
sample verifies the workflow, not general search quality or exact-match rank.

Historical October 9, pre-rollout 0.4.0 candidate: full local validation passed with **736 backend
cases** and **18 isolated PostgreSQL cases skipped locally**, **14 archive**,
**four core**, **22 SDK browser** and **seven frontend** cases, lint and an
18-page production build. Search scenarios cover both modes, byte validation,
ownership, atomic debit/refund, local image preview, explicit submission,
changed tariffs, cached repeats, empty results, failures, stale video responses
and candidate context clearing as playback moves. These were development checks;
actual ChatGPT image selection, inference, source seeking and reconnect had not
yet been verified at that checkpoint. The isolated database
cases run in CI, including the new event debit/refund test.

The reproducible portable 0.4.0 archive has 46 files and six skills; SHA256
`dda05089a514baf3b8c4f6251be8baab32054837578fdadaaa95bb95c6000226`.
At that pre-rollout checkpoint the installed private production plugin was
0.3.0. This archive validation did not establish portal validation, installation
or submission; the current production checkpoint is above.

Earlier native workspace verification:

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
in this web host; the selectable-text exports worked. The initial PowerPoint
attempt could not write in that test chat's output folder. The fresh learning
conversation subsequently delivered a real editable PowerPoint: five slides
and five timestamped source notes, native text shapes and no signed URLs. Its
download, package structure and all five rendered slides were inspected. The
three development boundary requests also passed without metered retrieval,
publishing, buying/refilling units or identifying a private audience member.

The global workspace's floating conversation accepted a render request without
showing the result. The first learning button now explicitly opens a new chat
when the host advertises `openai/message`; its source context travels in the
message, rather than being attached only to the previous conversation. Prepared
views keep follow-ups in place, and unsupported hosts retain their fallback.
The resource cache revision is `ui://vmf/workspace/0.3.2.html`; the portable
candidate version remains 0.3.0. Full local checks and both CI runs passed at
`c2b735a`, including sixteen SDK browser cases for new-chat context, active-chat
fallback and clearing the completed-upload notice. Staging deployed this
revision, eighteen free protocol checks matched its exact reviewed bundle and
security metadata, and disposable-auth cleanup passed. Actual new-chat creation
and same-account OAuth reconnect passed. The new chat requested authority before
metered retrieval because a widget message alone does not carry the publisher's
separate release-test approval. After that existing bounded approval was supplied,
the agent retrieved one transcript and one thumbnail batch, inspected frames,
and rendered a fresh native guide. Its 36-second citation sought to the correct
source slide. Final accounting at this checkpoint was 512 units used and 88
remaining; reconnect added no grant or trial. These are private staging
development checks, not public portal-release review results.

The fresh conversation subsequently generated six cards and the computed
Playground from its cached evidence, preserving the guide and five-slide deck as
companion views. Live controls passed: card reveal/advance, slide notes, prediction,
pinning the baseline dot product 6, changing the result to 8, and undefined
zero-vector angle/projection. Its final optional tutor handoff initially
encountered a ChatGPT `cloudflare_challenge` response. The publisher resumed the
cached-evidence tutor successfully; after a normal reload, the native Tutor
button also sent the selected 36-second moment and received the correct
zero-vector question. No security bypass was needed. The earlier full-conversation
tutor feedback remains verified. A subsequent read-only ledger check recorded
513 units used, 87 remaining, one completed first-attempt job, one allowance and
no trial; the extra recorded unit was a transcript retrieval. Publisher review
of the four generated views remains the next checkpoint.

Portable candidate: 46 files and six skills; archive SHA256
`4165b93edb382ad9271c930cf8156a0a58150c0291301adbec09ed685771ca19`. This is not portal or live-host approval.

## Release validation

Review the local fixture experience using `mcp-ui/README.md`. Staging transport
and private-package saving passed on October 7. Production rollout passed on
October 9; preserve these results and complete the remaining submission checks:

1. Real staging consent, global empty workspace, initial combined setup,
   full-conversation learning views, first-request new-chat routing, generated
   ready-video guide, same-account reconnect and the final native tutor handoff
   passed. The publisher reviewed the actual new conversation and four
   generated learning views before approving production rollout.
2. One original-video job, its first-attempt completion and ledger passed.
   Playback, seeking, guide/cards/slides/Playground and context handoff passed.
   Native CSV/JSON copy fallback passed. Editable PowerPoint delivery passed in
   the fresh learning conversation; it remains dependent on host capabilities.
3. Exact sandbox-origin CORS and storage-origin CSP passed hosted checks;
   actual browser transfer and playback now passed with those restrictions.
4. Reuse the indexed sample within the remaining approved allowance. Do not
   create another job or grant. Cancellation and transfer-recovery regressions
   passed in the harness; live retry of a failed byte transfer did not create
   a duplicate job. The latest ledger confirmed 513 units, one first-attempt job
   and one allowance event; any remaining checks must stay within
   the approved 600-unit cap.
   The staging UUID has no production website page; native playback is verified,
   while the production-domain “Open source on VMF” link is not a staging playback
   proof. Production reviewer links must be checked after the native rollout.
5. Production rollout and the update of the same private plugin identity passed.
   Run five positive and three negative review cases against the saved public
   portal release and dedicated reviewer account. Reserve the separately funded
   production test's one fresh job for that release. Recording follows the
   verified real version; portal upload, public submission and publication remain
   separate gates.

The 0.3.0 baseline required no new migration; the search release's ledger
migration is now applied as recorded above. A synthetic fixture does not satisfy any
live account, storage-transfer, charge or saved-release review requirement.

After building, check the local UI harness and intended ChatGPT host: placement,
initial-result reuse, themes, narrow layouts, keyboard controls, real file
selection/transfer, processing state, owned playback, expired links, reconnect,
zero allowance, source-text injection, exact context handoff, cards/CSV,
slides/notes, vector limits and finite-case completeness. A harness does not
replace a host test. Refresh the five positive and three negative cases and
record the actual new version only after those checks.

The 0.3.0 baseline used consent version 3; current native search requires version
4. Its production migration and reviewer OAuth renewal passed. The trial stays
disabled. The current search smoke test is limited to ten existing units and
zero new indexing jobs or grants. The separate earlier allowance of one fresh
job and at most 600 units remains reserved for the saved public-review release;
it is not used by this search rollout. Trial activation, recording, portal
upload and human attestations remain separate release gates.
