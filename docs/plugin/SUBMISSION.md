# VMF plugin release preparation

The portable source in `plugins/video-moment-finder/` owns listing text, prompts
and review cases. This record separates a private working package from a public
release. Current submission checkpoint checked: 2026-10-14 (America/Bogota). Historical
production user-flow observations are dated 2026-10-02; subsequent staging and
production database verification is dated separately below. The approved
production code rollout and subsequent reviewer ingestion/retrieval/playback
checks are recorded below. Human export review, the approved recording, identity,
domain, reviewer-login preparation and developer declarations are complete.
Package 0.4.2 is **In review** and **Not published**. Independent mobile checks
and public app identity runtime remain unverified; review approval and
publication are separate from this completed submission.

## Current release

### Submitted for review — October 14

The developer accepted the exported examples for initial testing, personally
checked all six current portal declarations, and explicitly approved submission
of VMF 0.4.2 for review. The first Submit request returned a network error.
A fresh portal read confirmed Not submitted before one bounded retry. The
retry succeeded: the detail page showed **Metadata & skills version 0.4.2 · In
review**, **Review status In review**, and **Publication Not published**.

[Review status](https://platform.openai.com/plugins/manage/plugin_asdk_app_6ace850757d081919db5f2badf9c6a10)
belongs to submission `appsub_6ace850757fc8191bda9419a3e8f1b5e`. No package
replacement, production deployment, public trial activation or publication was
performed. The submitted archive remains SHA-256
`ba3e8d65d2cb10d8948f562c77460d200b3a112a5a508688c0e0813fcf6abacb`.
The declaration panel's category note was non-blocking. Submission does not
establish review approval or verify the separate public app's end-user runtime.
Retain reviewer access and sample data while review is pending.

### Public-draft preparation — October 13

The portable 0.4.2 manifest now matches the uploaded public draft's verified
publisher name, user-facing listing/release notes, approved demo URL and original
upload fixture. It contains no reviewer password, personal app binding or signed
media capability. Identity and domain verification passed; the configured MCP
connection discovered ten tools with no latest scan findings. All six skill
checks passed. The publisher entered reviewer credentials directly in the
secure portal; those credentials were not inspected or included in the package.
At that checkpoint the draft remained unsubmitted and unpublished, with two non-blocking metadata
notes about category assessment and the automated privacy assessment.

Approved unlisted walkthrough: https://youtu.be/0fsteByQf9g.
The original synthetic 42-second reviewer upload fixture is available at
[the public fixture release](https://github.com/juancopi81/video-moment-finder/releases/tag/vmf-review-fixtures-2026-10-13).
P1 includes its attachment URL and an explicit download link. Anonymous download
returned 464,090 bytes, SHA-256
`c3b39b3f5bf4652042ed3e9ffd7ecc4f2eaa9ed659c8dd848869f8684e2627c4`.
The file is original generated math material, not a third-party lecture.

The reserved fresh production ingestion test completed through the native
workspace: one original upload, one completed first-attempt job and one
500-unit indexing charge. Playback loaded the 41.775976-second source, seeking
to 28 seconds passed, and same-ID completion retry returned ready without a
second charge. Transcript and frame controls succeeded. This does not establish
cancel behavior during transfer or direct observation of every network header.
The publisher manually selected the video and reference file because automated
nested-frame file selection was unavailable in this in-app browser.

All 40 installed private 0.4.2 skill/reference/template/tool/example files match
the public source byte-for-byte. Live tests use that installed private package
and the same production endpoint. The new public-draft app's direct workspace
preview reported App unavailable; do not claim that separate app identity was
end-user tested. Normal paths for P2–P5 and the three bounded negative requests were exercised
in the fresh review chat. Seven source claims, six native vector cases,
pin/reset/keyboard controls, card reveal/navigation, all six native slides and
speaker notes, and correct/incorrect/ambiguous tutor replies passed. CSV and
editable PowerPoint structures were checked independently. Source-ID validation
was repaired for free; two free native-delivery timeouts were retained in the
record and a subsequent unchanged-payload retry succeeded. Image search
initially lost its handoff after a one-unit charge; same-account renewal and one
explicit one-unit retry succeeded. Text/image cached repeats used no new units.
The additional evidence tally is five of twenty approved units.

Offline HTML visual inspection is blocked by this browser automation policy.
A requested phone viewport override had no effect on the actual desktop window
and was reset. Human export/narrow-window review was pending at this checkpoint
and was accepted on October 14. In-flight
cancellation, every edge variant, native CSV download receipt, actual mobile
host behavior and the new public-draft app identity were not independently
verified in this run. The five positive/three negative definitions and successful
normal paths do not imply that every declared check passed. The public trial
remains disabled.

### Verified runtime checkpoint — October 9

**0.4.2 is deployed to production and saved over the same private plugin**, adding
native text and reference-image moment search. The production constraint
migration preceded PR #97's merge at `7737afd`; main CI and API/worker/frontend
deployments passed. Private read-back verified all 48 files and six skills,
with identity, icons, starter prompts, publisher and MCP configuration preserved.
Reviewer OAuth renewal, free library access and actual native text/image search
passed. Text returned five visual/two spoken candidates; image returned five
visual candidates. Cached reuse, source seeking to 37/40 seconds and inspected
zero-vector thumbnails passed. A constrained image-only handoff reported
missing image evidence; source-text fallback at 0:28 produced one grounded
tutor question and waited without further metered retrieval. Image-only
handoff grounding remains unverified. The normal Explain this moment action
subsequently retrieved one permitted thumbnail at 0:40 and correctly read its
heading/footer before explaining the zero-vector exception. Record that actual
retrieval and cost, or use the verified transcript fallback; do not claim that
cached workspace images alone reached the model.
The four original learning views were reopened together in the current 0.4.2
workspace with free `render_learning_view`; actual exported view data match
the originals exactly. Guide citation seeking, card reveal/advance, computed
Playground pin/change/zero-vector controls and slide navigation/notes passed
again. One tutor question waits at 0:36; learner feedback remains to record.
The ledger confirms six of ten approved existing units used and a 2,068 balance,
with no new indexing, grants or trial enrollment. Public submission has not occurred.

Image selection is local until explicit search; current API tariffs apply,
similarity is a candidate, and clicked timestamps feed owned playback and chat.
The app-only image tool expands consent to version 4 and requires the new
`image_query` ledger migration before rollout. Migration/deployment and reviewer
renewal, actual image search/source checks, the text-grounded chat fallback and
visual explanation after one model-side thumbnail retrieval are verified.
Recording must distinguish that retrieval from unverified image-only grounding.
P2 now covers both searches within the existing five-positive/three-negative
case set. The October 13 fresh P1 ingestion result is recorded above.
See [Native Workspace](NATIVE_WORKSPACE.md) and [Walkthrough](WALKTHROUGH.md).

Approved staging migration, deployment, consent and actual text/image search
passed on 0.4.0 using the existing original lesson. Each search returned the
36-second zero-vector slide and repeats reused the cache. Actual ChatGPT image
selection revealed a broken blob-URL preview; 0.4.1 uses local canvas previews
under the existing exact-origin policy. The corrected reference and source-frame previews passed in actual ChatGPT.
A final handoff query completed in the backend at about 60 seconds but missed
the SDK's one-minute response deadline. The 0.4.2 patch gives searches a bounded
three-minute wait and states that a timeout may still consume units, with no
automatic retry. Local virtual-clock tests reproduce the former failure and
verify slow completion and the bound. Both CI runs passed at `1c7a89c`; staging
deployed the exact 0.4.2 bundle and all 20 free protocol checks passed. ChatGPT
metadata was refreshed and the local reference manually reselected. Actual
0.4.2 image search returned five candidates, source seeking and the inspected
36-second frame passed, and Quiz me produced one correct zero-vector question
then waited. Its constrained test conversation made no extra metered retrieval.
Six of ten approved additional staging units have been used; balance is 75,
with no new indexing, grants or trial. The subsequent documentation head
`f4dd2b3` also passed both CI runs and deployed the identical bundle. At that
staging checkpoint production/private plugin 0.3.0 was unchanged. The current
production checkpoint above supersedes it; public trials, submission and
publication remain unchanged.

**Historical 0.3.0 production baseline:** it was deployed and saved over the same private plugin.
After publisher review and explicit rollout approval on October 9, PR #95 was
squash merged at `f1d04aa`; main CI and API/worker/frontend deployments passed.
Ten free HTTP checks, current nine-tool consent, reconnect, free owned-library
retrieval, the actual native workspace and original-source playback passed.
Private R2 CORS now allows only the observed production ChatGPT origin in
addition to the preserved website upload rule. Eight read-only storage checks
passed; public access and retention were unchanged.

At that baseline the reviewer received the authorized once-only 600-unit test
allowance and had 2,080 units. No units or indexing jobs were consumed by that
infrastructure rollout; subsequent demo/search usage is recorded separately.
At that October 9 checkpoint, the single fresh job was reserved for review
ingestion; it was subsequently used once in the October 13 test above. The portable archive has 46 files and six skills; the same
private service record had 48 files including compatibility files. Public portal
upload, complete review cases, recording, developer verification, attestations,
submission and publication remained gates at that historical checkpoint. The
October 13 record above supersedes that status. Public trials remain disabled.

An authorized **0.3.0 native workspace candidate** is deployed to isolated
staging. Zero-balance consent and vector isolation were introduced at `ccdca31`;
the verified native resource revision is deployed at `c2b735a`.
The separate private staging package is at 0.3.1 with its verified app binding.
Real zero-balance OAuth, the global empty workspace and initial combined
onboarding passed. On October 8,
private bucket-scoped storage and exact-origin CORS were configured; nine storage
checks and seventeen free deployed protocol checks passed. The staging API now
uses an isolated Qdrant collection and a read-only, collection-scoped key;
eighteen hosted vector checks passed, including production-access and API-write
denials. The staging API also has an approved, dedicated Modal token expiring
November 7, 2026, for the existing VMF inference app; five metadata lookups passed
with no inference invocation.
Seventeen free protocol checks passed after that deployment, and temporary
authentication cleanup was verified. The token is workspace-wide on Starter,
and the model app is shared with
production. The separately approved private Railway worker now uses
`Dockerfile.worker`, one replica and the isolated staging database, bucket and
vector write key. CI passed for `fb83158`; deployment and idle startup passed
with the queue and video table empty before and after startup. Both vector keys expire
November 7, 2026.
The approved private 600-unit allowance was granted once, without trial enrollment.
After an initial uncharged transfer failure, the resource-policy fix at
`2c9cebf`, connector refresh and manual reselection, actual ChatGPT upload and
processing passed: one original 41.8-second video, one completed first-attempt
job, approximately 71 seconds. The normal worker retry setting was then restored.
The ledger recorded 510 units and 90 remaining, with one indexing charge.
Inspected source frames, transcript/search, authenticated playback, seeking and
four full-conversation native views passed. CSV/JSON copy exports worked where
direct downloads were unavailable. The initial PowerPoint attempt hit a read-only
test folder; the fresh learning chat delivered an editable file successfully.
Its five slides, five source-note pages, native text and absence of signed URLs
were checked, and every rendered slide was inspected. Three development boundary
cases passed without paid retrieval or real-world actions.
The first learning request now explicitly opens a new chat on supporting hosts
to avoid invisible results in the global workspace's floating conversation.
Full local checks, both CI runs and staging deployment passed at `c2b735a`.
Eighteen free protocol checks verified resource revision 0.3.2 and its exact
reviewed bundle, with disposable-auth cleanup. The actual first-learning button
created a new conversation; after the existing bounded test authorization was
supplied, it rendered a fresh guide from one transcript and one inspected
thumbnail batch. Its 36-second source citation sought correctly. Same-account
OAuth reconnect passed without a new grant or trial. Accounting at that
checkpoint was 512 units used, 88 remaining and one completed job. No second job,
public trial or production change occurred. These private development checks do
not replace review cases against the saved public portal release. See
[native workspace](NATIVE_WORKSPACE.md) for build/browser evidence, expanded
consent and required host checks. The
deployed/installed 0.2.0 evidence below is
historical proof of that release, not validation of the new UI. Record the
candidate only after its actual host checks pass.

The fresh conversation preserved all four generated native views while reusing
its cached evidence. Card controls, slide notes, prediction/pin/compare and
zero-vector handling passed. Its last optional tutor handoff returned a ChatGPT
security challenge; the earlier full-conversation tutor question/feedback passed.
The publisher then resumed the cached-evidence tutor. After a normal reload,
the native Tutor button successfully sent the selected 36-second moment and
received the correct zero-vector question. No security bypass was performed.
The latest read-only ledger recorded 513 units used, 87 remaining, one completed
first-attempt job, one allowance event and no trial; it includes one additional
transcript retrieval. The publisher subsequently completed review of the four
views and authorized the October 9 production rollout recorded above.

## Earlier 0.2.0 package

Version **0.2.0** had **44 portable files and five workflows**. It was saved
over the same USER-scoped PRIVATE plugin. Read-back confirmed both root and
compatibility manifests at 0.2.0, all five skill files, unchanged starter prompts,
icons, publisher, worldwide targeting and OAuth MCP endpoint. The service stores
46 files because it retains its two compatibility files. This read-back does not
establish a new clean-account installation or live OAuth test.

Playground expands the existing `assumption-lab` skill in place so installations
do not accumulate duplicate skills. It supports computed models where justified
and retains the finite case-table mode. Presentation is the new fifth skill.
Editable PPTX export requires the host's presentation runtime; HTML generation
uses Python's standard library. That 0.2.0 package added no production dependencies.

For the current portable package:

```sh
uv run python -m unittest discover -s scripts/plugin -p 'test_*.py'
uv run python scripts/plugin/build_package.py
uv run python scripts/plugin/validate_package.py dist/video-moment-finder-0.4.2.zip
```

Current 0.4.2 public-submission ZIP SHA-256:
`ba3e8d65d2cb10d8948f562c77460d200b3a112a5a508688c0e0813fcf6abacb`.
The normalized source build matches the uploaded final archive exactly.

The private review collection is `docs/private/vmf-learning/review-0.2.0/index.html`.
Its companion ZIP is `dist/vmf-private-review-0.2.0.zip`. They include the real
lecture guide, cards/CSV, computed playground, teaching presentation/PowerPoint,
original work briefing/PowerPoint, finite lab and a tutor prompt/rubric. The tutor
page is not a newly recorded conversation. Keep this third-party lecture
collection private; the distributable package includes only original examples.

The builder normalizes ordering, timestamps and modes and inspects the finished
archive for paths, supported fields, icons, references, private bindings and
recognizable credentials. It cannot prove all rights or live service behavior.
The archive service adds compatibility manifests; keep those generated files
and private identifiers out of the portable public upload.

The 0.1.1 root JSON files passed validation against schemas downloaded directly
from the declared Agent Plugins 1.0.0 URLs on 2026-10-02. Version 0.2.0 preserves
that structure and passes the portable archive contract. This is portable JSON
Schema validation, not OpenAI's submission validator. `--submission` continues
to fail on external gates; do not fabricate facts to clear it.

## Earlier development evidence and limits

| Area | Verified result | Remaining boundary |
| --- | --- | --- |
| Package installation | 0.2.0 and five skills were read back from the saved private release. On October 6 the publisher completed the requested private-plugin reviewer connection; the exposed tools switched to the empty reviewer library | Native Codex connection UI was unavailable to automation; recording must show the installed version. A saved portal-release check remains separate |
| Live VMF tools | All six tools succeeded with the dedicated reviewer account on October 6; an attempt to access the publisher's original sample returned 404 | No public directory installation or saved portal release tested |
| Original ingestion | A separately approved original reviewer MP4 completed start → PUT HTTP 200 without Authorization → complete → queued → processing → ready; exactly one reviewer indexing charge | 41.776 seconds; not a long-video load test. No duplicate reviewer upload attempted |
| Source retrieval | Reviewer sample: complete two-segment transcript, targeted search and three inspected 1280×720 frames. After the approved private-storage rollout, another actual high-resolution frame and thumbnail at 22 seconds succeeded without fallback | Selected visual moments, not every frame; extend review access if review exceeds the bounded window |
| Authenticated playback | Reviewer sample opened at 27 seconds and advanced to the end without a media error. Its original source was subsequently preserved under the approved 90-day review rule, with integrity verified and expiry scheduled for January 4, 2027 | Ordinary source retention is 30 days; already-deleted publisher sources remain unavailable |
| Second lecture | Cached dot-product evidence now also drives a computed Playground and an eight-slide editable teaching presentation, alongside the existing guide, cards and finite lab | Explicit 20.62–134.75-second excerpt; not full-lecture or duality coverage |
| Visual evidence | Four actual 320×180 thumbnails inspected; ASR omissions disclosed | No inferred unreadable numbers; generated exercises stay labeled |
| Browser checks | Guide answers and embedded images; deck navigation/answers/show-all; all four lab cases, pin and reset preserving a note; no lab console errors | Guide narrow view checked; original cloud run checked all three templates on desktop/mobile. This follow-up did not independently complete every mobile/download case |
| Playground 0.2.0 | Real dragging and keyboard coordinates updated calculations; presets, zero-vector limits, prediction, pin/reset and note preservation passed. Tested narrow/desktop rendering without horizontal overflow or console errors. Model invariants checked across 1,600 vector pairs | Download click produced the correct comparison text; the browser tool could not capture a download receipt, so a visible copyable text fallback is included |
| Presentation 0.2.0 | Eight-slide lecture deck and six-slide fictional work deck exported with editable text and source notes. All 14 slides inspected. Package/geometry/font/import checks passed; reopened final-file renders were pixel-identical. HTML navigation, source-note disclosure, images and narrow layout checked | No native Microsoft PowerPoint or Google Slides execution claimed; work fixture is original fiction, not a retrieved meeting. Low-resolution lecture frames remain disclosed |
| CSV | Eight rows, three fields, multiline evidence and quoting parsed successfully | In-app browser download event timed out; companion CSV is available. No external Anki import claimed |
| OAuth reconnect | Publisher custom connector reconnected October 2. On October 6 the reviewer connected and subsequently reconnected through the private plugin; a new production OAuth record and successful free list/status calls confirmed access. The single 2,000-unit grant, 507 units used and 1,493 balance were unchanged; no trial or API key | The prior reviewer connection remains active; this step did not test revocation or expired-token recovery. Saved portal-release connection remains separate |
| Trial | Default-off code, unit tests, isolated migration CI, hosted staging checks, approved production schema migration and approved code rollout passed | Activation remains a separate decision; the flag is absent/default-off and no trial grants were issued |
| Reviewer cases | Five positive/three negative definitions imported; normal paths exercised with installed private 0.4.2 and dedicated original sample | Manual export/mobile review and unperformed edge variants remain gaps; new public-draft app runtime unverified |
| Review recording | The approved narrated walkthrough is hosted unlisted at https://youtu.be/0fsteByQf9g and declared in the current manifest | Prepared outputs are disclosed; it is not a recording of fresh ingestion |
| Submission/publication | Public draft, identity/domain verification, reviewer access and metadata scans prepared October 13 | Remaining case checks, legal attestations and explicit submission/publication authorization |

Private evidence lives under ignored `docs/private/vmf-learning/`:
`second-lecture/` contains the real excerpt, outputs and claim audit;
`original-fixture/` contains the original MP4, narration timing, actual live
retrievals, guide and audit. Earlier cloud artifacts remain in their cloud
workspace. Do not add third-party transcripts, frames, signed URLs, credentials,
or private account identifiers to Git or the distribution ZIP.

The original fixture generator now supports Linux ffmpeg/libflite or an
installed macOS speech voice, with an explicit font override and fail-fast checks
for empty audio. It only creates local media; it never uploads. Generate it with:

```sh
uv run python scripts/learning/make_fixture_video.py docs/private/vmf-learning/original-fixture
```

On macOS, speech services may require the execution environment's supported
permission flow. Do not weaken machine security or index repeatedly to diagnose
local media generation. Check the final MP4 duration with ffprobe; scene timing
can differ slightly after audio/video encoding.

## Artifact-update verification (2026-10-03)

The full local repository workflow passed: backend tests, archive tests, seven
frontend tests, lint and the 18-page production build. After final artifact
changes, the complete backend suite passed **686 tests**, with **17 isolated
PostgreSQL tests skipped** because Docker was unavailable locally; the 77 plugin
tests and 13 archive tests also passed. GitHub CI runs the PostgreSQL checks.
See PR #91 for the exact commit and CI result.

All 14 final PowerPoint slides were reimported and rendered. Their pixels matched
the individually inspected authoring renders. Every slide retains native editable
text and a speaker-notes part; the lecture deck contains three actual source
frames. The public work example has no third-party assets, private identity or
measured customer claims. This does not claim native PowerPoint execution.

The portable archive was rebuilt byte-for-byte. Public examples and skill links
resolve. The private review ZIP has 12 files and validated local links. There
were no new production dependencies, permissions, API endpoints or data stores.

## Hosted staging database verification (2026-10-05)

An empty, separate Supabase staging project received all 20 repository migrations
in one transaction. The 17 existing PostgreSQL integration cases passed against
that hosted database, including concurrent one-time grants, spending and retry
serialization, refunds, historical offsets, paid-credit fallback and client-role
restrictions. A separate schema preflight passed. Synthetic accounts were removed;
an independent audit confirmed that all 18 application tables were empty afterward
and the migration ledger retained all 20 entries. No production data was copied,
real-user trial enrolled, VMF tool invoked or production service changed.

The staging project was created without automatic Data API table exposure. Server
CRUD privileges and the trial owner's RLS-scoped SELECT privilege were explicit.
This checkpoint verified the hosted database behavior; API and identity checks
continued separately below. Private receipts and guarded test runners are under
`docs/private/vmf-learning/release-0.2.0/`; credentials are excluded from Git and are not part of the release package.

## Hosted staging API verification (2026-10-05)

Railway's separate API-staging service deployed commit `d3ba818` from
`codex/vmf-learning-plugin` to
`https://api-staging-staging-72b8.up.railway.app` on port 8080. Its OAuth discovery
health check passed. The first runtime started before its public domain existed;
redeployment after domain creation resolved the issuer/resource configuration.

All 20 live HTTP checks passed: API version, authorization/protected-resource
discovery, MCP authentication challenge, authenticated empty-library retrieval
through the real Supabase SDK, zero balance with disabled trial, free reads,
JWT-only billing restrictions, missing evidence, CORS, dynamic client
registration, PKCE redirect, six-tool consent metadata, authenticated-consent
requirements, invalid-code/resource rejection and revoked-key rejection.
Disposable client/request/key records were removed; the cleanup audit found zero
trial grants, videos, balances or usage events. The final complete run needed no
transport retries; earlier client probes intermittently timed out, with no root
cause established or long-duration availability measurement.

Real Clerk sign-in and staging JWT authentication subsequently passed: the API
dashboard displayed the signed-in account and zero units; key, balance and usage
requests returned HTTP 200. A disposable 10-unit staging fixture enabled
publisher-approved PKCE token exchange and real MCP initialization, six-tool
discovery, prompt discovery and an empty-library call. A second approved
connection passed 17 checks, including account/resource/consent binding,
cross-account isolation for all five read tools, single-use authorization codes,
refresh rotation, rejection of superseded tokens and successful refreshed access.
Neither trial grants nor usage events were created; the fixture balance was
unchanged. Both connections and all disposable records were removed and audited.

The next check found that MCP SDK 1.26 rejects public-client revocation when the
secret field is omitted. Three regression cases reproduced the failure locally.
A narrow form-normalization fix passed full local validation (691 backend tests,
13 archive tests, seven frontend cases, lint and build); confidential-client
authentication and cross-client revocation restrictions remain covered. Commit
`d3c3826` passed both GitHub CI runs (703 tracked backend/PostgreSQL cases,
including all 17 isolated database cases) and deployed successfully to staging.
All 11 follow-up revocation checks passed with disposable synthetic-account
token fixtures: both token types invalidate the entire connection, another
client cannot revoke it, and an unknown token returns empty success. The cleanup
audit found zero owned fixture records; an independent read-only database audit
confirmed all 18 application tables are empty and all 20 migrations remain applied.
This follow-up verifies revocation
independently; it does not claim another real-account consent flow. Intended-host
package calls remain pending. Staging
currently has no worker, storage, embedding or payment credentials, so these
checks do not validate indexing or playback there.
Trial grants remain disabled. Production schema and code were subsequently
released under separate approvals, as recorded below.

## Approved production database checkpoint (2026-10-05)

After explicit publisher approval, the two missing migrations were applied in
one guarded transaction: `20261002120000_api_billing_retry_safety.sql` and
`20261002121000_verified_account_trial.sql`. Production now records all 20
repository migrations. The first preflight stopped before DDL because the SQL
editor's displayed function text normalized indentation. A byte-preserving
database snapshot confirmed the same original definitions; only the guard
digests were corrected before retrying the unchanged migration SQL.

All 11 independent read-only database checks passed: the ledger, billing/refund
functions, row-level security, owner-select policies and server-only mutating
RPC permissions. Both new tables remain empty: zero trial enrollments and zero
processing-charge records. No account balances were changed by the migrations.
Three free public production HTTP checks also passed afterward: OAuth discovery,
protected-resource discovery and tokenless MCP HEAD (204). These checks do not
establish authenticated production billing, packaged OAuth or playback behavior.

At this October 5 checkpoint, production API trial enablement was absent and
code deployment, trial activation, public submission and publication had not
been authorized or performed. Local frontend and callback services were stopped
for the end-of-day handoff. The separately approved code rollout followed on
October 6; do not replay the already-recorded migrations.

## Approved production code rollout (2026-10-06)

After explicit publisher approval of merge and production rollout, PR #91 at
head `b5edab4` was squash-merged into `main` as
`67f4c28e7a792a5b94a0c759b50632d63c27e4b7`. Railway's production API and worker
deployments and Vercel's current production frontend all reported success at that
exact commit. The API startup completed and the publisher's existing connected
MCP session successfully listed videos. No new indexing job or metered retrieval
was run.

Ten free public HTTP checks passed: OAuth/resource discovery, tokenless MCP HEAD,
unauthenticated MCP rejection, developers/support/privacy/terms pages, the consent
client shell and the published skill document. The consent heading is rendered
in the browser; it cannot be asserted from the initial HTTP body. A hydrated
browser check showed neutral “Connect Video Moment Finder” wording and no
checkout/purchase links on consent. The incomplete-link explanation was correct;
this is not proof of a fresh valid OAuth approval.

The authenticated API dashboard showed the unchanged balance of 5,437 units.
Production trial enablement remained absent/default-off. Four fresh read-only
database checks passed: all 20 migrations recorded, both approved October 2
migrations present, zero trial enrollments and zero shared website charges.
No new grant, purchase, migration replay or production configuration change
occurred during rollout.

The original sample link with `?t=27` rendered “Source timestamp 0:27,” but its
temporary source media had expired. The page correctly explained that playback
was unavailable while transcripts/thumbnails remain usable. Actual timestamp
seeking was not verified by that rollout check. The separately approved reviewer
test below subsequently verified it. Private rollout receipts and
screenshots are under `docs/private/vmf-learning/release-0.2.0/`.

The rollout does not establish a clean-account packaged installation, reviewer
credentials or walkthrough, and does not activate the trial, upload a public
draft, submit for review or publish the plugin.

## October 6 reviewer validation

The publisher created and signed in to a dedicated, verified-email reviewer
account with a configured password and no MFA requirement. Credentials were not
read, changed or stored in repository content. A separately approved manual,
idempotent grant funded it with 2,000 units once; it was not a purchase or public
trial enrollment. After the publisher completed the private-plugin connection,
the exposed tools returned the empty reviewer library and production metadata
confirmed a new active OAuth connection. No API key was created.

The one approved original upload completed successfully. Actual ledger charges
were indexing 500, transcript 1, search 1 and high-resolution frame batch 5:
**507 of the 600-unit agent cap**, with **1,493 units remaining**. The sample's
full spoken transcript and three actual frames were checked. Authenticated
playback loaded at 27 seconds and advanced to the end without a media error.
Access to the publisher's original video correctly returned 404.

Fresh private guide, six-card HTML/CSV, computed Playground and six-slide
editable presentation use this reviewer sample. Six substantive source claims,
five independent model results, CSV structure, JavaScript syntax and the deck's
package/geometry/font/import checks passed. Every final slide was visually
inspected after reopening the export. After browser automation's URL policy
denied local-file previews, the publisher reported completing the requested
manual preview check: guide answer reveal, flashcard reveal and advance,
Playground input change and zero-vector preset, and presentation navigation and
speaker notes. This is a human-reported check of those controls, not automated
desktop/mobile coverage, a CSV download/import or native PowerPoint validation.
The live tutor behavior check used
five actual publisher replies, including deliberately scripted incorrect and
ambiguous inputs. It confirmed source-cited feedback, a hint before the answer,
an ambiguity clarification without grading, and acceptance of the clarified
answer. The test added zero units and makes no learning-assessment claim.
The publisher also sent the three packaged negative requests in one message.
The response explained that VMF cannot edit/publish a video, purchase or refill
units, or identify a stranger and find private contact details. No VMF tool was
invoked, purchase/checkout initiated or unsupported result claimed; this added
zero units. These are three bounded development responses in this chat, not an
independent reliability evaluation or saved portal-release passes. The publisher
then completed reviewer reconnect. A new OAuth record and two successful free
list/status calls confirmed access to the same one-video reviewer library.
Read-only accounting checks found exactly one original 2,000-unit grant, unchanged
usage of 507 and balance of 1,493, no trial and no API key. The previous connection
remains active; no revocation or expired-token recovery is claimed by this check.
At this checkpoint the source header scheduled deletion on October 9, and the
storage credential could not read the full bucket lifecycle configuration.
The publisher subsequently opened the dashboard and separately approved the
private-media and retention rollout below. These are development checks, not
a completed eight-case run against a saved portal release.

Private receipts and outputs are in
`docs/private/vmf-learning/release-0.2.0/reviewer-sample/`; account and allowance
receipts stay outside the ZIP and Git.

## Approved private-media and retention rollout (2026-10-06)

PR #93 passed full local validation and GitHub CI, then merged as `ed31032`.
Production API, worker and frontend deployments all reported success. Website
search loaded five signed thumbnail previews before public bucket access was
disabled. The unsigned original test-video URL then returned an unauthorized
response in the browser. The public development URL is disabled and no public
custom domain is configured.

The dashboard confirms enabled 30-day `source/` deletion and a separate 90-day
`review-samples/` deletion rule. Existing incomplete-upload cleanup, thumbnail
retention and CORS were preserved; no bucket lock was added. The single original
reviewer fixture was copied into the bounded review prefix. Its 464,090-byte
length and SHA-256 matched, its expiry header is January 4, 2027, and only its
existing source pointer changed. No new video, index or allowance was created.
The actual MCP status, 1280×720 high-resolution frame and 320×180 thumbnail
retrieval succeeded after this change.

One metered search timed out at the MCP transport while deployment was underway;
the ledger confirms its one-unit charge. Later status and frame calls succeeded.
The high-resolution and thumbnail checks cost five and one units respectively.
Total reviewer usage is now **514 of the approved 600 units**, with **1,486 units
remaining** and the same one grant and one indexed video. No API key or trial
enrollment was created. Keep this updated accounting separate from the earlier
507-unit reconnect checkpoint. The original ingestion fixture and recording
still need durable reviewer delivery; private media links are temporary.

## Metered validation

Artifact generation for this 0.2.0 update reused previously retrieved evidence
with zero additional VMF calls, units or indexing jobs. Subsequent isolated
staging checks used free metadata/library calls and denied cross-account reads;
they created no metered usage events and used no production units or indexing
jobs. Production rollout checks also used only free public, account and library
reads. The prior cumulative usage below remains unchanged.

The publisher authorized at most 700 existing units and one new indexing job.
The live account dashboard reconciled the readiness calls to **4 actual units**,
rather than the conservative 8-unit estimate (high-resolution fallback billed as
a thumbnail call). This follow-up used 3 units on the second lecture and 506 on
the original ingestion/transcript/high-frame batch: **513 actual units total**.
The account balance moved from 5,946 before this follow-up to 5,437 afterward.
The reconnect consent confirmed deployed tariffs: indexing 500, search 1,
transcript 1, thumbnail call 1, high-resolution call 5; list/status free.
No credit purchase or trial grant occurred. Recheck tariffs and allowance before
new metered work; these observations are not a permanent price guarantee.

## Publisher and current policy

The portal approved the individual identity **Juan Carlos Piñeros Pazmiño**
on October 13. The manifest now uses that exact author/developerName. The
publisher selected **all platform-supported countries**, declared through
`publication.countries: []`; the imported country selection was verified in
the portal. Identity verification does not substitute for policy attestations.

Official guidance was rechecked on 2026-10-14:
[submission requirements](https://developers.openai.com/plugins/deploy/submission)
and [commerce guidance](https://developers.openai.com/plugins/plugin-guidelines#commerce-and-monetization).
The earlier cloud destination-policy failure is resolved as an evidence gap.

Current guidance allows access through an existing paid account, while prohibiting
digital-credit sales, upgrade promotion and checkout initiation through a plugin.
An informational entitlement explanation can be appropriate. Accordingly,
`review.commerce` is false **for this plugin's actions**, with an explicit
description of the separate website's paid credit/unit service and the inactive
trial. This describes plugin behavior; the developer's separately completed
portal attestations do not establish review approval.

## Anonymous listing URLs

On 2026-10-06 all four exact URLs returned HTTP 200 with the approved updated
production copy. The consent route was also inspected after browser hydration.

| URL | Current evidence | Release action |
| --- | --- | --- |
| [Website](https://www.videomomentfinder.com/developers) | Neutral VMF connection documentation is deployed | Verify imported listing and the full clean-account path in the portal |
| [Support](https://www.videomomentfinder.com/support) | Public support contact and updated API-unit/source-retention wording | Include the exact URL in the final listing checks |
| [Privacy](https://www.videomomentfinder.com/privacy) | Approved connected-app, transcript and authorization-record disclosure is deployed | Publisher completes final submission attestations |
| [Terms](https://www.videomomentfinder.com/terms) | Approved website-credit/API-unit and connected-app terminology is deployed | Verify the final plugin flow against current portal requirements |

The website's independent purchase flow is separate from plugin-facing
consent/errors/outputs. Public accessibility alone does not clear policy review.
Tool annotations and the consent surface must describe actual deployed behavior.

## Finish release in this order

1. Await the review decision for the submitted 0.4.2 package. Keep the dedicated
   reviewer account and original sample data accessible. Do not read or export
   saved credentials, replace the submitted bundle, or cancel review without
   authorization.
2. If feedback requires changes, reconcile it against the submitted version
   before preparing a correction. Preserve the tested normal paths, exact
   private/source comparison, repairs and unverified edge/mobile/public identity
   checks. No additional indexing, grants or purchases are authorized here.
3. After approval, verify the intended approved package and country targeting,
   and test the public app identity when available. Obtain separate publication
   authorization. Public trial activation remains separate; do not replay
   migrations or production deployments.

To retrieve the earlier cloud examples, ask that agent to attach its final
real-lecture HTML files, CSV, tutor transcript and ZIP as downloadable chat files,
excluding credentials and signed URLs. The cloud's workspace paths alone are
not local downloads.
