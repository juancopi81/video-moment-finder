# VMF plugin release preparation

The portable source in `plugins/video-moment-finder/` owns listing text, prompts
and review cases. This record separates a private working package from a public
release. Package and guidance checked: 2026-10-03 (America/Bogota). Deployed-service
observations below are dated 2026-10-02 and have not been rerun for this artifact-only update.

## Current release

Version **0.2.0** has **44 portable files and five workflows**. It was saved
over the same USER-scoped PRIVATE plugin. Read-back confirmed both root and
compatibility manifests at 0.2.0, all five skill files, unchanged starter prompts,
icons, publisher, worldwide targeting and OAuth MCP endpoint. The service stores
46 files because it retains its two compatibility files. This read-back does not
establish a new clean-account installation or live OAuth test.

Playground expands the existing `assumption-lab` skill in place so installations
do not accumulate duplicate skills. It supports computed models where justified
and retains the finite case-table mode. Presentation is the new fifth skill.
Editable PPTX export requires the host's presentation runtime; HTML generation
uses Python's standard library. No production dependencies were added.

```sh
uv run python -m unittest discover -s scripts/plugin -p 'test_*.py'
uv run python scripts/plugin/build_package.py
uv run python scripts/plugin/validate_package.py dist/video-moment-finder-0.2.0.zip
```

ZIP SHA-256:
`4e05430e8b8a626b53d87dc4fd650d822e63960c8858b6255143ee5c16ece3a4`.

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

## Evidence and remaining gates

| Area | Verified result | Remaining boundary |
| --- | --- | --- |
| Package installation | The prior version appeared installed; 0.2.0 source and five skills were read back from the saved private release | A clean-account installation must prove the packaged MCP connection independently of the existing custom connector |
| Live VMF tools | Listing, status, search, transcripts, frames and one upload worked | Existing publisher account, not a dedicated reviewer account |
| Original ingestion | One original 42-second narrated MP4: start → PUT HTTP 200 without Authorization → complete → queued → processing → ready | No second upload was attempted; this is not a long-video load test |
| Source retrieval | Complete original transcript and three inspected 1280×720 high-resolution frames matched the lesson | Original source retention is temporary |
| Authenticated playback | Original sample loaded and playback time advanced with no media error | Deployed `?t=27` link started at zero; the seeking implementation in this PR needs post-deployment verification |
| Second lecture | Cached dot-product evidence now also drives a computed Playground and an eight-slide editable teaching presentation, alongside the existing guide, cards and finite lab | Explicit 20.62–134.75-second excerpt; not full-lecture or duality coverage |
| Visual evidence | Four actual 320×180 thumbnails inspected; ASR omissions disclosed | No inferred unreadable numbers; generated exercises stay labeled |
| Browser checks | Guide answers and embedded images; deck navigation/answers/show-all; all four lab cases, pin and reset preserving a note; no lab console errors | Guide narrow view checked; original cloud run checked all three templates on desktop/mobile. This follow-up did not independently complete every mobile/download case |
| Playground 0.2.0 | Real dragging and keyboard coordinates updated calculations; presets, zero-vector limits, prediction, pin/reset and note preservation passed. Tested narrow/desktop rendering without horizontal overflow or console errors. Model invariants checked across 1,600 vector pairs | Download click produced the correct comparison text; the browser tool could not capture a download receipt, so a visible copyable text fallback is included |
| Presentation 0.2.0 | Eight-slide lecture deck and six-slide fictional work deck exported with editable text and source notes. All 14 slides inspected. Package/geometry/font/import checks passed; reopened final-file renders were pixel-identical. HTML navigation, source-note disclosure, images and narrow layout checked | No native Microsoft PowerPoint or Google Slides execution claimed; work fixture is original fiction, not a retrieved meeting. Low-resolution lecture frames remain disclosed |
| CSV | Eight rows, three fields, multiline evidence and quoting parsed successfully | In-app browser download event timed out; companion CSV is available. No external Anki import claimed |
| OAuth reconnect | Existing ChatGPT custom connector returned to connected state; subsequent free list succeeded | Current consent still has Claude branding; new package's independent OAuth path and fresh-account onboarding remain to verify |
| Trial | Default-off code, unit tests and isolated migration CI prepared | No production migrations, trial grants or activation performed |
| Reviewer cases | Exactly five positive and three negative cases packaged | P3 now covers flashcards plus a teaching deck; P5 covers Playground. A full eight-case run against a saved portal release and dedicated account remains pending |
| Review recording | Original sample and a concrete walkthrough script are ready | No screen recording or reviewer-accessible recording URL exists |
| Submission/publication | None performed | Developer/domain verification, account access, scans, legal attestations and release authorization remain |

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

Real Clerk sign-in, approved OAuth token exchange/reconnect and intended-host MCP
calls remain pending. Staging currently has no worker, storage, embedding or
payment credentials, so these checks do not validate indexing or playback there.
Trial grants remain disabled. Production has not been migrated or deployed.

## Metered validation

This 0.2.0 update made **zero VMF calls and used zero additional units or indexing
jobs**. It reused the previously retrieved evidence. The prior cumulative usage
below remains unchanged.

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

The publisher selected **Juan Carlos Pineros** and **all platform-supported
countries**. The manifest now has that author/developerName and explicitly sets
`publication.countries: []`. The selected name is not proof of identity
verification: the directory ultimately uses the verified identity chosen in the
portal.

Official guidance was rechecked on 2026-10-03:
[submission requirements](https://developers.openai.com/plugins/deploy/submission)
and [commerce guidance](https://developers.openai.com/plugins/plugin-guidelines#commerce-and-monetization).
The earlier cloud destination-policy failure is resolved as an evidence gap.

Current guidance allows access through an existing paid account, while prohibiting
digital-credit sales, upgrade promotion and checkout initiation through a plugin.
An informational entitlement explanation can be appropriate. Accordingly,
`review.commerce` is false **for this plugin's actions**, with an explicit
description of the separate website's paid credit/unit service and the inactive
trial. This is a factual draft declaration, not a legal attestation or review
approval. Check the complete deployed flow before submission.

## Anonymous listing URLs

On 2026-10-02 all four exact URLs were accessible, and their returned content was inspected.
They are still the older production pages, not the updated source in this PR.

| URL | Current evidence | Release action |
| --- | --- | --- |
| [Website](https://www.videomomentfinder.com/developers) | Old Claude-specific connection steps and paid Developer Pack promotion | Deploy neutral informational source and inspect it again |
| [Support](https://www.videomomentfinder.com/support) | Public support contact and FAQs | Verify updated unit/source-retention wording after deployment |
| [Privacy](https://www.videomomentfinder.com/privacy) | Public policy; old production wording lacks the explicit transcript/connected-app disclosure added here | Authorized publisher reviews the policy changes before deployment |
| [Terms](https://www.videomomentfinder.com/terms) | Public terms; old credit terminology | Verify API-unit and connected-app terms added here after deployment |

The website's independent purchase flow is separate from plugin-facing
consent/errors/outputs. Public accessibility alone does not clear policy review.
Tool annotations and the consent surface must describe actual deployed behavior.

## Finish release in this order

1. Review PR #91 and the default-off rollout in `docs/DEPLOYMENT.md`. Production
   database migrations, backend/frontend deployment and trial activation remain
   separate authorized actions. Apply schema before the dependent backend.
   Validate in staging or an equivalent isolated deployment first; local
   migration tests are not a staging OAuth test.
2. Verify deployed neutral consent, informational listing pages and timestamp
   playback. Test the packaged server in a clean intended-host account rather
   than relying on the publisher's existing custom connector.
3. Provide a dedicated reviewer account with authorized sample material and
   enough units. It must work without the publisher's personal Google login,
   phone, mailbox or private network. Store access details only in the secure
   portal fields. The original sample can be reused with authorized provisioning;
   do not silently consume another indexing job from the exhausted one-job test
   allowance.
4. Record the real walkthrough in `docs/plugin/WALKTHROUGH.md`, verify the
   finished video and reviewer-accessible hosting, then add the actual
   `review.demo_recording_url`. No recording API is exposed in this local
   browser-control session; screenshots and a script do not substitute for it.
5. Build and inspect the exact final ZIP, then perform the separately authorized
   public-draft upload. Check imported fields, category, connection, five positive
   and three negative cases, required scans, domain and developer verification.
6. The authorized developer completes legal/policy attestations. Submit for
   review only after authorization; publication after approval is another action.

To retrieve the earlier cloud examples, ask that agent to attach its final
real-lecture HTML files, CSV, tutor transcript and ZIP as downloadable chat files,
excluding credentials and signed URLs. The cloud's workspace paths alone are
not local downloads.
