# VMF plugin release preparation

The portable source in `plugins/video-moment-finder/` owns listing text, prompts
and review cases. This record separates a private working package from a public
release. Last checked: 2026-10-02 (America/Bogota).

## Current release

Version **0.1.1** has 28 portable files and four skills. It was saved over the
existing USER-scoped PRIVATE plugin, preserving its identity and audience.
Read-back confirmed both manifests at 0.1.1, all four skills, the unchanged
starter prompts, icons, and the same OAuth MCP endpoint. The installed-plugin
page displays the selected publisher name and version.

```sh
uv run python -m unittest discover -s scripts/plugin -p 'test_*.py'
uv run python scripts/plugin/build_package.py
uv run python scripts/plugin/validate_package.py dist/video-moment-finder-0.1.1.zip
```

ZIP SHA-256:
`ddca03a0939c310a98b738323baaa67f164455d8a062a7a06c156c9af7f004a5`.

The builder normalizes ordering, timestamps and modes and inspects the finished
archive for paths, supported fields, icons, references, private bindings and
recognizable credentials. It cannot prove all rights or live service behavior.
The archive service adds compatibility manifests; keep those generated files
and private identifiers out of the portable public upload.

Both root JSON files also passed validation against schemas downloaded directly
from the declared Agent Plugins 1.0.0 URLs on this date. This is portable JSON
Schema validation, not OpenAI's submission validator. `--submission` continues
to fail on external gates; do not fabricate facts to clear it.

## Evidence and remaining gates

| Area | Verified result | Remaining boundary |
| --- | --- | --- |
| Package installation | Four skills appear in the installed sidebar; 0.1.1 metadata was read back | A clean-account installation must prove the packaged MCP connection independently of the existing custom connector |
| Live VMF tools | Listing, status, search, transcripts, frames and one upload worked | Existing publisher account, not a dedicated reviewer account |
| Original ingestion | One original 42-second narrated MP4: start → PUT HTTP 200 without Authorization → complete → queued → processing → ready | No second upload was attempted; this is not a long-video load test |
| Source retrieval | Complete original transcript and three inspected 1280×720 high-resolution frames matched the lesson | Original source retention is temporary |
| Authenticated playback | Original sample loaded and playback time advanced with no media error | Deployed `?t=27` link started at zero; the seeking implementation in this PR needs post-deployment verification |
| Second lecture | Dot-product excerpt transferred to a guide, eight-card HTML/CSV deck and four-case Assumption Lab | Explicit 20.62–134.75-second excerpt; not full-lecture or duality coverage |
| Visual evidence | Four actual 320×180 thumbnails inspected; ASR omissions disclosed | No inferred unreadable numbers; generated exercises stay labeled |
| Browser checks | Guide answers and embedded images; deck navigation/answers/show-all; all four lab cases, pin and reset preserving a note; no lab console errors | Guide narrow view checked; original cloud run checked all three templates on desktop/mobile. This follow-up did not independently complete every mobile/download case |
| CSV | Eight rows, three fields, multiline evidence and quoting parsed successfully | In-app browser download event timed out; companion CSV is available. No external Anki import claimed |
| OAuth reconnect | Existing ChatGPT custom connector returned to connected state; subsequent free list succeeded | Current consent still has Claude branding; new package's independent OAuth path and fresh-account onboarding remain to verify |
| Trial | Default-off code, unit tests and isolated migration CI prepared | No production migrations, trial grants or activation performed |
| Reviewer cases | Exactly five positive and three negative cases packaged | Full eight-case run against a saved portal release and dedicated account remains pending |
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

## Metered validation

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

Official guidance was accessible in the follow-up environment and was checked:
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

All four exact URLs were accessible, and their returned content was inspected.
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
