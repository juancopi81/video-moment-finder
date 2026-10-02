# VMF plugin release preparation

The portable source lives in `plugins/video-moment-finder/`. Its manifest is the
source of truth for listing text, prompts and review cases. This document records
release gates that cannot be asserted by a source archive. Private creation,
account connection, public draft upload, submission for review and publication
are separate actions.

## Build and inspect the distributable

From the repository root, with the existing development environment installed:

```sh
uv run python -m unittest discover -s scripts/plugin -p 'test_*.py'
uv run python scripts/plugin/build_package.py
uv run python scripts/plugin/validate_package.py dist/video-moment-finder-0.1.0.zip
```

The builder writes a ZIP and an inventory/validation report in `dist/`. Archive
members are sorted, timestamps and file modes are normalized, and the final ZIP
is inspected without extraction. Repeating the build with the same content
produces identical bytes. The report checks portable manifest fields, the exact
MCP endpoint, discoverable skills, case counts and types, PNG dimensions, local
Markdown references, private bindings, dangerous paths, symlinks and recognizable
credentials/signed URLs. It rejects unsupported top-level content. It cannot
prove every source is licensed, detect every possible secret, validate a live
server, or replace the target portal's validator. Review the final inventory too.

`--submission` deliberately fails while external review facts remain unverified.
Do not remove required fields or invent values to make that command pass. The
portable manifest uses the Agent Plugins 1.0 schema URLs, but fetching those
schemas and the current official policy pages is presently blocked by the
environment's destination policy. The checked-in validator implements the
available plugin-creator package guidance; it is not advertised as an official
JSON Schema or portal validation result.

## Evidence and status, checked 2026-10-02

| Stage | State | Evidence or next action |
| --- | --- | --- |
| Source and archive contract | Passed for frozen version 0.1.0, 28 files and four skills | Final archive inventory and validation report are alongside the ZIP; frozen hash below |
| Current official requirements | Blocked | All three requested developer-document URLs returned proxy CONNECT 403; details below |
| Package metadata | Drafted with factual gaps omitted | No invented publisher, country targeting, commerce declaration or demo URL |
| Review cases | Drafted, not run as a saved release | Exactly five positive and three negative cases in `plugin.json` |
| Live tool observations | Separate development evidence | Existing connection proves neither this package installation nor saved-release review cases |
| Private package creation | Saved successfully and inspected | Stored version 0.1.0 is USER-scoped and PRIVATE; all four skills and the exact MCP endpoint were read back; private identifiers remain outside public docs |
| End-user installation / OAuth | Unverified | No supported direct installer was exposed; package discovery did not surface the new private package. Saving it does not prove host installation or an authenticated connection |
| Local HTML / intended-host UI / reconnect | All three HTML workflows passed local desktop/mobile browser checks; package host connection and reconnect pending | Development results/screenshots are in ignored `docs/private/vmf-learning/{render-checks,flashcard-render-checks,fourth-render-checks}/`; installing the final version remains a separate gate |
| Local frontend smoke | Developers page passed desktop/mobile layout and menu checks; consent shell only | `docs/private/vmf-learning/frontend-checks/smoke.json`; no pricing/checkout links in the consent shell, but authentication remained loading and was not verified |
| Tutor agent harness | Seven tutor turns with six scripted learner follow-ups | A fresh agent handled ambiguous, correct and incorrect answers, hints, generated practice, an unsupported claim and learner stop against cached evidence; this was not a real learner or installed-host test |
| Dedicated reviewer access | Not supplied | Human supplies a suitable test account through secure dashboard fields |
| Walkthrough recording | Script prepared; no recording or hosted URL | Follow `docs/plugin/WALKTHROUGH.md` |
| Public draft / review / publication | Not performed | None is implied by archive creation or private installation |
| Production trial rollout | Not performed | Trial code and migrations require a separately authorized deployment |

Frozen portable ZIP SHA-256:
`150b90b819464de0d56eb6c520406a77a2144a3c118b85b1c941783d8e97c1aa`.
The private package service added `.codex-plugin/plugin.json` and `.mcp.json`
compatibility files to its stored release. The original 28 portable files were
verified unchanged. Those service-generated files and private identifiers do
not need to be copied into the public source archive. Private backend acceptance
is separate from public submission validation and end-user connection testing.

The available plugin-creator guidance requires five positive and three negative
cases for an MCP app entering initial review, a verified recording, all four
listing URLs, suitable reviewer access, release metadata and correct tool
annotations. The repository's six MCP tools have explicit boolean
`readOnlyHint`, `openWorldHint` and `destructiveHint` annotations. Upload is a
write action that consumes units and can replace an incomplete upload record;
the other five tools retrieve existing account data. Review these annotations
again if the service behavior changes.

## Current requirements and monetization check

The user requested a fresh check of:

- <https://developers.openai.com/plugins/build/plugins>
- <https://developers.openai.com/plugins/deploy/submission>
- <https://developers.openai.com/plugins/plugin-guidelines>

On 2026-10-02, the initial shell request could not reach the network proxy from
the execution sandbox. A retry using the supported network-only permission
reached the inherited proxy, which returned CONNECT 403 for each destination.
The managed environment reports an enforced restricted policy that omits
`developers.openai.com`. No proxy, TLS or route change was made. The Agent
Plugins schema host was also blocked. These results establish a network-policy
gap, not that the public URLs are broken.

Consequently this record contains **no verified current monetization quote**.
Do not infer approval from the lack of a citation. Reopen the exact official
guidelines when permitted and record their date, relevant text and the resulting
decision before public submission. The implementation keeps normal website
checkout separate and keeps plugin-facing consent, errors, tool responses and
onboarding free of digital-credit purchase or upgrade promotion. The plugin
does not purchase, top up or automatically refill units. Its informational
unit accounting and prepared trial do not constitute an approval under a policy
we could not read live.

The publisher must confirm the truthful commerce declaration required by the
current portal, including the relationship between free trial usage, existing
unit balances and any independent website purchases. `review.commerce` and
`review.commerce_description` are intentionally absent until the correct portal
type and declaration are known. Omission is a review gap, not a declaration that
the business has no paid service.

## Listing and policy references

These exact URLs are backed by existing route source and are stored in
`extensions.com.openai.interface`. Their live contents and anonymous access
could not be verified because the VMF destination was also blocked by CONNECT
403. An HTTP status alone would not suffice; inspect the actual content after
access is available.

| Field | URL | Local source / live status |
| --- | --- | --- |
| `websiteURL` | <https://www.videomomentfinder.com/developers> | `frontend/src/app/developers/page.tsx`; live verification pending |
| `supportURL` | <https://www.videomomentfinder.com/support> | `frontend/src/app/support/page.tsx`; contains support contact; live verification pending |
| `privacyPolicyURL` | <https://www.videomomentfinder.com/privacy> | `frontend/src/app/privacy/page.tsx`; live coverage verification pending |
| `termsOfServiceURL` | <https://www.videomomentfinder.com/terms> | `frontend/src/app/terms/page.tsx`; live coverage verification pending |

Before release, ensure the published policy covers account authentication,
OAuth grants, video transcripts/frames, processors, retention, deletion, and
usage accounting truthfully. The accompanying source changes now disclose
transcripts, OAuth connection/token-hash records, trial/usage accounting, and
data returned to authorized apps. They distinguish website credits from API
units and remove the inaccurate blanket claim of no third-party sharing.
Source changes do not publish those policy updates. Have the publisher review
the final policy commitments.
The support page must offer an actual functioning contact path; no test support
message has been sent during this work.

The 512×512 listing PNG and 192×192 composer PNG are copied from existing VMF
branding. They are below 5 MiB, have a legible play/time silhouette, and do not
need a new publisher identity. Optional dark-mode assets and brand colors are
omitted; their omission is not a release blocker. No author/developer name is
inferred from the repository maintainer or website footer. Confirm the intended
verified individual or business and use the matching portal identity.

## Review fixtures and case execution

The five positive cases cover one authorized ingestion, an HTML study guide,
flashcards/CSV export, multi-turn tutoring and an Assumption Lab. The three
negative cases cover video editing/publishing, purchasing units and identifying
an unknown person. They use natural prompts, explicit tool expectations and
observable pass conditions. See the exact draft in `plugin.json`.

| Case | Initial status | Required evidence to change the status |
| --- | --- | --- |
| P1 ingestion | Not run as a release review case | Actual attachment → start/PUT/complete/status, account units, resulting ID and status; no secret upload URL in notes |
| P2 study guide | Not run as a release review case | Actual tools/arguments, inspected frame, rendered artifact, explicit coverage and five-claim audit |
| P3 flashcards | Not run as a release review case | Reused evidence, focused deck, working review format and round-trip export validation |
| P4 tutor | Not run as a release review case | Correct, incorrect and ambiguous response transcripts with grounded adaptation |
| P5 Assumption Lab | Not run as a release review case | All discrete combinations; reference pinning, baseline reset and comparison-note export; audit of outcomes/assumptions/evidence |
| N1 video editing | Not run as a release review case | Clear limit, no VMF calls and no fabricated edit/publication |
| N2 commerce | Not run as a release review case | Clear limit, no tool/checkout/top-up/upgrade action |
| N3 identification | Not run as a release review case | Clear limit, no identification lookup or VMF tool call |

Use original, redistribution-safe sample data. The portable vector fixture is a
local explanatory example; it is not already indexed in every review account.
The reviewer account needs its own ready sample and known UUID. An original
41.37-second narrated vector fixture (440,919 bytes) has been prepared locally
with `scripts/learning/make_fixture_video.py`; it has not been uploaded while
the effective tariff is awaiting confirmation. Its current local development
copy is in ignored `docs/private/vmf-learning/ingestion-fixture/`. It is original
material, but no public attachment URL or successful ingestion is claimed.
P1 may provide that sample once; do not fabricate an attachment URL or reuse the owner's
private lecture library as reviewer access. Keep real third-party lecture
evaluation artifacts only in ignored `docs/private/vmf-learning/`, never in the
package or Git. Never include full third-party transcripts, signed source URLs,
access tokens, passwords or reviewer-account details in this document.

Development tests may substantiate a workflow without counting as a saved-release
portal case. When a case is actually run, record Passed, Failed, Blocked or Not
run with concrete observations and the package version/hash. Do not add arbitrary
test-status fields to the manifest. A supported workflow that fails due to
authentication or units needs an honest recovery check; it is not one of the
three out-of-scope negative cases.

Development browser checks passed at desktop width 1440 and mobile width 390 for
the private real-lecture study guide, an eight-card flashcard deck, and a four-case
Assumption Lab. Screenshots were visually inspected. Checks covered embedded
frames, keyboard/hint interactions and the guide's teaching slider; deck
navigation and a downloaded CSV parsed as nine rows including its header; and
lab case selection, pinning, reset and a downloaded comparison note. There were
no JavaScript page errors. These observations do not establish an external
flashcard import, private-plugin installation, host connection/reconnect or a
real learner tutoring session. The bundled tutor response scenarios are authored
simulations and are labeled accordingly. Separately, a fresh tutor agent was
exercised through seven tutor turns with six scripted learner follow-ups covering ambiguous, correct and incorrect
answers, hint requests, a generated 64-cell calculation, an unsupported
temperature claim and a learner stop. The actual transcript is preserved in
ignored `docs/private/vmf-learning/simclr-tutor-agent-harness.md`. This checks
model behavior in that harness; it does not establish real-learner outcomes or
an authenticated installed-host experience. Final repository checks and the
resolved independent review are recorded in `STATUS.md`.

The local frontend smoke also passed `/developers` at desktop and 390px mobile
width, with no horizontal overflow and a working mobile menu. The consent route
served its shell with HTTP 200 and no pricing or checkout links. Its authentication
state remained loading, so this was not an authenticated consent, account-label,
trial-eligibility or OAuth round-trip check. Local rendering does not verify the
currently published listing pages.

## Remaining preparation in order

1. Confirm the intended verified publisher, supported countries (or explicitly
   all available countries), and commerce facts. Preserve unknown fields as
   absent; do not use empty country or translation collections accidentally.
2. Verify the published listing URLs, policy coverage and current official
   guidelines. Reconcile any policy rule with the final deployed service and
   package. Check category availability in the actual portal.
3. Open the already-created private package and install/connect it through the
   intended host's supported flow; reuse this identity rather than creating a
   duplicate. Complete OAuth and run the host
   workflows, including reconnect and missing-unit handling. No credentials may
   be extracted or stored in source.
4. Prepare a dedicated reviewer account with sufficient test units and authorized
   sample data, then verify reviewers can sign in without the publisher's phone,
   mailbox or private network. Put its login URL, tenant and exact instructions
   only in secure portal fields. Do not invent or publicly distribute them.
5. Rehearse, record and verify the real walkthrough described in
   `docs/plugin/WALKTHROUGH.md`. Host it with reviewer access, verify playback
   without private login, write the actual `review.demo_recording_url`, and
   rebuild and inspect the final ZIP.
6. When separately authorized, upload a public draft through the intended
   organization. Connect the actual MCP server through the portal OAuth flow,
   inspect imported metadata and target country settings, and run all eight
   cases against the saved version. Check required scans and domain/developer
   verification in the current portal. Preserve portal-generated app bindings
   in the saved release, while keeping them out of author-supplied public ZIPs.
7. The authorized publisher completes legal/policy attestations. Submission for
   review and publication of an approved release each require their own
   authorization. Neither action is part of this implementation handoff.
