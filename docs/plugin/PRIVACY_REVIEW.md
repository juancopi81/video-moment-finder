# Privacy and resubmission review — 0.4.3

Checked November 12, 2026. PR #103 was merged and deployed to production API,
worker and frontend at `04242b8b87299ed596bee9331aaf1aa9ffddf855`;
main-branch CI passed. Package 0.4.3 is now saved as an unsubmitted portal draft.
The existing MCP app and endpoint were preserved and OAuth was refreshed for a
new discovery scan. The rejection email identifies
unnecessary personal identifiers and undisclosed data categories, so the fixes
address every identified response path rather than assume one exact field was
the reviewer's concern.

## Response contract

`src/api/mcp.py` constructs explicit response allowlists. These also cover nested
video and match records; adding a REST field does not automatically publish it
through MCP. `src/api/mcp_privacy.py` catches wrapped execution and validation
errors at the MCP boundary. It uses authored messages, never backend exception
text or invalid input values. Image blocks and requested source content remain
available; source material itself can contain personal information.

| Tool | Model-visible result and purpose | Component-only data / omitted fields |
| --- | --- | --- |
| `open_workspace` | Library labels, source IDs/types/status, available API units and indexing cost | UI origin/limits/tariffs in metadata. No account ID, trial/grant history, record times or playback links. |
| `list_videos` | Source ID, filename or YouTube reference, type/status and returned count, so the user can select a source | No creation time, signed source URL or raw processing exception. |
| `get_video_status` | Same minimal video record; failed processing has a fixed message | No backend diagnostic text or playback capability. |
| `get_workspace_video` | Minimal record for the selected owned video | Temporary playback URL only in `_meta.vmf.source_url`. |
| `render_learning_view` | Requested guide/cards/Playground/slides, companion views, cited source references and minimal video record | Playback capability only in metadata. Validates same-source ownership; no new stored learner profile. |
| `upload_video` start | New video ID, temporary PUT URL, method/expiry, next step and instruction to omit Authorization | A necessary exception: the client needs this capability to transfer the explicitly requested file. No account ID or internal storage-key field. |
| `upload_video` complete | Action/video ID and minimal completed video record | No signed URL, record-creation time or stored exception. |
| `search_video` | Video ID/status and matches containing timestamp, similarity score, visual/spoken source and requested transcript excerpt | Signed thumbnail URLs only in `_meta.vmf.search_thumbnails`; no source playback URL or nested diagnostic fields. |
| `search_video_image` | Same allowlisted match contract; tool is workspace-only | Same metadata-only thumbnails. Query image is transient, not added to the library. |
| `get_transcript` | Requested source ID, language and timestamped transcript segments | Explicit transcript projection; no account, authorization or processing records. |
| `get_frames` | Source ID, resolution/fallback, requested/actual times, image-index mapping and actual images | Failed frames use a fixed message. No ffmpeg exception, filesystem path or signed URL in errors. |

Video IDs are resource selectors required by follow-up tools, not account
identifiers. Filenames/YouTube references distinguish the user's sources.
Transcript times and scores explain the requested evidence. Balance and tariffs
let the user avoid an unaffordable operation. Signed upload/playback/thumbnail
capabilities are restricted to the operation or component that needs them and
must never appear in exported learning artifacts or chat handoffs.

## Privacy policy reconciliation

`frontend/src/app/privacy/page.tsx` now discloses account identifiers and
verification state; content and filenames; usage/credits/trials; website payment
notifications; OAuth connection/token records; analytics identifiers and events;
operational diagnostics; and prepared learning content. It explains nested tool
data, component metadata, the necessary upload capability, recipients, purposes,
retention and deletion/revocation requests. Railway and Sentry were missing from
the processor list; a read-only production configuration check confirmed Sentry
is enabled for both API and worker. Signed-in PostHog events use the account ID.
Billing handlers store the original payment notification for duplicate-grant
prevention; its potential customer/payment fields are now disclosed. The plugin
does not return those notifications.

Retention evidence is deliberately separate from credential expiry:

- Uploaded sources: the existing documented production lifecycle is 30 days,
  with the separately configured reviewer-sample exception. This patch does not
  alter storage rules or recover deleted originals.
- Database: no scheduled deletion was found for account, completed OAuth,
  usage/billing or first-party product-event records. The policy states that
  limitation and offers deletion by support request.
- OAuth: pending requests expire after one hour and are cleaned on a subsequent
  authorization; codes/access/refresh credentials expire after ten minutes/one
  hour/thirty days. Expiry does not delete their database rows.
- PostHog: the current project retention response has no configured event
  deletion cutoff and `default_period_deletes=false`. A default/plan window of
  twelve months is not proof that events are automatically deleted; the policy
  does not claim it is.
- Modal: the signed-in workspace is on Starter. VMF calls Functions with
  `.remote` for text/image queries, video frames, audio and transcript text.
  [Modal documents](https://modal.com/docs/guide/security) up to seven days of
  input/output retention before deletion, one day for Starter app/container
  logs, and account-lifetime app/function/container metadata. Corrected the
  inaccurate claim that search images are not retained after a request:
  VMF does not save them in its library, database or object store, but processor
  retention still applies. This code does not configure customer-data Volumes
  or memory snapshots.
- Sentry: the signed-in subscription is Developer. Its current
  [error-event retention](https://docs.sentry.io/security-legal-pii/security/data-retention-periods/)
  is thirty days; an event keeps the retention applicable at ingestion. VMF
  sends exception events, not Sentry session replays or tracing spans.
- Railway: production deployment metadata confirms Hobby and no attached
  volumes. The [seven-day log window](https://docs.railway.com/observability/logs)
  is not proof of permanent deletion: Railway says upgrading restores older
  logs. No underlying purge deadline was verified.
- Vercel: the signed-in team is Hobby. All four project deployment-retention
  selectors are thirty days. [Deployment retention](https://vercel.com/docs/deployment-retention)
  includes exceptions for active/recent deployments and a recovery period;
  it is not a thirty-day guarantee for all copies. Runtime logs have a one-hour
  [Hobby window](https://vercel.com/docs/limits); build logs are tied to retained
  deployments. Neither establishes a backup-purge deadline.
- Supabase: the signed-in organization is Free. The production backup screen
  says the plan does not include project backups and exposes no backup list.
  Official pages conflict: the [Free backup FAQ](https://supabase.com/kb/faqs/platform/free-plan-backups)
  says no automatic backups, while the [upgrade FAQ](https://supabase.com/kb/troubleshooting/will-backups-be-accessible-from-the-dashboard-immediately-after-upgrading-to-a-paid-plan-hXY4rs)
  says up to seven daily copies may exist for Free projects. Therefore absence
  of a dashboard backup is not evidence of no provider copies. Free API/database
  logs have a [one-day window](https://supabase.com/pricing). Read-only SQL
  confirmed `pg_cron` is not installed; repository inspection found no scheduled
  account/OAuth/usage/billing deletion. Other provider backup copies and physical
  purge timing remain unverified; the policy does not invent a deadline.

No deletion jobs, retention settings or grants are changed. Live checks consume
existing API units, and the submission connection uses normal OAuth renewal.
The operator must still review the final policy before resubmission.
Unlinking in a host is not represented as deleting the account or as
proof that every outstanding credential was revoked.

## Other submission checks

Review basis: [OpenAI plugin guidelines](https://developers.openai.com/plugins/plugin-guidelines).
The stored website billing categories were checked against [Lemon Squeezy's
webhook documentation](https://docs.lemonsqueezy.com/help/webhooks/example-payloads).

- Ten tools have explicit read-only/destructive/open-world annotations; upload
  remains a write, and image search remains app-only. Consent stays at version 4.
- No checkout or top-up tool is exposed. Insufficient-unit responses remain
  neutral. Removed the inactive trial proposal from the package's user-facing
  README and trial inference from setup. Existing entitlements are still needed;
  listing does not promise free access or additional units on installation.
- Listing category now uses the exact supported `Education & Research` title.
  Publisher, endpoint and identity are preserved. Six skills, five positive and
  three negative review cases remain; P1/P2 now explicitly audit privacy.
- Archive validation checks required public URLs, metadata lengths, icons,
  categories, commerce boolean, case counts, unsafe paths, secrets and private
  app bindings. It validates the actual ZIP but does not replace a portal scan.
- Developer, support, privacy and terms pages are reachable. PR #103's policy
  is deployed; the processor-retention follow-up corrects its remaining
  search-image retention claim.
- The existing [0.4.2 demo](https://youtu.be/0fsteByQf9g) loads and plays; duration
  is approximately 153 seconds. This was checked in the signed-in browser, not
  an anonymous session. It remains an earlier-version walkthrough, not evidence
  that the 0.4.3 privacy changes have been deployed or tested in ChatGPT.
- The original reviewer MP4 downloads anonymously: 464,090 bytes, SHA-256
  `c3b39b3f5bf4652042ed3e9ffd7ecc4f2eaa9ed659c8dd848869f8684e2627c4`.
- Desktop plus 390/768-pixel fixture tests verify search thumbnails supplied
  only in metadata, playback seeking, learning controls, no horizontal overflow
  and clean chat handoffs. They do not establish actual ChatGPT mobile behavior.
- Dependency audit found six affected workspace packages. Bundle analysis
  confirmed Zod was shipped; the other five were unused server-side transitive
  packages. Updated existing Zod to 4.6.7 and refreshed those lockfile entries;
  `npm audit` then reported zero vulnerabilities. See the [Zod advisory](https://github.com/advisories/GHSA-mrhx-c7xc-m3w5).
- GitHub's moving `ubuntu-latest` image switched to 26.04, which the pinned
  Playwright release cannot install on. The browser-test job now explicitly
  uses supported Ubuntu 24.04; no test is bypassed.

## Validation

`./scripts/workflow/check_all.sh` passed: 769 backend tests, 15 archive tests,
four workspace core tests, 27 browser tests, seven frontend checks, frontend
lint and the production build. The 34 isolated PostgreSQL integration cases were
skipped locally; no database migration is included. The new MCP tests use an
actual client/server transport with mocked owning services and synthetic private
markers, covering nested responses, failures, malformed arguments and frame
errors. They are not live production tests.

Candidate archive: `dist/video-moment-finder-0.4.3.zip`; its matching
`.validation.json` records the SHA-256 and inventory. The generated report
intentionally leaves submission eligibility false until online gates are met.

## Production verification — November 12

Ten public HTTP checks passed, including OAuth discovery, the protected MCP
endpoint and public review URLs. Actual connected-tool calls passed for library,
allowance, status, source refresh, text search, transcript, high-resolution frame
and prepared learning-view responses. Their model-visible records omit account
identifiers, trial history, creation timestamps, signed media URLs and backend
diagnostics. A malformed source identifier returned a safe `INVALID_ARGUMENT`
without echoing the input. The connector does not expose component-only metadata
to this client; its complete contract remains covered by transport regression
tests, not claimed as an inspected live component payload.

R2 object headers confirm the retained reviewer copy expires January 4, 2027,
and the October 13 ordinary copy expires November 12, 2026 at 20:32 UTC. Bucket
lifecycle/CORS listing returned `AccessDenied` under the existing object-scoped
credential; no permission expansion was attempted. The retained reviewer copy
has the same two-segment transcript. Ordinary source expiry must not be confused
with the reviewer exception. No lifecycle rule or object expiry was changed.

The developer authorized one original synthetic fixture upload. Its 464,090
bytes matched the published SHA-256; PUT returned HTTP 200 without forwarding
Authorization. Completion returned the minimized nested record and processing
was observed. The ledger recorded exactly one 500-unit indexing charge, with
1,047 units remaining after eight retrieval units. A duplicate-completion live
probe was blocked by automatic approval review; local retry/idempotency checks
passed. This does not establish native-host file transfer or actual mobile-host
behavior. Those checks and final developer attestations remain separate gates.

## Deployment and final review gates

1. PR #103 is merged and deployed. Deploy the processor-retention correction;
   no migration or change to consent, tariffs or CSP is needed.
2. Finish the refreshed tool scan. Verify the 0.4.3 resource and inspect complete success
   and error responses against the table above, including component metadata.
   Check owned playback and text/image search after the response change.
3. Package 0.4.3 is saved in the existing submission with the same MCP app.
   Complete the new metadata/skill/MCP scans and read back the saved version.
4. Reconfirm reviewer login, adequate existing units, retained ready sample and
   attachment access. Run updated P1–P5/N1–N3 on the exact saved release, including
   upload retry/cancel and actual desktop/mobile host behavior. Replace or
   supplement the earlier demo if it misrepresents that release. Keep credentials
   in the secure portal, never the ZIP or public report.
5. The developer reviews privacy/retention statements and completes the current
   legal/policy attestations. Then submit for review. Neither local tests nor a
   draft upload establishes approval or publication.

Suggested reviewer explanation, to use only after the deployment and live checks:

> We minimized each tool's response with explicit nested field allowlists.
> Library/status responses no longer include creation timestamps, signed media
> links or raw processing errors; workspace allowance omits trial/grant history.
> Search thumbnails and playback capabilities are restricted to component
> metadata. The upload-start capability is retained only because the client
> needs it for the requested file transfer. Tool validation/backend failures and
> failed-frame entries now use safe messages. The privacy policy describes the
> necessary source identifiers and content, nested/component data, OAuth,
> analytics, diagnostics, payment records, recipients, retention and controls.
> Regression tests exercise complete MCP responses and private-marker failures.
