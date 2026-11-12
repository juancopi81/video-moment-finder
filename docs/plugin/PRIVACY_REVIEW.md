# Privacy and resubmission review — 0.4.3

Checked November 12, 2026. This is a prepared candidate, not a deployed or
resubmitted release. The portal shows 0.4.2, Changes required, Not published;
the associated MCP connection is authorized and domain-verified. No field-level
MCP findings are available in that view. The rejection email identifies
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
- Provider logs/backups: exact current deletion windows have not been verified.
  The policy describes provider-controlled retention without inventing a VMF
  guarantee. Confirm these settings before making any more specific promise.

No deletion jobs, retention settings, grants, balances or credentials are changed.
The owner still needs to review the policy as the service operator before it is
published. Unlinking in a host is not represented as deleting the account or as
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
- Developer, support, privacy and terms pages are reachable. The deployed
  privacy page still needs replacement by this candidate.
- The existing [0.4.2 demo](https://youtu.be/0fsteByQf9g) loads and plays; duration
  is approximately 153 seconds. This was checked in the signed-in browser, not
  an anonymous session. It remains an earlier-version walkthrough, not evidence
  that the 0.4.3 privacy changes have been deployed or tested in ChatGPT.
- The original reviewer MP4 downloads anonymously: 464,090 bytes, SHA-256
  `c3b39b3f5bf4652042ed3e9ffd7ecc4f2eaa9ed659c8dd848869f8684e2627c4`.
- Desktop plus 390/768-pixel fixture tests verify search thumbnails supplied
  only in metadata, playback seeking, learning controls, no horizontal overflow
  and clean chat handoffs. They do not establish actual ChatGPT mobile behavior.

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

## Deployment and final review gates

1. Review and merge the patch; deploy the API/workspace and frontend policy.
   No migration, new dependency or change to consent, tariffs or CSP is needed.
2. Refresh tool metadata. Verify the 0.4.3 resource and inspect complete success
   and error responses against the table above, including component metadata.
   Check owned playback and text/image search after the response change.
3. Upload the candidate to the existing rejected submission, preserving the
   configured MCP app. Run the new scans; the rejected view currently disables
   rescan and diagnostic-download controls. Read back the saved version.
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
