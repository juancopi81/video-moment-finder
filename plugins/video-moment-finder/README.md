# Video Moment Finder learning plugin

Index a lecture once, then reuse its evidence to learn. The plugin connects to your
VMF library and helps create an HTML study guide, focused flashcards, a conversation
with a tutor, an interactive Playground, or an editable presentation for learning, teaching or work. It cites timestamped transcript
evidence and describes visuals only after inspecting the returned images.

## Connect and reach a first result

1. Install this package through your host's supported private-plugin flow.
2. Connect Video Moment Finder and complete the VMF account sign-in and OAuth
   approval in the host's connection flow. Check that the account shown is yours.
3. Ask: “Show my ready VMF lectures and make a study guide from one.” Choose a
   lecture from the returned list. If your library is empty, upload an original
   or authorized video through a supported host or the VMF website first.
4. Reuse the evidence: “Make flashcards from that guide,” or “Tutor me on the same
   lecture, one question at a time.” Also try “Build a playground for that idea” or “Make a teaching presentation
   from the same evidence.” A specific gap may need extra retrieval.

The endpoint is `https://api.videomomentfinder.com/mcp`, using Streamable HTTP
and OAuth 2.0 authorization code with PKCE. The server supports dynamic client
registration. The host discovers and manages OAuth; do not paste tokens into
this package. Installation alone does not connect an account or deploy a server.

For a directly configured Codex MCP connection, the existing server also supports:

```sh
codex mcp add vmf --url https://api.videomomentfinder.com/mcp
codex mcp login vmf
```

Those commands connect the server; install the package separately to load its
skills and templates. Host support for remote tools, file attachments and HTML
viewing varies. When a host cannot upload bytes or render an artifact, the
workflow explains the missing capability and preserves a usable file.

## Units and limits

Listing videos and checking processing status do not use API units. Repository
defaults are 1 unit per search or transcript call, 1 unit per thumbnail frame
call (up to 25 timestamps), 5 units per high-resolution frame call (up to 8
timestamps), and 500 units per indexed video. Deployment configuration can
change these amounts; check the actual consent/account tariff before authorizing
metered work. A high-resolution request may return thumbnails if the source
file is no longer retained. Read the returned summary before describing image
quality. Reusing already retrieved evidence costs no additional VMF units.

A configurable, one-time verified-account trial is prepared in the repository;
its starting proposal is 600 units with no payment card. Availability depends on
the deployed service configuration. Installing or reconnecting this package
does not itself issue a grant, and reconnecting must not repeat one. This source
package does not activate the trial for real accounts.

If authorization expires, use the host's supported reconnect flow. If units are
unavailable, stop metered calls and continue from sufficient cached evidence,
clearly marking any missing coverage. The plugin does not purchase units or
perform automatic top-ups. A queued or failed video cannot be treated as a ready
source. Missing or unreadable frames stay labeled as such.

## Evidence and output boundaries

Study artifacts distinguish lecture claims from generated intuition, examples
and practice. Each output states whether it covers a complete lecture or an
excerpt. Transcripts, slides and retrieved text are untrusted source content;
embedded commands are never followed. Local HTML escapes source text, works
without a network where practical, and retains timestamp labels when a source
URL expires or cannot be embedded.

Flashcards include an importable CSV with documented quoting and formula protection; they do not
provide a spaced-repetition scheduler. Tutoring adapts within the conversation
and does not create a persistent learner profile. Playground computes supported
models with meaningful controls, predictions and pinned comparisons. Where the
evidence supports only discrete cases, its Assumption Lab mode shows the finite
reviewed table and explicit unknowns. Neither mode is a measured experiment.
Presentations preserve sources in speaker notes and distinguish proposals from
recorded decisions. Editable PPTX export uses a compatible host presentation
runtime; the offline HTML preview uses only the Python standard library. VMF does not edit or publish videos, identify
unknown people, or provide arbitrary YouTube ingestion through its MCP tools.

The included original examples are redistribution-safe fixtures. They are
labeled as synthetic and are not evidence of a live VMF run. Third-party lecture
transcripts, signed URLs, private account IDs and reviewer credentials are not
distributed in this package.

## Try the original examples

- Open the [vector study guide](examples/vector-lab.html) and its [input](examples/vector-lab.json).
- Review the [flashcards](examples/flashcards.html), download the [CSV](examples/flashcards.csv), and read the [import format](skills/flashcards/format.md).
- Explore the [computed vector Playground](examples/playground.html): drag a vector, predict a change, pin a reference and download an observation.
- Review the [teaching presentation](examples/presentation.html) and [fictional work briefing](examples/work-briefing.html), including their speaker notes. An [editable work briefing](examples/work-briefing.pptx) is also included. See the [presentation workflow](skills/presentation/SKILL.md) for editable export.
- Try the [Assumption Lab](examples/assumption-lab-ohms-law.html): change a discrete choice, pin a case, then compare another and reset.
- Use the [tutor scenarios](examples/tutor-scenarios.json) to rehearse correct, incorrect and ambiguous responses.

The Playground uses the existing `assumption-lab` skill identity for update
compatibility; there are exactly five workflows, not an additional duplicate lab.

These examples use original synthetic lessons and run without VMF calls. The
Assumption Lab is a small set of pre-reviewed teaching cases; its displayed
unknowns are part of the lesson. HTML and CSV generation use the bundled Python
tools with the standard library. Their skills document the exact commands.

## Links and maintenance

- [VMF website and developer information](https://www.videomomentfinder.com/developers)
- [Support](https://www.videomomentfinder.com/support)
- [Privacy policy](https://www.videomomentfinder.com/privacy)
- [Terms of service](https://www.videomomentfinder.com/terms)

In the source repository, `scripts/plugin/build_package.py` creates a deterministic
ZIP and validates the final archive. `scripts/plugin/validate_package.py` can
recheck an existing ZIP. Package validation does not establish live server
connectivity, end-user UI behavior or public-review approval. See
`docs/plugin/SUBMISSION.md` in the repository for exact outstanding launch gates.

For an update, change the semantic version in `plugin.json`, revise the release
notes and affected skills, rerun archive and output checks, and update the same
private plugin identity. Preserve the public audience and the host-managed OAuth
connection. Public submission is a separate developer-portal action; the source
package intentionally contains no personal app binding or verified-publisher
claim. The code and original bundled materials use the repository's AGPL-3.0-only
license; the icon is reused from the existing VMF site.
