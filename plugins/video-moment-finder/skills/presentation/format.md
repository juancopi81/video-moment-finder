# Presentation format and portable export

Use the common `format_version`, `title`, `lecture`, `coverage` and `sources` envelope from the [evidence contract](../../references/evidence-format.md). Add `audience`, `purpose` and a `slides` array of 3–16 items. Each slide has:

- `title`, `body` (up to three short paragraphs), `notes` (presenter guidance).
- `layout`: `cover`, `statement`, `evidence`, `comparison`, or `exercise`.
- `origin`: `source`, `generated`, or `mixed`, with nonempty `citations` referencing the envelope's source IDs.
- Optional `headline` (a large equation or result), `image_source` (a cited frame ID), and `columns` (exactly two objects with `title` and `body`, for comparison layout).

See the [original teaching deck](../../examples/presentation.json) and [fictional work briefing](../../examples/work-briefing.json). The renderer escapes source text, embeds bounded PNG/JPEG frames from the evidence directory, and emits a sanitized prepared JSON file for PPTX export. It never executes source markup or fetches remote assets. Missing images have a visible fallback; source references survive. Image alternatives and provenance accompany the image.

`tools/presentation.mjs` uses JavaScript ES modules and the host-provided `@oai/artifact-tool`. If bare imports cannot resolve, use the host's runtime discovery to link the private build directory's `node_modules` to the supplied modules, and copy the exporter there unchanged. Do not commit that link or a machine-specific path. It creates `candidate.pptx`, `slide-N.png` and per-slide layout files in the build directory. This is a draft: follow the host presentation skill's finalization/visual review before delivering it. An optional third argument sets an installed font family (default Arial).

No Node package or PowerPoint runtime is added to VMF's production service. The HTML preview has no dependencies. It can be printed to PDF through the host/browser, but the workflow must not claim a PDF was created unless the actual file exists and was inspected. Keep the prepared input and validation receipts private when they contain private lecture evidence.
