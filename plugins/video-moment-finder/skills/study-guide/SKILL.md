---
name: study-guide
description: Turn an indexed lecture, or a clearly bounded excerpt, into a source-grounded HTML study guide with inspected visual evidence, explanations, and practice. Use when the learner asks for lecture notes, a study guide, or a portable learning artifact.
---

# Make a study guide

Use VMF's existing six MCP tools; discover their installed schemas rather than inventing arguments. Start with the learner's selected ready video, or use `list_videos` and clarify only if the intended lecture is ambiguous. Follow [shared evidence rules](../../references/evidence-format.md) before retrieving or writing.

1. Confirm status. A failed/queued/processing video is not evidence. Explain the state and offer another ready video or existing supplied material. Do not re-upload/index without the user's instruction and a checked allowance.
2. Set scope: whole lecture or specific timestamps/topic. Retrieve the transcript once, reuse task-local evidence, and search only to close a specific gap. Report missing transcript sections. An excerpt must never be titled or described as full-lecture coverage.
3. Select visual moments by what the explanation needs. Search can locate a slide; requesting a frame only at the end of speech can instead capture the speaker. Inspect the returned image content. Check requested versus actual timestamps and per-frame errors. A successful image response does not prove it is useful. If source retention is absent, use the returned thumbnails without claiming high resolution. Make one bounded nearby attempt if useful; otherwise explain the visual limitation.
4. Write a coherent explanation: objectives, intuitive model, precise result/conditions, one worked example, common confusion, and practice. Distinguish the lecture's claims from generated examples, intuition, and exercises. Cite exact source IDs/timestamps for substantive lecture claims. Resolve contradictions explicitly; unreadable board writing and suspect ASR are uncertainty, not permission to invent equations.
5. Create a JSON guide using the shared format. Use `tools/render.py input.json output.html --template templates/study-guide.html` relative to this plugin's directory. The renderer produces offline HTML, accessible navigation, keyboard-operable answer reveals, inline styling, and optional local-image embedding. If the host cannot run files, produce equivalent HTML with the same escaping and source disclosures; do not claim rendering or interaction tests.
6. Check at least five substantive claims, answers, or visual interpretations against cached evidence. Open/render the HTML with supported tools when available; check narrow-screen navigation, source links, image fallback, and each interaction. Otherwise report static validation only.

The tool renders text as text, not source-supplied HTML. Keep full third-party transcripts and lecture frames private; publish examples only when redistribution is permitted. The final artifact should say what it covers, link to the source when available, and remain readable when playback URLs expire.

If authorization expires, request reconnection through the host's supported flow and pause new calls. If units are unavailable, explain the limit without purchases, upgrades, or checkout links; offer already-cached evidence or a learner-supplied excerpt. Never request credentials in chat.
