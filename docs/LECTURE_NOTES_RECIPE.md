# Turn a Lecture Video into Study Notes with Claude

This is the canonical recipe for turning an indexed lecture video into
structured Markdown study notes using Video Moment Finder plus Claude (or any
MCP-compatible client). It's built on the `lecture_notes` MCP prompt and the
`get_transcript` / `get_frames` tools described in `docs/MCP_GUIDE.md`.

Audience: a Video Moment Finder user with the Claude connector configured (or,
for the manual variant, anyone with a `vmf_` API key and REST access).

## Prerequisites

- The lecture video is already indexed and `ready` (see `docs/CLI_API_GUIDE.md`
  or `docs/MCP_GUIDE.md` for upload and status polling).
- The Claude connector is added and authorized (server URL
  `https://api.videomomentfinder.com/mcp`), **or** you have a `vmf_` API key
  for the REST/manual variant below.
- A positive Developer Pack API-unit balance (see the cost estimate below).

## One-Liner: The `lecture_notes` MCP Prompt

With the connector configured, invoke the `lecture_notes` prompt with the
video's UUID:

```text
/lecture_notes video_id=<video_id>
```

Optional arguments:

- `course_context` — course name, lecture number, and/or related links to
  include at the top of the notes.
- `own_notes` — text of your own handwritten notes. When provided, Claude
  treats your notes as the primary skeleton and uses the video to verify,
  correct, complete, and enrich them, flagging any conflicts explicitly. If
  you'd rather share photos of your notes, paste them directly into the chat
  instead of using this argument.

Claude runs the workflow described below and returns one Markdown document.

## Manual Equivalent (REST / CLI Users)

If you're driving the REST API directly instead of through the MCP prompt,
give Claude (or any agent) this equivalent instruction:

```text
Turn the indexed lecture video <video_id> into polished Markdown study notes.

1. GET /api/v1/videos/<video_id> to confirm status is "ready".
2. GET /api/v1/videos/<video_id>/transcript for the full transcript.
3. Read the transcript and identify the lecture's natural sections and
   "board moments" — places where the speaker references something visual
   without fully describing it ("let's draw...", "as you can see here...",
   "this diagram...").
4. For the 5-15 most important board moments, POST
   /api/v1/videos/<video_id>/frames with resolution="high" (or "thumb" if
   the source isn't retained) at a timestamp near the END of each
   explanation, since board content accumulates while the speaker writes.
5. Describe each visual faithfully in the notes as text, LaTeX, or a
   described diagram — never as a "see figure" reference.
6. Write one Markdown document: title + metadata, a source-status line
   disclosing the notes are AI-assisted and generated from the transcript
   and frames, numbered sections following the lecture's structure, all math
   in LaTeX with key results boxed, and a final "Main Takeaways" list.
```

This is exactly what the `lecture_notes` prompt does internally — the
one-liner is just a shortcut for connector users.

## What the Workflow Does, Step by Step

1. **Confirm readiness.** Calls `get_video_status` (or `GET
   /api/v1/videos/{video_id}`) to make sure the video finished indexing.
2. **Fetch the full transcript.** Calls `get_transcript` (or `GET
   .../transcript`) once, with no time filter, to read everything the
   speaker said along with per-segment timestamps.
3. **Identify board moments.** Reads the transcript for the lecture's natural
   sections and for moments where the speaker points at something visual
   without spelling it out in words — the transcript alone won't capture
   what was drawn or written.
4. **Fetch high-res frames near the END of each explanation.** For the 5-15
   most important board moments, it calls `get_frames`. Board content (an
   equation, a diagram) accumulates while the speaker writes it, so the
   workflow deliberately requests a timestamp a few seconds after the
   speaker finishes describing the visual, not the moment they start — a
   frame taken too early is a half-finished board. If a frame comes back
   unclear, it requests 2-3 nearby timestamps and uses the clearest one.
5. **Describe visuals as LaTeX/text.** Every visual that made it into a
   frame is transcribed into the notes directly — equations as LaTeX, plots
   and diagrams as faithful text descriptions — rather than left as a
   dangling reference to "the figure above."
6. **Assemble one structured Markdown document** with:
   - a title and lecture/course metadata block at the top,
   - a source-status line disclosing the notes are AI-generated from the
     video's transcript and frames (and from your own notes, if you supplied
     them),
   - numbered sections following the lecture's own structure,
   - all math in LaTeX (`$$` blocks), with key results in `\boxed{}`,
   - a final **Main Takeaways** bullet list.

## Cost Estimate

| Step | Cost |
| --- | --- |
| Index the video (one time) | ~500 API units |
| Each notes session | typically ~6-16 API units |

A notes session is 1 transcript fetch (1 unit) plus 1-4 frame calls (1 unit
for thumbnails, 5 units for high-res, per call — not per frame): typically
~6-16 units (one transcript fetch plus one to three high-res frame calls);
minimum ~2, maximum ~21. That cost is **independent of video length**: a
10-minute clip and a 90-minute lecture both cost one transcript fetch and a
handful of frame calls, because both are billed per API call, not per minute
of footage. The video only needs to be indexed once; you can re-run the notes
workflow as many times as you like afterward for the same typical ~6-16 units
per run.

None of the above includes the LLM tokens spent reading the transcript,
reasoning about board moments, and writing the notes — those are spent by
your own Claude session (or API key) outside of Video Moment Finder's
billing.

## Limitations

- **YouTube imports fall back to 320px thumbnails.** `get_frames` only
  extracts sharp high-resolution frames from a *retained* source video —
  currently, direct uploads only. YouTube imports don't retain the source, so
  `get_frames` automatically falls back to the stored 1-fps thumbnails, which
  may not resolve dense board writing or small text. For lectures with heavy
  board/slide content, a direct upload gives materially better frames than a
  YouTube import.
- **ASR transcripts can misrender spoken math.** Automatic speech recognition
  turns spoken math into plain prose ("f prime of x" instead of `f'(x)`), and
  can mishear technical terms entirely. Treat the transcript as a guide to
  *where* things were said, not as a verbatim source for equations — that's
  exactly why the workflow leans on frames for anything mathematical.
- **Always disclose AI assistance.** The notes this workflow produces are
  AI-generated from a transcript and a small number of frames, not a full
  human review of the lecture. Keep the source-status line in the final
  document, and treat the notes as a strong first draft to check against the
  original recording before relying on them for anything high-stakes (exams,
  citations, etc.).
