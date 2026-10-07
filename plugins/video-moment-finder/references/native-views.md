# Native learning views

Use `render_learning_view` when available and the user wants to work inside VMF. Discover its real schema. The six evidence tools acquire evidence; this rendering tool displays prepared content without additional units. Follow the [shared evidence rules](evidence-format.md), inspect frames, and audit claims before rendering. Schema/ownership validation does not prove educational correctness.

Pass `view` and optional `companion_views` (at most three) containing other already-prepared workflows from this conversation. All refer to the same owned ready UUID and different kinds. This keeps four displays together without repeated retrieval and repopulates a remounted component from its tool result. The tutor stays in the conversation. Views are not stored as a permanent cross-chat library. Do not promise that unsaved card navigation, experiments or selections survive remounting.

```json
{
  "kind": "study-guide",
  "video_id": "00000000-0000-4000-8000-000000000001",
  "title": "A precise idea",
  "subtitle": "What the learner can do",
  "coverage": {"kind": "excerpt", "start_s": 0, "end_s": 30, "gaps": []},
  "sources": [{"id": "definition", "kind": "transcript", "start_s": 2, "end_s": 8, "summary": "What the source supports"}],
  "sections": [{"title": "The rule", "origin": "lecture", "paragraphs": ["A supported explanation"], "bullets": [], "citations": ["definition"]}]
}
```

The example UUID is not an indexed video. Supply the connected account's real ready UUID. Fields are plain text, never HTML, scripts, arbitrary URLs, local paths, credentials or signed media links. Sources have unique lowercase IDs, kind `transcript` or inspected `frame`, finite nonnegative ordered bounds, and summaries. Every cited item references included evidence IDs. Full coverage means complete spoken coverage with disclosed visual sampling, not every frame watched; otherwise use excerpt. State missing/contradictory evidence in `gaps`.

Use readable plain text and Unicode notation for native formulas; this view does not interpret Markdown or LaTeX. Rich offline HTML, inspected images and supported presentation exports remain available through the corresponding skill. Do not claim that the native preview executes a bespoke HTML application or renders every offline diagram.

Exactly one content field matches `kind`:

- **study-guide:** nonempty `sections` with `title`, `origin` (`lecture` or `generated`), `paragraphs` and/or `bullets`, and `citations`. Label generated assumptions/derivations. Native source buttons seek the player. Use the existing renderer for a requested offline HTML export.
- **flashcards:** nonempty `cards` with `front`, `back`, `origin`, `rationale`, `citations`, optional `hint` and lowercase `tags`. Generated answers require reasoning; questions are unique. Native cards reveal answers, advance and export CSV with evidence in Back and formula protection. They do not provide multiple choice or spaced-repetition scheduling.
- **presentation:** nonempty `slides` with `title`, `bullets` (1–8), `notes`, `origin`, and `citations`. Notes distinguish source from generated interpretation. Native preview and notes are immediate; editable PowerPoint still uses the packaged skill and a compatible host runtime. JSON is not PPTX.
- **playground:** `playground` includes `question`, `model_limit`, `extension_note` and a supported `model`. `dot-product-2d` adds `initial: {"v":[3,4],"w":[2,0]}` within [-5,5] and `citations`. It reuses the checked vector math; zero-vector angle and projection onto zero are undefined. Choose this only when the source justifies it. Otherwise use `reviewed-cases` with `controls` and a complete `cases` table. Controls have `id`, `label`, `options: [{"id":"choice","label":"Choice"}]`. Cases have `when: {"control-id":"choice-id"}`, `title`, `outcome`, `explanation`, `assumptions`, `unknowns`, and `citations`. Include exactly one reviewed case per combination. Unknown outcomes stay unknown; never invent numerical simulations or execute generated code.

Native playback receives fresh owned capabilities through UI-only metadata. Downloads retain stable timestamps, not temporary URLs. The user can explicitly inspect a frame in the workspace; rendering does not automatically retrieve/bill more frames. Preserve privately inspected visuals in an offline guide when requested. An unavailable retained source must be disclosed.

If native rendering is unavailable, deliver the existing offline artifact or concise cited chat result. Do not repeatedly reopen unsupported UI. Test the intended host before claiming its sidebar, picker, direct transfer or placement works.
