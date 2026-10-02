# Assumption lab format, version 1

Reuse the shared `format_version`, `title`, `lecture`, `coverage`, and `sources`
fields. The lab renderer uses those evidence fields and its own `lab` object;
study-guide sections and questions are not required. Do not store private IDs,
account metadata, credentials, signed URLs, or full third-party transcripts in
redistributable files.

See the complete original [Ohm's-law example](../../../examples/assumption-lab-ohms-law.json).
Its calculations are fixture values, not a VMF measurement or a hardware recommendation.

`lab` requires:

- `question`, `premise`, `model_limit`: nonempty text describing the extension
  being tested, why it matters, and what the model cannot establish.
- `controls`: one to three controls. Each has a unique lowercase `id`, a `label`,
  and two or three `options` with unique `id` and `label` fields.
- `baseline`: one existing state ID.
- `states`: exactly one reviewed state for every combination of options, with
  no duplicate combination. A 2 × 2 lab therefore has four states; at most 27
  states are allowed by the control bounds.

Each state requires:

- Unique `id`; `when` maps every control ID to a valid option ID.
- `title`, `outcome`, `explanation`: plain text, not HTML or executable formulas.
- `entities`: exactly two nonempty strings naming the items/conditions in the
  simple visual comparison. These are generated labels, not source screenshots.
- `boundary`: `applies`, `caution`, or `not-supported`; `boundary_note` explains
  that judgment. `applies` means the given model's conditions hold, not that the
  model proves a real-world outcome.
- Nonempty arrays `assumptions`, `reasoning`, `unknowns`, and `citations`. Use
  real source IDs. Unknowns should name concrete missing information; "none"
  is not an adequate account of a bounded teaching model.
- `next_step`: a suggested next controlled change or evidence check.

The renderer validates a complete finite state table and citation references.
It does not evaluate formulas, retrieve evidence, verify scientific reasoning,
or grade learner notes. Text is escaped once; the script receives only a small
allowlist of fields. There is no remote JavaScript, external font, tracking,
local storage, or server request. Frame bytes are embedded from safe relative
PNG/JPEG paths using the shared helper. Missing frames produce readable notes.

From the plugin directory:

```sh
python tools/assumption_lab.py examples/assumption-lab-ohms-law.json examples/assumption-lab-ohms-law.html
```

Pinning copies the current reviewed case into the left comparison panel. Reset
restores both panels and all controls to the supplied baseline; it preserves
the learner's unsaved note. Download creates a plain-text comparison with the
current note and source IDs. Keep the HTML alongside it for timestamps and
full evidence notes. These actions use no VMF units and make no network calls.
