# Deck format and CSV import

`tools/flashcards.py` reads JSON with `artifact: "flashcards"`, `format_version: 1`, nonempty `title`, and the shared `lecture`, `coverage`, and `sources` objects from the [evidence contract](../../references/evidence-format.md). `subtitle` is optional. Image paths are relative to the JSON file and must stay inside its directory.

`cards` is a nonempty ordered array:

```json
{
  "id": "dot-product-sign",
  "kind": "concept",
  "origin": "generated",
  "front": "Can a real vector have a negative dot product with itself?",
  "hint": "Expand the coordinate formula.",
  "back": "No. It is a sum of squares, so it is nonnegative.",
  "rationale": "For v = (a, b), v · v = a² + b² ≥ 0. This example assumes real coordinates.",
  "citations": ["s-definition"],
  "tags": ["dot-product", "sign"]
}
```

- IDs use lowercase letters, digits, and hyphens, starting with a letter, maximum 64 characters. Card IDs and normalized question strings must be unique.
- `kind` is `recall`, `concept`, or `application`. `origin` is `lecture` or `generated` and describes the answer, not whether the question appeared in the lecture.
- Every card cites one or more existing evidence IDs. A `lecture` answer needs non-synthetic evidence. Generated answers require nonempty `rationale` explaining assumptions/derivation; a citation alone does not make a new example a lecture claim.
- `hint` is optional. `tags` are optional lowercase tokens with digits, hyphens, or underscores. All content fields are plain text. NUL and non-whitespace ASCII control characters are rejected.

The CSV is deterministic: UTF-8 without a BOM, comma delimiter, three columns in the order `Front,Back,Tags`, a header row, CRLF record endings, and every field double-quoted. Embedded quotes are doubled. Tabs and newlines inside fields are preserved within quoted fields; do not parse by splitting lines or commas. Card order follows JSON order; tags are space-separated and de-duplicated in first-seen order. The Back field includes the answer, origin, generated reasoning when present, lecture title, coverage, and source IDs/timestamps/VMF links. Thus importing only the first two columns still preserves the evidence references.

All exported cells are checked for formula injection. If the first significant character is `=`, `+`, `-`, or `@` after leading Unicode whitespace/control/format characters, or if the field begins with a tab/newline/carriage return, an ASCII apostrophe is prepended. CSV quoting by itself is insufficient protection. Plain-text importers may show the apostrophe; exact original text stays in JSON. Do not strip it in spreadsheet workflows or claim universal safety after downstream editing/re-exporting.

For Anki or a comparable importer: select comma-separated UTF-8 input, skip the header, map the first two fields to Front and Back, optionally map Tags, and disable HTML interpretation. Use an importer supporting quoted multiline fields and preview several cards. The package validates CSV structure; it does not bundle or claim end-user verification of an external card app. No scheduling fields are written.

HTML escapes text once and embeds the CSV as a base64 data download. It contains no external scripts, fonts, trackers, or source-provided markup. Card navigation is optional JavaScript; native hint/answer disclosure remains usable in the full list without JavaScript. The renderer does not interpret source instructions or equations as code.
