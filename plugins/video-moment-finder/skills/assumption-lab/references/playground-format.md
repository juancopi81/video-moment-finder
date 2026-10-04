# Computed playground input

Use the shared `format_version`, `title`, `lecture`, `coverage` and `sources` fields from the [evidence contract](../../../references/evidence-format.md). The renderer accepts only the following model:

```json
{
  "playground": {
    "model": "dot-product-2d",
    "question": "What does the sign tell you about the arrows?",
    "initial": {"v": [3, 4], "w": [2, 0]},
    "citations": ["s-coordinate"],
    "model_limit": "Real 2D Euclidean vectors; generated teaching model, not measured data.",
    "extension_note": "The zero-vector boundary and practice values are generated extensions."
  }
}
```

Coordinates range from −5 to 5 with 0.1 control steps. The model computes dot product, norms, signed scalar projection of w along v, projection vector and angle. It returns undefined angle when either norm is zero and undefined projection when v is zero. Internal calculations use unrounded values; the UI rounds to two decimal places.

The [original example](../../../examples/playground.json) has no private video identity. The [rendered example](../../../examples/playground.html) includes draggable tips, keyboard coordinate inputs, real calculations, presets, a prediction experiment, pin/reset and a plain-text observation download. All model code is bundled in the page. No network or persistent storage is used.

For another topic, write a model suited to its actual equations and document its domain, assumptions, error and independent checks. This schema is a ready example, not a universal simulation language. The [finite Assumption Lab](lab-format.md) remains useful when the source cannot justify continuous calculations.
