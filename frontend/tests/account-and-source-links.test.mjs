import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Exercise the production helpers with the existing TypeScript dependency; no
// browser fixtures, extra packages, or network calls are needed for these cases.
async function importTypeScript(relativePath) {
  const source = await readFile(new URL(relativePath, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
  });
  return import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
}

const { apiUnitTrialMessage } = await importTypeScript("../src/lib/api-unit-trial.ts");
const { parseVideoTimestamp, boundedVideoTimestamp, buildTimestampUrl } = await importTypeScript("../src/lib/video-timestamps.ts");

test("an old or disabled server does not advertise an unissued trial", () => {
  for (const summary of [
    {},
    { trial_enabled: false, trial_status: "disabled", trial_allowance_units: 600 },
    { trial_enabled: false, trial_status: "verification_required", trial_allowance_units: 600 },
    { trial_enabled: true, trial_status: "granted" },
  ]) {
    assert.equal(apiUnitTrialMessage(summary), null);
  }
});

test("verification guidance describes eligibility without promising a grant", () => {
  const message = apiUnitTrialMessage({ trial_enabled: true, trial_status: "verification_required", trial_allowance_units: 600 });
  assert.match(message, /Verify your primary email/);
  assert.match(message, /check trial eligibility/);
  assert.doesNotMatch(message, /600|granted|buy|upgrade/i);
  assert.match(apiUnitTrialMessage({ trial_enabled: true, trial_status: "verification_unavailable" }), /temporarily unavailable/);
});

test("historical grants remain truthful after rollout is paused or units run out", () => {
  for (const trial_status of ["granted", "exhausted"]) {
    const message = apiUnitTrialMessage({ trial_enabled: false, trial_status, trial_units_granted: 100, trial_legacy_units_offset: 500, api_units_balance: 10_000 });
    assert.match(message, /100 API units granted/);
    assert.match(message, /500 units of the allowance/);
    assert.match(message, /Reconnecting does not add another trial/);
    assert.doesNotMatch(message, /remaining|10,000|buy|upgrade/i);
  }
});

test("a fully offset legacy trial does not appear to grant fresh units", () => {
  const message = apiUnitTrialMessage({ trial_status: "exhausted", trial_units_granted: 0, trial_legacy_units_offset: 600 });
  assert.match(message, /0 API units granted/);
  assert.match(message, /600 units of the allowance/);
});

test("source links accept unambiguous nonnegative seconds", () => {
  assert.equal(parseVideoTimestamp("223"), 223);
  assert.equal(parseVideoTimestamp("223.5"), 223.5);
  assert.equal(parseVideoTimestamp("0"), 0);
  for (const value of [undefined, "", "-1", "1e3", "Infinity", "NaN", "1:30", "20s", " 1 ", ["1", "2"], "9007199254740992"]) {
    assert.equal(parseVideoTimestamp(value), null, String(value));
  }
});

test("deep links wait for metadata and clamp the seek to the source duration", () => {
  assert.equal(boundedVideoTimestamp(223.5, 600), 223.5);
  assert.equal(boundedVideoTimestamp(800, 600), 600);
  assert.equal(boundedVideoTimestamp(0, 600), 0);
  for (const [seconds, duration] of [[null, 600], [223, NaN], [223, Infinity], [223, 0], [-1, 600], [Infinity, 600]]) {
    assert.equal(boundedVideoTimestamp(seconds, duration), null);
  }
});

test("external timestamp links preserve source identity and reject active URL schemes", () => {
  const url = new URL(buildTimestampUrl("https://www.youtube.com/watch?v=abc&t=1", 223.5));
  assert.equal(url.searchParams.get("v"), "abc");
  assert.equal(url.searchParams.get("t"), "223");
  for (const baseUrl of ["javascript:alert(1)", "data:text/html,<h1>unsafe</h1>", "file:///video.mp4", "/relative"]) {
    assert.equal(buildTimestampUrl(baseUrl, 223), null);
  }
  assert.equal(buildTimestampUrl("https://example.com/video", -1), null);
  assert.equal(buildTimestampUrl("https://example.com/video", Infinity), null);
});
