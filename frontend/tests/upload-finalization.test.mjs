import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

test("a transfer that outlasts the initial token finalizes the same video once", async () => {
  const source = await readFile(new URL("../src/app/home-content.tsx", import.meta.url), "utf8");
  const file = ts.createSourceFile("home-content.tsx", source, ts.ScriptTarget.Latest, true);
  const home = file.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === "HomeContent");
  const upload = home?.body.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === "handleUpload");
  assert.ok(upload, "Find the upload handler used by the homepage");
  const { outputText } = ts.transpileModule(upload.getText(file), {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });

  const fileToUpload = { name: "slow.mp4", size: 123, type: "video/mp4" };
  const tokens = new Map();
  const requests = [];
  const destinations = [];
  const failures = [];
  let seconds = 0;
  let transfers = 0;

  // Run the production handler with a virtual clock and fake network boundaries.
  // Only the transfer advances time, so the first token expires during the PUT.
  await runInNewContext(outputText + "\nhandleUpload({ preventDefault() {} });", {
    API_URL: "https://api.example.test",
    MAX_UPLOAD_FILE_SIZE_BYTES: 8 * 1024 ** 3,
    uploadFile: fileToUpload,
    setError: (message) => { if (message) failures.push(message); },
    setYoutubeFallbackDetail() {},
    setIsUploading() {},
    setUploadProgress() {},
    router: { push: (path) => destinations.push(path) },
    async getToken() {
      const token = `token-${seconds}`;
      tokens.set(token, seconds + 60);
      return token;
    },
    async fetch(url, options) {
      const path = new URL(url).pathname;
      const token = options.headers.Authorization.slice("Bearer ".length);
      const body = JSON.parse(options.body);
      requests.push({ path, token, body });

      if ((tokens.get(token) ?? 0) <= seconds) {
        return { ok: false, status: 401, detail: "Invalid authentication token" };
      }
      if (path.endsWith("/init")) {
        return { ok: true, json: async () => ({ video_id: "video-1", upload_url: "https://storage.example.test/transfer" }) };
      }
      return { ok: true, json: async () => ({ id: body.video_id }) };
    },
    parseApiError: async (response) => response.detail,
    posthog: undefined,
    XMLHttpRequest: class {
      upload = {};
      status = 200;
      open(method, url) {
        assert.equal(method, "PUT");
        assert.equal(url, "https://storage.example.test/transfer");
      }
      setRequestHeader() {}
      send(file) {
        assert.equal(file, fileToUpload);
        transfers++;
        seconds += 75;
        this.onload();
      }
    },
  });

  assert.deepEqual(failures, []);
  assert.equal(transfers, 1);
  assert.deepEqual(requests, [
    { path: "/api/v1/videos/upload/init", token: "token-0", body: { filename: "slow.mp4", content_type: "video/mp4" } },
    { path: "/api/v1/videos/upload/complete", token: "token-75", body: { video_id: "video-1", filename: "slow.mp4" } },
  ]);
  assert.deepEqual(destinations, ["/video/video-1"]);
});
