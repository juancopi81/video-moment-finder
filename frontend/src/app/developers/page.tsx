import Link from "next/link";

const connectorSteps = [
  "Add a custom connector in Claude with server URL https://api.videomomentfinder.com/mcp.",
  "Use the static client credentials from your secure review or internal test configuration.",
  "Click Connect, then sign in to Video Moment Finder if needed.",
  "Buy a Developer Pack if your API-unit balance is zero.",
  "Review the four MCP tools and approve access.",
];

const promptExamples = [
  {
    prompt: "List my recent videos in Video Moment Finder.",
    expectation:
      "Claude uses list_videos and returns recent IDs, status, and source details.",
  },
  {
    prompt: "Check whether video <video_id> is ready and tell me if it failed.",
    expectation:
      "Claude uses get_video_status and reports the current processing state or failure reason.",
  },
  {
    prompt:
      "Search video <video_id> for the moment they explain the model and give me timestamps.",
    expectation:
      "Claude uses search_video and returns timestamped text-search matches.",
  },
];

const tools = [
  {
    name: "upload_video",
    title: "Upload Video",
    summary:
      "Starts a presigned upload or completes it after the bytes are written.",
    tone: "Write",
  },
  {
    name: "get_video_status",
    title: "Get Video Status",
    summary: "Checks the current indexing state for one video.",
    tone: "Read",
  },
  {
    name: "list_videos",
    title: "List Videos",
    summary: "Lists recent videos for the connected account.",
    tone: "Read",
  },
  {
    name: "search_video",
    title: "Search Video",
    summary: "Runs text search against a ready video and returns timestamps.",
    tone: "Read",
  },
];

export default function DevelopersPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="rounded-3xl border border-zinc-200 bg-surface-card p-8 shadow-sm dark:border-zinc-800">
        <p className="text-sm font-medium uppercase tracking-[0.2em] text-accent">
          Developers
        </p>
        <h1 className="mt-3 font-heading text-4xl font-bold">
          Build with Video Moment Finder
        </h1>
        <p className="mt-4 max-w-3xl text-lg text-zinc-600 dark:text-zinc-400">
          OAuth-protected Claude connector plus a public REST API and CLI for
          upload, indexing, and text search.
        </p>
        <div className="mt-6 flex flex-wrap gap-3 text-sm">
          <Link
            href="/skill.md"
            className="rounded-lg bg-accent px-4 py-2 font-medium text-white"
          >
            Read /skill.md
          </Link>
          <a
            href="https://api.videomomentfinder.com/openapi.json"
            className="rounded-lg border border-zinc-300 px-4 py-2 font-medium text-zinc-900 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-900"
          >
            OpenAPI schema
          </a>
          <Link
            href="/dashboard/api"
            className="rounded-lg border border-zinc-300 px-4 py-2 font-medium text-zinc-900 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-900"
          >
            API dashboard
          </Link>
        </div>
      </div>

      <section className="mt-12 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
          <h2 className="font-heading text-2xl font-bold">Claude Connector</h2>
          <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
            The remote MCP server is available at{" "}
            <code className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs dark:bg-zinc-800">
              https://api.videomomentfinder.com/mcp
            </code>
            . It uses OAuth 2.0 authorization-code + PKCE and bills against
            Developer Pack API units.
          </p>
          <div className="mt-5 space-y-3">
            {tools.map((tool) => (
              <div
                key={tool.name}
                className="rounded-2xl border border-zinc-200 bg-white/80 p-4 dark:border-zinc-800 dark:bg-zinc-950/40"
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="font-medium text-zinc-900 dark:text-zinc-100">
                      {tool.title}
                    </p>
                    <p className="mt-1 text-xs text-zinc-500">
                      <code>{tool.name}</code>
                    </p>
                  </div>
                  <span className="rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                    {tool.tone}
                  </span>
                </div>
                <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
                  {tool.summary}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
          <h2 className="font-heading text-2xl font-bold">Developer Pack</h2>
          <p className="mt-3 text-3xl font-bold">$20</p>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            10,000 API units
          </p>
          <ul className="mt-4 space-y-2 text-sm text-zinc-700 dark:text-zinc-300">
            <li>500 units per indexed video</li>
            <li>1 unit per text query</li>
            <li>Required for Claude connector usage</li>
            <li>Also powers REST API and CLI indexing/search calls</li>
          </ul>
          <Link
            href="/dashboard/api"
            className="mt-5 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white"
          >
            Open API dashboard
          </Link>
        </div>
      </section>

      <section className="mt-12 grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
          <h2 className="font-heading text-2xl font-bold">Connect Flow</h2>
          <ol className="mt-4 list-inside list-decimal space-y-3 text-sm text-zinc-700 dark:text-zinc-300">
            {connectorSteps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-400">
            Reviewer note: keep the confidential client secret out of public
            docs. Share review credentials only through the secure review packet
            or environment configuration.
          </p>
        </div>

        <div className="rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
          <h2 className="font-heading text-2xl font-bold">REST API And CLI</h2>
          <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
            REST and CLI remain the canonical public contract for direct
            programmatic use. They still authenticate with{" "}
            <code className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs dark:bg-zinc-800">
              vmf_
            </code>{" "}
            API keys.
          </p>
          <div className="mt-5 space-y-3 text-sm">
            <a
              href="https://api.videomomentfinder.com/docs"
              className="block rounded-xl border border-zinc-200 px-4 py-3 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900"
            >
              Swagger UI
            </a>
            <a
              href="https://api.videomomentfinder.com/openapi.json"
              className="block rounded-xl border border-zinc-200 px-4 py-3 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900"
            >
              OpenAPI schema
            </a>
            <Link
              href="/skill.md"
              className="block rounded-xl border border-zinc-200 px-4 py-3 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900"
            >
              Public skill file
            </Link>
          </div>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Example Prompts</h2>
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          {promptExamples.map((example) => (
            <div
              key={example.prompt}
              className="rounded-2xl border border-zinc-200 bg-surface-card p-5 dark:border-zinc-800"
            >
              <p className="font-medium text-zinc-900 dark:text-zinc-100">
                {example.prompt}
              </p>
              <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
                {example.expectation}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-12 rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <h2 className="font-heading text-2xl font-bold">Privacy And Support</h2>
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Source uploads are temporary and may be deleted after processing.
          Indexed search data, transcript embeddings, and thumbnails remain
          available while the account is active.
        </p>
        <div className="mt-5 flex flex-wrap gap-4 text-sm">
          <Link href="/privacy" className="text-accent hover:underline">
            Privacy policy
          </Link>
          <Link href="/support" className="text-accent hover:underline">
            Support
          </Link>
          <Link href="/dashboard/api" className="text-accent hover:underline">
            API dashboard
          </Link>
        </div>
      </section>
    </div>
  );
}
