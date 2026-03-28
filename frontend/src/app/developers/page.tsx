import Link from "next/link";

export default function DevelopersPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="font-heading text-4xl font-bold animate-fade-in-up">
        Build with Video Moment Finder
      </h1>
      <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
        Agent-ready REST API and CLI for upload, wait, and text search.
      </p>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Agent bootstrap</h2>
        <div className="mt-4 rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
          <p className="text-sm text-zinc-700 dark:text-zinc-300">
            For agents and coding assistants, start with the public skill file.
            It defines the canonical URLs, security rules, and the supported
            upload -&gt; wait -&gt; search flow. This remains the primary public
            contract today.
          </p>
          <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
            A thin remote MCP endpoint is also available at{" "}
            <code className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs dark:bg-zinc-800">
              https://api.videomomentfinder.com/mcp
            </code>{" "}
            for manual <code className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs dark:bg-zinc-800">vmf_</code> API-key
            testing. The clean OAuth install/connect flow is not shipped yet.
          </p>
          <div className="mt-4 flex flex-wrap gap-3 text-sm">
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
            <a
              href="https://api.videomomentfinder.com/docs"
              className="rounded-lg border border-zinc-300 px-4 py-2 font-medium text-zinc-900 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-900"
            >
              Swagger UI
            </a>
          </div>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Public contract</h2>
        <div className="mt-4 rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
          <p className="text-sm text-zinc-700 dark:text-zinc-300">
            The published OpenAPI schema covers API keys, developer billing,
            one-shot upload, direct upload init and complete, status polling,
            video listing, and text search.
          </p>
          <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
            Web-only routes like YouTube submit, creator credit checkout, and
            image search still exist for the signed-in product, but they are
            intentionally excluded from the public schema.
          </p>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">What the API does</h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl border border-zinc-200 bg-surface-card p-5 dark:border-zinc-800">
            <p className="font-medium text-zinc-900 dark:text-zinc-100">
              Index videos
            </p>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              Upload a video file and the platform extracts frames, generates
              embeddings, and transcribes spoken content.
            </p>
          </div>
          <div className="rounded-xl border border-zinc-200 bg-surface-card p-5 dark:border-zinc-800">
            <p className="font-medium text-zinc-900 dark:text-zinc-100">
              Search by text
            </p>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              Query indexed videos with natural language and get timestamped
              results ranked by relevance.
            </p>
          </div>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Quick start</h2>
        <p className="mt-4 text-sm text-zinc-700 dark:text-zinc-300">
          The CLI path below assumes you already cloned this repository. For raw
          HTTP usage or autonomous agents, start with <Link href="/skill.md" className="text-accent hover:underline">/skill.md</Link>.
        </p>
        <div className="mt-4 space-y-4 rounded-xl border border-zinc-200 bg-zinc-50 p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <div>
            <p className="text-sm font-medium">1. If you cloned the repo, install the CLI</p>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-100 p-3 text-xs dark:bg-zinc-800">
              <code>uv sync</code>
            </pre>
          </div>
          <div>
            <p className="text-sm font-medium">2. Save your API key</p>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-100 p-3 text-xs dark:bg-zinc-800">
              <code>uv run vmf auth set --api-base-url https://api.videomomentfinder.com --api-key vmf_YOUR_KEY</code>
            </pre>
          </div>
          <div>
            <p className="text-sm font-medium">3. Upload and search</p>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-100 p-3 text-xs dark:bg-zinc-800">
              <code>{`uv run vmf videos upload ./sample.mp4
uv run vmf videos wait <video_id>
uv run vmf videos search <video_id> --query-text "when do they explain the model?"`}</code>
            </pre>
          </div>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Authentication</h2>
        <ol className="mt-4 list-inside list-decimal space-y-2 text-sm text-zinc-700 dark:text-zinc-300">
          <li>
            <Link href="/pricing" className="text-accent hover:underline">
              Sign in
            </Link>{" "}
            to Video Moment Finder.
          </li>
          <li>
            Purchase a{" "}
            <Link
              href="/dashboard/api"
              className="text-accent hover:underline"
            >
              Developer Pack
            </Link>{" "}
            from the API dashboard.
          </li>
          <li>
            Create an API key from{" "}
            <Link
              href="/dashboard/api"
              className="text-accent hover:underline"
            >
              /dashboard/api
            </Link>{" "}
            or via <code className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs dark:bg-zinc-800">uv run vmf keys create</code>.
          </li>
        </ol>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Pricing</h2>
        <div className="mt-4 rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
          <p className="text-2xl font-bold">Developer Pack &mdash; $20</p>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            10,000 API units
          </p>
          <ul className="mt-4 space-y-1 text-sm text-zinc-700 dark:text-zinc-300">
            <li>500 units per indexed video</li>
            <li>1 unit per text query (launch pricing)</li>
          </ul>
          <Link
            href="/dashboard/api"
            className="mt-4 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white"
          >
            Get started
          </Link>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Data retention</h2>
        <p className="mt-4 text-sm text-zinc-700 dark:text-zinc-300">
          Source uploads are temporary and may be auto-deleted after processing.
          Indexed search data, transcript embeddings, and thumbnails remain
          available while your account is active.
        </p>
      </section>

      <div className="mt-12 text-sm text-zinc-600 dark:text-zinc-400">
        <div className="flex flex-wrap gap-x-4 gap-y-2">
          <Link href="/dashboard/api" className="text-accent hover:underline">
            Go to API dashboard
          </Link>
          <Link href="/skill.md" className="text-accent hover:underline">
            Read /skill.md
          </Link>
        </div>
      </div>
    </div>
  );
}
