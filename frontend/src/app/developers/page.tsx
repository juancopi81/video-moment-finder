import Link from "next/link";

export default function DevelopersPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="font-heading text-4xl font-bold">
        Build with Video Moment Finder
      </h1>
      <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
        Index videos and search by text through a REST API and CLI.
      </p>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">What the API does</h2>
        <ul className="mt-4 space-y-2 text-sm text-zinc-700 dark:text-zinc-300">
          <li>
            <strong>Index videos</strong> &mdash; upload a video file and the
            platform extracts frames, generates embeddings, and transcribes
            spoken content.
          </li>
          <li>
            <strong>Search by text</strong> &mdash; query indexed videos with
            natural language and get timestamped results ranked by relevance.
          </li>
        </ul>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Quick start</h2>
        <div className="mt-4 space-y-3 rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-sm font-medium">1. Install the CLI</p>
          <pre className="overflow-x-auto rounded bg-zinc-100 p-2 text-xs dark:bg-zinc-800">
            <code>uv sync</code>
          </pre>
          <p className="text-sm font-medium">2. Save your API key</p>
          <pre className="overflow-x-auto rounded bg-zinc-100 p-2 text-xs dark:bg-zinc-800">
            <code>uv run vmf auth set --api-base-url https://api.videomomentfinder.com --api-key vmf_YOUR_KEY</code>
          </pre>
          <p className="text-sm font-medium">3. Upload and search</p>
          <pre className="overflow-x-auto rounded bg-zinc-100 p-2 text-xs dark:bg-zinc-800">
            <code>{`uv run vmf videos upload ./sample.mp4
uv run vmf videos wait <video_id>
uv run vmf videos search <video_id> --query-text "when do they explain the model?"`}</code>
          </pre>
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
            or via <code className="text-xs">uv run vmf keys create</code>.
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
        <Link href="/dashboard/api" className="text-accent hover:underline">
          Go to API dashboard
        </Link>
      </div>
    </div>
  );
}
