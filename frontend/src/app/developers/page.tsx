import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Connect Video Moment Finder",
  description: "Connect your video library with MCP, understand API units, and get your first learning result.",
  alternates: { canonical: "/developers" },
};

const connectorSteps = [
  "Add Video Moment Finder in your app, or add a remote MCP server with URL https://api.videomomentfinder.com/mcp.",
  "Start Connect and sign in to the Video Moment Finder account that holds your videos. Remote MCP uses OAuth; you do not need to paste an API key into chat.",
  "Review the requesting app, available tools, operation costs, and your account allowance on the approval screen. Any available one-time trial is reported there after account verification; reconnecting does not grant another trial.",
  "Approve access, then return to your app. If your account has no API units, the approval screen explains the limit without starting any video operation.",
];

const tools = [
  { name: "upload_video", title: "Upload video", summary: "Starts or completes a file upload for indexing. Use only a file you own or are authorized to use.", cost: "500 units per indexed video", access: "Write" },
  { name: "get_video_status", title: "Check video status", summary: "Reports whether a video is queued, processing, ready, or failed.", cost: "No units", access: "Read" },
  { name: "list_videos", title: "List your videos", summary: "Lists videos available to the connected account.", cost: "No units", access: "Read" },
  { name: "search_video", title: "Find a moment", summary: "Finds timestamped matches in a ready video using a text query.", cost: "1 unit per query", access: "Read" },
  { name: "get_transcript", title: "Read a transcript", summary: "Retrieves spoken content with timestamps, optionally for a selected time range.", cost: "1 unit per call", access: "Read" },
  { name: "get_frames", title: "Inspect frames", summary: "Retrieves images at selected timestamps. High-resolution retrieval may fall back to stored thumbnails when the source is no longer retained.", cost: "1 unit per thumbnail call (up to 25 timestamps); 5 per high-resolution call (up to 8)", access: "Read" },
];

const promptExamples = [
  { prompt: "List my ready videos, then help me understand one key idea from a lecture, with a timestamp and a frame where available.", expectation: "Choose an available video, retrieve a small amount of evidence, and explain the idea with its source timestamp." },
  { prompt: "Explain this diagram at 03:43 in video <video_id>. What does each part mean?", expectation: "Inspect the actual frame and nearby transcript, distinguish visible labels from interpretation, and identify anything unreadable." },
  { prompt: "Turn this indexed lecture into study notes with the main takeaways and source timestamps.", expectation: "Use transcript evidence and selected frames. In clients that support MCP prompts, the user can also invoke lecture_notes." },
  { prompt: "Help me study this idea using the evidence we already retrieved.", expectation: "Continue the explanation or a practice question using existing evidence when more video retrieval is unavailable." },
];

export default function DevelopersPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="rounded-3xl border border-zinc-200 bg-surface-card p-8 shadow-sm dark:border-zinc-800">
        <p className="text-sm font-medium uppercase tracking-[0.2em] text-accent">Video learning and developer access</p>
        <h1 className="mt-3 font-heading text-4xl font-bold">Connect Video Moment Finder</h1>
        <p className="mt-4 max-w-3xl text-lg text-zinc-600 dark:text-zinc-400">
          Bring your indexed videos into a supported app to understand a lecture,
          inspect a diagram, or find a specific explanation. The connection uses
          your Video Moment Finder account and its available API units.
        </p>
        <div className="mt-6 flex flex-wrap gap-4 text-sm">
          <a href="#connect" className="rounded-lg bg-accent px-4 py-2 font-medium text-white">Connection steps</a>
          <Link href="/skill.md" className="rounded-lg border border-zinc-300 px-4 py-2 dark:border-zinc-700">Public integration reference</Link>
        </div>
      </div>

      <section id="connect" className="mt-12 rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <h2 className="font-heading text-2xl font-bold">Connect your account</h2>
        <ol className="mt-4 list-decimal space-y-3 pl-5 text-sm text-zinc-700 dark:text-zinc-300">
          {connectorSteps.map((step) => <li key={step}>{step}</li>)}
        </ol>
        <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-400">
          This is a client-neutral remote MCP server with OAuth authorization-code
          and PKCE. Apps that support dynamic client registration can register
          automatically. Keep tokens and client secrets out of chat and public documents.
        </p>
      </section>

      <section className="mt-12 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <h2 className="font-heading text-2xl font-bold">Tools and operation costs</h2>
          <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
            Default rates are shown below. The approval screen reports the configured
            rates for your connection. Frame retrieval is charged per call, not per frame.
          </p>
          <div className="mt-5 space-y-3">
            {tools.map((tool) => (
              <div key={tool.name} className="rounded-2xl border border-zinc-200 bg-surface-card p-4 dark:border-zinc-800">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="font-medium">{tool.title}</h3>
                  <span className="rounded-full bg-zinc-100 px-3 py-1 text-xs dark:bg-zinc-800">{tool.access}</span>
                </div>
                <p className="mt-1 text-xs text-zinc-500"><code>{tool.name}</code></p>
                <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">{tool.summary}</p>
                <p className="mt-2 text-sm font-medium">{tool.cost}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="space-y-6">
          <div className="rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
            <h2 className="font-heading text-xl font-bold">Start with one useful result</h2>
            <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
              List your ready videos first, then choose one idea to explain. At the
              default rates, one transcript retrieval plus one thumbnail call uses
              2 API units; one transcript plus one high-resolution frame call uses
              6. Additional searches or frame calls add to the total.
            </p>
            <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
              Indexing a new video costs 500 API units at the default rate and is
              separate from learning with an already indexed video. Review the
              expected cost before an upload begins.
            </p>
          </div>
          <div className="rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
            <h2 className="font-heading text-xl font-bold">Account allowance</h2>
            <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
              Website video credits and API units are separate balances. Connected
              apps use API units. If your account receives a one-time verified-account
              trial, its actual grant and shared unit balance are shown after sign-in.
              Prior free website processing can count toward that allowance.
            </p>
            <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
              A trial is not available on every deployment. Signing out, reconnecting,
              or creating a new API key does not reset an account allowance. When no
              units remain, metered operations stop; you can keep studying evidence
              already retrieved in your chat.
            </p>
          </div>
        </div>
      </section>

      <section className="mt-12 rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <h2 className="font-heading text-2xl font-bold">If a video or frame is unavailable</h2>
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Only videos accessible to the connected account can be used. Wait for a
          queued or processing video to become ready, or choose another video if
          processing failed. A public link does not by itself make a video available
          through MCP. File upload requires an app that can transfer the bytes to a
          temporary upload URL; never send your account token to that URL.
        </p>
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Original uploaded video files are scheduled for deletion 30 days after
          upload; cleanup may complete later. Playback and high-resolution frames
          need the original, while stored transcripts and thumbnails remain available
          after source cleanup. Media links are temporary and may need refreshing
          independently of source retention. A learning answer should name evidence
          gaps, keep source timestamps, and avoid guessing details from an unreadable frame.
        </p>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Try a learning prompt</h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {promptExamples.map((example) => (
            <div key={example.prompt} className="rounded-2xl border border-zinc-200 bg-surface-card p-5 dark:border-zinc-800">
              <p className="font-medium">{example.prompt}</p>
              <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">{example.expectation}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-12 rounded-2xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <h2 className="font-heading text-2xl font-bold">REST API and CLI</h2>
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Direct programmatic requests use <code>vmf_</code> API keys instead of
          MCP OAuth tokens. The public REST API supports upload, status, listing,
          search, transcripts, and frames. The CLI wraps upload, status, and search.
        </p>
        <div className="mt-5 flex flex-wrap gap-4 text-sm">
          <a href="https://api.videomomentfinder.com/docs" className="text-accent hover:underline">REST API reference</a>
          <a href="https://api.videomomentfinder.com/openapi.json" className="text-accent hover:underline">OpenAPI schema</a>
          <Link href="/privacy" className="text-accent hover:underline">Privacy policy</Link>
          <Link href="/support" className="text-accent hover:underline">Support</Link>
        </div>
      </section>
    </div>
  );
}
