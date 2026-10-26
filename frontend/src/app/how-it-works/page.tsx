import type { Metadata } from "next";
import Link from "next/link";
import { howItWorksArticleSchema } from "@/lib/schema";

export const metadata: Metadata = {
  title: "How it works",
  description:
    "How Video Moment Finder searches visual moments and spoken explanations in your own videos, with timestamped results and practical accuracy limits.",
  alternates: { canonical: "/how-it-works" },
};

const DATE_PUBLISHED = "2026-04-21";
const DATE_MODIFIED = "2026-10-26";

const pipelineSteps = [
  {
    title: "1. Upload or import",
    body: "You upload a video file up to 90 minutes, or import a YouTube video you own. The upload is direct-to-storage via a presigned URL, so the file never round-trips through our API servers.",
  },
  {
    title: "2. Frames and speech",
    body: "A background worker samples video frames and transcribes speech in direct uploads. YouTube imports use available subtitle or automatic caption tracks; they do not fall back to speech transcription when captions are missing. Frames and transcript segments keep their source timestamps.",
  },
  {
    title: "3. Multimodal embeddings (Qwen3-VL)",
    body: "Qwen3-VL represents sampled frames as vectors for visual search. Available transcript segments are also embedded for semantic search, so a text query can retrieve a spoken explanation as well as something shown on screen.",
  },
  {
    title: "4. Vector store",
    body: "Visual and transcript vectors are stored for retrieval. Before searching, the API checks that you can access the selected video, then restricts retrieval to that video.",
  },
  {
    title: "5. Search",
    body: "A text query searches visual moments and available transcripts, returning timestamped visual and spoken matches. Spoken matches include transcript text; visual matches include thumbnails. An example-image query searches visual frames only. Review the source moment to check whether it answers your question.",
  },
];

const tradeoffs = [
  {
    title: "Best at",
    body: 'Finding a teaching, demo, or explanation moment in a lesson, webinar, or workshop you own. Try "a whiteboard diagram of a neural network" for a visual moment or "where I explain the difference between training and inference" for a spoken explanation.',
  },
  {
    title: "Weaker at",
    body: "Speech retrieval depends on the available transcript: transcription errors, unclear audio, or missing YouTube captions can hide relevant passages. Visually similar frames and details between sampled frames can also be missed. Results are candidates, not guaranteed exact matches.",
  },
  {
    title: "Practical tips",
    body: "Describe what you showed or explained, then review both visual and spoken results. If the first query misses, try a more specific topic or rephrase it. For image search, choose a clear example of the scene you want; an image query does not search spoken words.",
  },
];

export default function HowItWorksPage() {
  const schema = howItWorksArticleSchema({
    datePublished: DATE_PUBLISHED,
    dateModified: DATE_MODIFIED,
  });

  return (
    <article className="mx-auto max-w-3xl px-4 py-16">
      <script type="application/ld+json">{JSON.stringify(schema)}</script>

      <h1 className="font-heading text-4xl font-bold">How it works</h1>
      <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
        Find the teaching, demo, or explanation moment in your own video using
        visual and transcript search.
      </p>

      <section className="mt-10 space-y-5 text-zinc-700 dark:text-zinc-300">
        <p>
          Video Moment Finder indexes sampled frames and available speech
          transcripts. Select a video from your private library and describe
          what you showed or said: text search can return both visual moments
          and spoken passages with timestamps. You can also upload an example
          image to find visually similar frames.
        </p>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">The pipeline</h2>
        <ol className="mt-6 space-y-6">
          {pipelineSteps.map((step) => (
            <li
              key={step.title}
              className="rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800"
            >
              <h3 className="font-heading text-lg font-bold">{step.title}</h3>
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                {step.body}
              </p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Why Qwen3-VL</h2>
        <div className="mt-4 space-y-4 text-zinc-700 dark:text-zinc-300">
          <p>
            Qwen3-VL is an open-weight vision-language model trained jointly on
            images and text, which means a photo of a whiteboard and the phrase
            &ldquo;a whiteboard with equations&rdquo; end up close together in
            the same embedding space. That joint space is what makes
            text-to-frame and image-to-frame search work with the same index.
          </p>
          <p>
            We picked it over alternatives for three reasons. First, it is
            open-weight, which matters for a project shipped under AGPLv3 and
            removes a hard dependency on a proprietary embedding API. Second,
            its training explicitly includes visual reasoning and on-screen
            text, both of which matter for real-world video (slides, UI
            recordings, signage). Third, the model is small enough to run
            cost-effectively at the scale of hundreds to thousands of frames
            per video without making the unit economics fall apart.
          </p>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="font-heading text-2xl font-bold">Accuracy tradeoffs</h2>
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {tradeoffs.map((t) => (
            <div
              key={t.title}
              className="rounded-xl border border-zinc-200 bg-surface-card p-5 dark:border-zinc-800"
            >
              <h3 className="font-heading text-base font-bold">{t.title}</h3>
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                {t.body}
              </p>
            </div>
          ))}
        </div>
        <p className="mt-6 text-sm text-zinc-600 dark:text-zinc-400">
          Results are AI-generated and won&apos;t always be right — see the{" "}
          <Link href="/support" className="text-accent hover:underline">
            support page
          </Link>{" "}
          for more on expected accuracy and refunds.
        </p>
      </section>

      <section className="mt-12 rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <h2 className="font-heading text-lg font-bold">Building on top</h2>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          The same pipeline is exposed as an HTTP API and as a remote MCP
          server, so agents and scripts can upload, check status, and search
          videos directly. See the{" "}
          <Link href="/developers" className="text-accent hover:underline">
            developers page
          </Link>{" "}
          for the MCP connector and OpenAPI reference.
        </p>
      </section>

      <p className="mt-12 text-xs text-zinc-500">
        Last updated {DATE_MODIFIED}.
      </p>
    </article>
  );
}
