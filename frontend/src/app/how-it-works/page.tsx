import type { Metadata } from "next";
import Link from "next/link";
import { howItWorksArticleSchema } from "@/lib/schema";

export const metadata: Metadata = {
  title: "How it works",
  description:
    "The Video Moment Finder pipeline: frame extraction, Qwen3-VL multimodal embeddings, vector similarity search, and where the approach works best (and where it doesn't).",
  alternates: { canonical: "/how-it-works" },
};

const DATE_PUBLISHED = "2026-04-21";
const DATE_MODIFIED = "2026-07-09";

const pipelineSteps = [
  {
    title: "1. Upload or import",
    body: "You upload a video file up to 90 minutes, or import a YouTube video you own. The upload is direct-to-storage via a presigned URL, so the file never round-trips through our API servers.",
  },
  {
    title: "2. Frame extraction",
    body: "A background worker samples frames from the video at a fixed cadence. We deliberately sample frames rather than run on raw video — it keeps the embedding cost bounded by video length and makes every hit in search map back to a timestamp you can jump to.",
  },
  {
    title: "3. Multimodal embeddings (Qwen3-VL)",
    body: "Each frame is passed through Qwen3-VL, an open-weight vision-language model. The output is a dense vector that encodes what is visually and semantically present in the frame — objects, actions, scene composition, on-screen text — in a space that text queries also live in.",
  },
  {
    title: "4. Vector store",
    body: "Frame vectors are written to a vector database with row-level security scoping them to your account. A single video becomes a few hundred to a few thousand vectors depending on length.",
  },
  {
    title: "5. Search",
    body: "At query time, your text prompt or example image is embedded with the same model into the same space. We run an approximate nearest-neighbor search, rank the top matches, and return each hit with its timestamp and thumbnail.",
  },
];

const tradeoffs = [
  {
    title: "Best at",
    body: 'Describing visually distinctive moments in plain language ("a whiteboard with a diagram of a neural network", "two people shaking hands outdoors"), or showing an example frame and asking for moments that look like it.',
  },
  {
    title: "Weaker at",
    body: "Precise dialogue recall or anything that hinges on spoken words — a transcript search will usually beat us there. Very subtle visual distinctions that look nearly identical across many frames also degrade precision.",
  },
  {
    title: "Practical tips",
    body: "Describe what would be on screen, not what is being said. If the first query misses, rephrase rather than going deeper into the results — a better query beats a longer scroll. For image-based search, pick an example frame that is visually clean and unambiguous.",
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
        From an uploaded video to a searchable index of visual moments, in five
        steps.
      </p>

      <section className="mt-10 space-y-5 text-zinc-700 dark:text-zinc-300">
        <p>
          Video Moment Finder treats video as a sequence of images, not a
          sequence of words. Instead of indexing transcripts and searching
          dialogue, we index the frames themselves with a multimodal
          vision-language model, then search that visual space directly. If you
          can describe what the moment would <em>look like</em> — or show an
          example frame — you can find it.
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
