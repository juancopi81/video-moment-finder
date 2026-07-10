import type { Metadata } from "next";
import { faqPageSchema } from "@/lib/schema";

export const metadata: Metadata = {
  title: "Support",
  description: "Get help with Video Moment Finder. FAQ and contact information.",
  alternates: { canonical: "/support" },
};

const FAQ_BASELINE_REVIEWED = "2026-07-09";

const faqs = [
  {
    question: "What video sources work best?",
    answer:
      "Direct upload is the reliable path. You can upload video files you own or are authorized to use, and videos must be 90 minutes or shorter and under 8 GB. YouTube URL import is also available for owned videos, but server-side restrictions can block it.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    id: "youtube-import",
    question: "What should I do if YouTube import fails?",
    answer:
      "Upload the video file directly instead. If this is your own YouTube video, use official download paths such as YouTube Studio or Google Takeout, then upload the file here. We do not recommend relying on third-party downloader sites.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    question: "How does the search work?",
    answer:
      "We use AI-powered semantic search. When you process a video, we extract frames and create embeddings using a multimodal AI model (Qwen3-VL). When you search, your text query or uploaded example image is embedded into the same space to find the most visually relevant moments.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    question: "How long does processing take?",
    answer:
      "Processing time depends on the video length: roughly 7-8 minutes per 30 minutes of video, so a 90-minute video typically takes around 22-25 minutes. You can leave the page and come back — processing continues in the background.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    question: "What happens to my video data?",
    answer:
      "We extract frames and generate AI embeddings for search. Thumbnail images and search vectors are stored to enable search. Uploaded source files are automatically cleaned up after processing, and submitted YouTube URLs are only used to attempt import and processing. We do not share your video data with third parties.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    question: "How do credits work?",
    answer:
      "Each credit allows you to process one video (up to 90 minutes). Free accounts include 1 credit to try the service. Paid credit packs are available on the pricing page for signed-in users. Credits do not expire.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    question: "How accurate are the search results?",
    answer:
      "Search results are AI-generated and may not always be perfectly accurate. The quality depends on video content, query phrasing, and how closely an example image matches the scene you want. Try different wording or a different image if the first search doesn't find what you need.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    question: "Can I get a refund?",
    answer:
      "Credits that have been used for processing are non-refundable. If you experience a technical issue that consumes a credit without delivering results, contact us and we will review your case.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
  {
    question: "Is my data secure?",
    answer:
      "Yes. All connections use HTTPS encryption. Your data is scoped to your authenticated account with row-level security. We use industry-standard infrastructure providers for storage and processing.",
    dateModified: FAQ_BASELINE_REVIEWED,
  },
];

export default function SupportPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <script type="application/ld+json">
        {JSON.stringify(faqPageSchema(faqs))}
      </script>
      <h1 className="font-heading text-4xl font-bold">Support</h1>
      <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
        Have a question? Check our FAQ below or reach out directly.
      </p>

      <div className="mt-8 rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <h2 className="font-heading text-lg font-bold">Contact Us</h2>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          Email us at{" "}
          <a
            href="mailto:support@videomomentfinder.com"
            className="text-accent hover:underline"
          >
            support@videomomentfinder.com
          </a>
        </p>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          We typically respond within 24 hours on business days.
        </p>
      </div>

      <div className="mt-12">
        <h2 className="font-heading text-2xl font-bold">
          Frequently Asked Questions
        </h2>
        <div className="mt-6 space-y-6">
          {faqs.map((faq) => (
            <div key={faq.question} id={faq.id}>
              <h3 className="font-semibold">{faq.question}</h3>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">{faq.answer}</p>
              <p className="mt-1 text-xs text-zinc-500">
                Updated <time dateTime={faq.dateModified}>{faq.dateModified}</time>
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
