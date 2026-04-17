import type { Metadata } from "next";
import Image from "next/image";
import { personSchema } from "@/lib/schema";

export const metadata: Metadata = {
  title: "About",
  description:
    "About the creator of Video Moment Finder — background, motivation, and how to get in touch.",
  alternates: { canonical: "/about" },
};

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <script type="application/ld+json">
        {JSON.stringify(personSchema)}
      </script>

      <h1 className="font-heading text-4xl font-bold">About</h1>
      <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
        Who builds Video Moment Finder, and why.
      </p>

      <div className="mt-10 flex items-center gap-5 rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <Image
          src="https://github.com/juancopi81.png"
          alt="Juan Piñeros"
          width={96}
          height={96}
          className="rounded-full"
          unoptimized
        />
        <div>
          <p className="font-heading text-xl font-bold">Juan Piñeros</p>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Creator &amp; maintainer, Video Moment Finder
          </p>
        </div>
      </div>

      <div className="mt-10 space-y-5 text-zinc-700 dark:text-zinc-300">
        <p>
          I&apos;m an engineer working in applied machine learning, with a focus on
          multimodal models and search. Video Moment Finder is a personal
          project I build and maintain in the open — the whole codebase is
          public on GitHub under AGPLv3.
        </p>
        <p>
          I started VMF after spending too many evenings scrubbing through long
          recordings — lectures, tutorials, gameplay captures — looking for one
          specific moment I half-remembered. Existing tools searched{" "}
          <em>transcripts</em>, not what was actually{" "}
          <em>on screen</em>. With multimodal embedding models like Qwen3-VL,
          that problem is finally tractable: you can describe a scene in plain
          language, or show an example image, and get the frames that look most
          like it.
        </p>
        <p>
          VMF is intentionally small and single-purpose. If you have a bug
          report, a feature idea, or you just want to talk about multimodal
          retrieval, the best places to reach me are below.
        </p>
      </div>

      <div className="mt-10 rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800">
        <h2 className="font-heading text-lg font-bold">Elsewhere</h2>
        <ul className="mt-3 space-y-1 text-sm">
          <li>
            <a
              href="https://github.com/juancopi81"
              target="_blank"
              rel="noreferrer"
              className="text-accent hover:underline"
            >
              GitHub — juancopi81
            </a>
          </li>
          <li>
            <a
              href="https://www.linkedin.com/in/juancarlospinerosp/"
              target="_blank"
              rel="noreferrer"
              className="text-accent hover:underline"
            >
              LinkedIn
            </a>
          </li>
          <li>
            <a
              href="https://x.com/juancopi81"
              target="_blank"
              rel="noreferrer"
              className="text-accent hover:underline"
            >
              X / Twitter
            </a>
          </li>
          <li>
            <a
              href="mailto:support@videomomentfinder.com"
              className="text-accent hover:underline"
            >
              support@videomomentfinder.com
            </a>
          </li>
        </ul>
      </div>
    </div>
  );
}
