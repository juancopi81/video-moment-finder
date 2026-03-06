"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  SignInButton,
  SignedIn,
  SignedOut,
  useAuth,
} from "@clerk/nextjs";
import { AuthLoadingFallback } from "@/components/auth-loading-fallback";
import { API_URL, parseApiError } from "@/lib/api";
import { trackEvent } from "@/lib/analytics";

function modeButtonClass(isActive: boolean): string {
  const base = "flex-1 rounded-lg px-4 py-2 text-sm font-medium";
  return isActive
    ? `${base} bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900`
    : `${base} border border-zinc-300 text-zinc-700 dark:border-zinc-700 dark:text-zinc-200`;
}

export default function Home() {
  const router = useRouter();
  const { getToken, isLoaded, isSignedIn } = useAuth();
  const [mode, setMode] = useState<"youtube" | "upload">("youtube");
  const [url, setUrl] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    trackEvent("landing_visit");
  }, []);

  useEffect(() => {
    if (!isSignedIn) return;
    getToken().then((token) => {
      trackEvent("signup_complete", undefined, token);
    });
  }, [isSignedIn, getToken]);

  function handleModeChange(nextMode: "youtube" | "upload") {
    setMode(nextMode);
    setError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!url.trim()) {
      setError("Please enter a YouTube URL");
      return;
    }

    setIsLoading(true);

    try {
      const token = await getToken();
      if (!token) {
        setError("Please sign in to process a video.");
        setIsLoading(false);
        return;
      }

      const response = await fetch(`${API_URL}/videos`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ youtube_url: url }),
      });

      if (!response.ok) {
        throw new Error(await parseApiError(response, "Failed to process video"));
      }

      const data = await response.json();
      router.push(`/video/${data.id}`);
    } catch (err) {
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setError("Cannot connect to server. Please try again later.");
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred");
      }
      setIsLoading(false);
    }
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!uploadFile) {
      setError("Please choose a video file to upload.");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const token = await getToken();
      if (!token) {
        setError("Please sign in to process a video.");
        setIsUploading(false);
        setUploadProgress(null);
        return;
      }

      const initResponse = await fetch(`${API_URL}/videos/upload/init`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          filename: uploadFile.name,
          ...(uploadFile.type ? { content_type: uploadFile.type } : {}),
        }),
      });

      if (!initResponse.ok) {
        throw new Error(await parseApiError(initResponse, "Failed to prepare upload"));
      }

      const initData = await initResponse.json();

      await new Promise<void>((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open("PUT", initData.upload_url);
        if (uploadFile.type) {
          request.setRequestHeader("Content-Type", uploadFile.type);
        }

        request.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            setUploadProgress(Math.round((event.loaded / event.total) * 100));
          }
        };

        request.onload = () => {
          if (request.status >= 200 && request.status < 300) {
            resolve();
          } else {
            reject(new Error("Failed to upload video to storage"));
          }
        };

        request.onerror = () => {
          reject(new TypeError("Network error"));
        };

        request.send(uploadFile);
      });

      const completeResponse = await fetch(`${API_URL}/videos/upload/complete`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          video_id: initData.video_id,
          filename: uploadFile.name,
        }),
      });

      if (!completeResponse.ok) {
        throw new Error(await parseApiError(completeResponse, "Failed to finalize upload"));
      }

      const data = await completeResponse.json();
      router.push(`/video/${data.id}`);
    } catch (err) {
      if (err instanceof TypeError && err.message.includes("Network")) {
        setError("Cannot connect to server. Please try again later.");
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred");
      }
      setIsUploading(false);
      setUploadProgress(null);
    }
  }

  if (!isLoaded) {
    return <AuthLoadingFallback />;
  }

  return (
    <div>
      {/* Hero */}
      <section className="flex flex-col items-center px-4 pt-20 pb-16 text-center animate-fade-in-up">
        <h1 className="font-heading text-5xl font-bold leading-tight sm:text-6xl">
          Find the exact moment
          <br />
          in any video
        </h1>
        <p className="mt-4 max-w-xl text-lg text-zinc-600 dark:text-zinc-400">
          Search by description or example image to jump to the exact
          timestamp — in any YouTube video or upload.
        </p>
        <span className="mt-3 inline-block rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
          1 free credit &middot; videos up to 30 min
        </span>
        <div className="mt-8 flex items-center gap-4">
          <SignedOut>
            <a
              href="#tool"
              className="rounded-lg bg-accent px-6 py-3 text-sm font-medium text-white"
            >
              Try 1 free video
            </a>
            <Link
              href="/pricing"
              className="text-sm font-medium text-zinc-600 transition-colors hover:text-accent dark:text-zinc-400"
            >
              See pricing
            </Link>
          </SignedOut>
          <SignedIn>
            <a
              href="#tool"
              className="rounded-lg bg-accent px-6 py-3 text-sm font-medium text-white"
            >
              Process a video
            </a>
          </SignedIn>
        </div>
      </section>

      {/* Tool section */}
      <section id="tool" className="flex flex-col items-center px-4 pb-16">
        <p className="mb-4 max-w-md text-center text-xs text-zinc-500 dark:text-zinc-500">
          A 30-minute video typically processes in about 7–8 minutes. You can
          leave and come back while it finishes.
        </p>
        <SignedOut>
          <div className="w-full max-w-xl rounded-lg border border-zinc-300 dark:border-zinc-700 p-6 text-center">
            <p className="mb-4 text-zinc-600 dark:text-zinc-400">
              Sign in to start processing your videos.
            </p>
            <SignInButton mode="modal">
              <button className="w-full rounded-lg bg-accent px-4 py-3 font-medium text-white">
                Sign In
              </button>
            </SignInButton>
          </div>
        </SignedOut>

        <SignedIn>
          <div className="w-full max-w-xl">
            <div className="mb-4 flex gap-2">
              <button
                type="button"
                onClick={() => handleModeChange("youtube")}
                className={modeButtonClass(mode === "youtube")}
                disabled={isLoading || isUploading}
              >
                YouTube URL
              </button>
              <button
                type="button"
                onClick={() => handleModeChange("upload")}
                className={modeButtonClass(mode === "upload")}
                disabled={isLoading || isUploading}
              >
                Upload Video
              </button>
            </div>

            {mode === "youtube" ? (
              <form key="youtube-form" onSubmit={handleSubmit}>
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="w-full px-4 py-3 border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-900"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  className="mt-4 w-full px-4 py-3 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded-lg font-medium disabled:opacity-50"
                  disabled={isLoading}
                >
                  {isLoading ? "Processing..." : "Process Video"}
                </button>
              </form>
            ) : (
              <form key="upload-form" onSubmit={handleUpload}>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(event) => {
                    setUploadFile(event.target.files?.[0] ?? null);
                  }}
                  className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
                  disabled={isUploading}
                />
                <button
                  type="submit"
                  className="mt-4 w-full px-4 py-3 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded-lg font-medium disabled:opacity-50"
                  disabled={isUploading}
                >
                  {isUploading ? "Uploading..." : "Upload Video"}
                </button>
                {uploadProgress !== null && (
                  <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400 text-center">
                    Uploading: {uploadProgress}%
                  </p>
                )}
              </form>
            )}
            {error && (
              <p className="mt-2 text-sm text-red-600 dark:text-red-400 text-center">
                {error}
              </p>
            )}
          </div>
        </SignedIn>
      </section>

      {/* More than transcript search */}
      <section className="mx-auto max-w-3xl px-4 pb-16 text-center">
        <h2 className="font-heading text-2xl font-bold">
          More than transcript search
        </h2>
        <p className="mt-3 text-zinc-600 dark:text-zinc-400">
          Find moments that are hard to locate with chapters or transcript
          search:
        </p>
        <ul className="mt-4 inline-block space-y-2.5 text-left text-sm text-zinc-600 dark:text-zinc-400">
          {[
            "Visual scenes",
            "Slides and dashboards",
            "Product shots",
            "Reactions and gestures",
            "Moments similar to an example image",
          ].map((item) => (
            <li key={item} className="flex items-center gap-2.5">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent/10 dark:bg-accent/20">
                <svg className="h-3 w-3 text-accent" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2.5 6l2.5 2.5 4.5-4.5" />
                </svg>
              </span>
              {item}
            </li>
          ))}
        </ul>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-4xl px-4 pb-16">
        <h2 className="font-heading text-center text-2xl font-bold">
          How it works
        </h2>
        <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              step: "1",
              title: "Add a video",
              description: "Paste a YouTube URL or upload a file.",
            },
            {
              step: "2",
              title: "We index the visuals",
              description:
                "We extract frames and make the video searchable.",
            },
            {
              step: "3",
              title: "Search by text or image",
              description:
                "Describe the scene you want, or upload one example image.",
            },
            {
              step: "4",
              title: "Jump to the moment",
              description:
                "Open timestamped matches with thumbnails.",
            },
          ].map((item, i) => (
            <div
              key={item.step}
              className="rounded-xl border border-zinc-200 bg-surface-card p-6 dark:border-zinc-800 animate-fade-in-up"
              style={{ animationDelay: `${i * 150}ms` }}
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-sm font-bold text-white">
                {item.step}
              </div>
              <h3 className="mt-3 font-heading font-semibold">
                {item.title}
              </h3>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA banner */}
      <SignedOut>
        <section className="border-t border-zinc-200 bg-surface-card px-4 py-16 text-center dark:border-zinc-800">
          <h2 className="font-heading text-2xl font-bold">
            Stop scrubbing. Start searching.
          </h2>
          <p className="mt-2 text-zinc-600 dark:text-zinc-400">
            Try Video Moment Finder on one real video.
          </p>
          <a
            href="#tool"
            className="mt-6 inline-block rounded-lg bg-accent px-6 py-3 text-sm font-medium text-white"
          >
            Try 1 free video
          </a>
        </section>
      </SignedOut>
    </div>
  );
}
