"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  SignInButton,
  SignedIn,
  SignedOut,
  useAuth,
} from "@clerk/nextjs";
import { API_URL, parseApiError } from "@/lib/api";

function modeButtonClass(isActive: boolean): string {
  const base = "flex-1 rounded-lg px-4 py-2 text-sm font-medium";
  return isActive
    ? `${base} bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900`
    : `${base} border border-zinc-300 text-zinc-700 dark:border-zinc-700 dark:text-zinc-200`;
}

export default function Home() {
  const router = useRouter();
  const { getToken, isLoaded } = useAuth();
  const [mode, setMode] = useState<"youtube" | "upload">("youtube");
  const [url, setUrl] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <p className="text-zinc-600 dark:text-zinc-400">Loading authentication...</p>
      </div>
    );
  }

  return (
    <div>
      {/* Hero */}
      <section className="flex flex-col items-center px-4 pt-20 pb-16 text-center animate-fade-in-up">
        <h1 className="font-heading text-5xl font-bold leading-tight sm:text-6xl">
          Find any moment
          <br />
          in any video
        </h1>
        <p className="mt-4 max-w-xl text-lg text-zinc-600 dark:text-zinc-400">
          Paste a YouTube URL or upload a video. Search for any scene using natural language.
          AI-powered semantic search finds the exact frame you need.
        </p>
        <div className="mt-8 flex gap-4">
          <a
            href="#tool"
            className="rounded-lg bg-accent px-6 py-3 text-sm font-medium text-white"
          >
            Try it free
          </a>
          <Link
            href="/pricing"
            className="rounded-lg border border-zinc-300 px-6 py-3 text-sm font-medium transition-colors hover:border-accent hover:text-accent dark:border-zinc-700"
          >
            See pricing
          </Link>
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-4xl px-4 pb-16">
        <h2 className="font-heading text-center text-2xl font-bold">
          How it works
        </h2>
        <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-3">
          {[
            {
              step: "1",
              title: "Paste or upload",
              description: "Submit a YouTube URL or upload a video file up to 30 minutes long.",
            },
            {
              step: "2",
              title: "AI processes",
              description:
                "Our AI extracts every frame and builds a semantic understanding of the visual content.",
            },
            {
              step: "3",
              title: "Search moments",
              description:
                'Type what you\'re looking for — "person standing at whiteboard" — and find it instantly.',
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

      {/* Tool section */}
      <section id="tool" className="flex flex-col items-center px-4 pb-16">
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

      {/* CTA banner */}
      <section className="border-t border-zinc-200 bg-surface-card px-4 py-16 text-center dark:border-zinc-800">
        <h2 className="font-heading text-2xl font-bold">
          Ready to find your moments?
        </h2>
        <p className="mt-2 text-zinc-600 dark:text-zinc-400">
          Start searching through your videos today.
        </p>
        <Link
          href="/pricing"
          className="mt-6 inline-block rounded-lg bg-accent px-6 py-3 text-sm font-medium text-white"
        >
          View pricing
        </Link>
      </section>
    </div>
  );
}
