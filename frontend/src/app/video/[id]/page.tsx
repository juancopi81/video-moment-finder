"use client";

import { useState, useEffect, use } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  SignInButton,
  SignedIn,
  SignedOut,
  UserButton,
  useAuth,
} from "@clerk/nextjs";

type VideoPageProps = {
  params: Promise<{ id: string }>;
};

type VideoStatus = "queued" | "processing" | "ready" | "failed";

type SearchResult = {
  timestamp_s: number;
  thumbnail_url: string | null;
  score: number;
};

type VideoStatusResponse = {
  id: string;
  youtube_url: string | null;
  status: VideoStatus;
  error_message: string | null;
};

type VideoSearchResponse = {
  video_id: string;
  youtube_url: string | null;
  status: VideoStatus;
  results: SearchResult[];
};

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 300; // 10 minutes at 2s interval

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function buildTimestampUrl(baseUrl: string, seconds: number): string | null {
  try {
    const url = new URL(baseUrl);
    url.searchParams.set("t", Math.floor(seconds).toString());
    return url.toString();
  } catch {
    return null;
  }
}

export default function VideoPage({ params }: VideoPageProps) {
  const { id } = use(params);
  const { getToken, isLoaded, userId } = useAuth();

  const [status, setStatus] = useState<VideoStatus>("queued");
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Poll for video status
  useEffect(() => {
    if (!isLoaded || !userId) return;
    if (status !== "queued" && status !== "processing") return;

    let attempts = 0;
    let stopped = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      if (stopped) return;
      if (attempts >= MAX_POLL_ATTEMPTS) {
        setError(
          "Processing is taking longer than expected. Please refresh in a bit."
        );
        stopped = true;
        if (interval) clearInterval(interval);
        return;
      }
      attempts += 1;

      try {
        const token = await getToken();
        if (!token) {
          setError("Please sign in to continue.");
          stopped = true;
          if (interval) clearInterval(interval);
          return;
        }

        const res = await fetch(`${apiUrl}/videos/${id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.status === 401) {
          setError("Session expired. Please sign in again.");
          stopped = true;
          if (interval) clearInterval(interval);
          return;
        }
        if (!res.ok) {
          throw new Error("Failed to fetch video status");
        }
        const data: VideoStatusResponse = await res.json();
        setStatus(data.status);
        setStatusMessage(data.error_message);
        if (data.youtube_url) {
          setVideoUrl(data.youtube_url);
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    interval = setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);
    void poll(); // Initial fetch

    return () => {
      stopped = true;
      if (interval) clearInterval(interval);
    };
  }, [apiUrl, getToken, id, isLoaded, status, userId]);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setError(null);
    setResults([]);

    try {
      const token = await getToken();
      if (!token) {
        throw new Error("Please sign in to search this video.");
      }

      const res = await fetch(`${apiUrl}/videos/${id}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query_text: searchQuery }),
      });

      if (!res.ok) {
        if (res.status === 401) {
          throw new Error("Session expired. Please sign in again.");
        }
        const data = await res.json().catch(() => null);
        const detail = typeof data?.detail === "string" ? data.detail : "Search failed";
        throw new Error(detail);
      }

      const data: VideoSearchResponse = await res.json();
      setResults(data.results);
      if (data.youtube_url) {
        setVideoUrl(data.youtube_url);
      }
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred");
      }
    } finally {
      setIsSearching(false);
    }
  }

  if (!isLoaded) {
    return (
      <main className="flex min-h-screen items-center justify-center p-8">
        <p className="text-zinc-600 dark:text-zinc-400">Loading authentication...</p>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="absolute top-4 right-4">
        <SignedIn>
          <UserButton afterSignOutUrl="/" />
        </SignedIn>
      </div>

      <h1 className="text-2xl font-bold mb-2">Video: {id}</h1>
      {error && (
        <p className="mb-4 text-sm text-red-600 dark:text-red-400 text-center">
          {error}
        </p>
      )}

      <SignedOut>
        <div className="w-full max-w-xl rounded-lg border border-zinc-300 dark:border-zinc-700 p-6 text-center">
          <p className="mb-4 text-zinc-600 dark:text-zinc-400">
            Sign in to check processing status and search this video.
          </p>
          <SignInButton mode="modal">
            <button className="w-full rounded-lg bg-zinc-900 px-4 py-3 font-medium text-white dark:bg-zinc-100 dark:text-zinc-900">
              Sign In
            </button>
          </SignInButton>
        </div>
      </SignedOut>

      <SignedIn>
        {(status === "queued" || status === "processing") && (
          <div className="text-center">
            <p className="text-zinc-600 dark:text-zinc-400 mb-4">
              {status === "queued" ? "Queued for processing..." : "Processing your video..."}
            </p>
            <div className="w-64 h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
              <div className="w-1/2 h-full bg-zinc-900 dark:bg-zinc-100 animate-pulse" />
            </div>
          </div>
        )}

        {status === "failed" && (
          <div className="text-center">
            <p className="text-red-600 dark:text-red-400 mb-4">
              {statusMessage ?? "Failed to process video"}
            </p>
            <Link
              href="/"
              className="text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
            >
              Try another video
            </Link>
          </div>
        )}

        {status === "ready" && (
          <div className="w-full max-w-xl">
            <form onSubmit={handleSearch}>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for a moment..."
                className="w-full px-4 py-3 border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-900"
                disabled={isSearching}
              />
              <button
                type="submit"
                className="mt-4 w-full px-4 py-3 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded-lg font-medium disabled:opacity-50"
                disabled={isSearching || !searchQuery.trim()}
              >
                {isSearching ? "Searching..." : "Search"}
              </button>
            </form>

            {results.length > 0 && (
              <div className="mt-8 grid grid-cols-3 gap-4">
                {results.map((result, index) => {
                  const timestampUrl = videoUrl
                    ? buildTimestampUrl(videoUrl, result.timestamp_s)
                    : null;

                  return (
                    <div key={index} className="relative">
                      <div className="aspect-video bg-zinc-200 dark:bg-zinc-800 rounded overflow-hidden">
                        {result.thumbnail_url ? (
                          <Image
                            src={result.thumbnail_url}
                            alt={`Result at ${formatTimestamp(result.timestamp_s)}`}
                            width={320}
                            height={180}
                            unoptimized
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-xs text-zinc-500 dark:text-zinc-400">
                            No thumbnail
                          </div>
                        )}
                      </div>
                      <p className="mt-1 text-sm text-center text-zinc-600 dark:text-zinc-400">
                        {formatTimestamp(result.timestamp_s)}
                      </p>
                      {timestampUrl && (
                        <a
                          href={timestampUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="mt-1 block text-xs text-center text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                        >
                          Open at timestamp
                        </a>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </SignedIn>

      <Link
        href="/"
        className="mt-8 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
      >
        &larr; Back to home
      </Link>
    </main>
  );
}
