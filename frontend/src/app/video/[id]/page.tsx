"use client";

import { useState, useEffect, use, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  SignInButton,
  SignedIn,
  SignedOut,
  useAuth,
} from "@clerk/nextjs";
import { AuthLoadingFallback } from "@/components/auth-loading-fallback";
import { API_URL, parseApiError } from "@/lib/api";

type VideoPageProps = {
  params: Promise<{ id: string }>;
};

type VideoStatus = "queued" | "processing" | "ready" | "failed";
type SearchMode = "text" | "image";

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
  source_type: "youtube" | "upload";
  source_url: string | null;
};

type VideoSearchResponse = {
  video_id: string;
  youtube_url: string | null;
  source_url: string | null;
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
  const [sourceType, setSourceType] = useState<"youtube" | "upload" | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [searchMode, setSearchMode] = useState<SearchMode>("text");
  const [searchQuery, setSearchQuery] = useState("");
  const [queryImageFile, setQueryImageFile] = useState<File | null>(null);
  const [queryImagePreviewUrl, setQueryImagePreviewUrl] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const queryImageInputRef = useRef<HTMLInputElement | null>(null);

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

        const res = await fetch(`${API_URL}/videos/${id}`, {
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
        if (data.source_type) {
          setSourceType(data.source_type);
        }
        setSourceUrl(data.source_url ?? null);
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
  }, [getToken, id, isLoaded, status, userId]);

  useEffect(() => {
    if (!queryImageFile) {
      setQueryImagePreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(queryImageFile);
    setQueryImagePreviewUrl(objectUrl);
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [queryImageFile]);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (searchMode === "text" && !searchQuery.trim()) return;
    if (searchMode === "image" && !queryImageFile) return;

    setIsSearching(true);
    setError(null);
    setResults([]);

    try {
      const token = await getToken();
      if (!token) {
        throw new Error("Please sign in to search this video.");
      }

      let res: Response;
      if (searchMode === "text") {
        res = await fetch(`${API_URL}/videos/${id}/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ query_text: searchQuery }),
        });
      } else {
        if (!queryImageFile) return;
        const formData = new FormData();
        formData.append("query_image", queryImageFile);
        res = await fetch(`${API_URL}/videos/${id}/search/image`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        });
      }

      if (!res.ok) {
        throw new Error(await parseApiError(res, "Search failed"));
      }

      const data: VideoSearchResponse = await res.json();
      setResults(data.results);
      if (data.youtube_url) {
        setVideoUrl(data.youtube_url);
      }
      setSourceUrl(data.source_url ?? null);
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
    return <AuthLoadingFallback />;
  }

  return (
    <div className="flex flex-1 flex-col items-center justify-center p-8">
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
            {sourceType === "upload" && sourceUrl && (
              <div className="mb-6">
                <video
                  ref={videoRef}
                  src={sourceUrl}
                  controls
                  className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700"
                />
                <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                  Uploaded video playback
                </p>
              </div>
            )}

            {sourceType === "upload" && !sourceUrl && (
              <p className="mb-6 text-sm text-amber-600 dark:text-amber-400">
                The original video file has expired and is no longer available for playback. Your search results and thumbnails are not affected.
              </p>
            )}

            <form onSubmit={handleSearch}>
              <div className="grid grid-cols-2 gap-2 rounded-lg border border-zinc-300 bg-zinc-100 p-1 dark:border-zinc-700 dark:bg-zinc-900">
                <button
                  type="button"
                  onClick={() => setSearchMode("text")}
                  className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    searchMode === "text"
                      ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-800 dark:text-zinc-100"
                      : "text-zinc-600 dark:text-zinc-400"
                  }`}
                  disabled={isSearching}
                >
                  Text
                </button>
                <button
                  type="button"
                  onClick={() => setSearchMode("image")}
                  className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    searchMode === "image"
                      ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-800 dark:text-zinc-100"
                      : "text-zinc-600 dark:text-zinc-400"
                  }`}
                  disabled={isSearching}
                >
                  Image
                </button>
              </div>

              {searchMode === "text" ? (
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search for a moment..."
                  className="mt-4 w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 dark:border-zinc-700 dark:bg-zinc-900"
                  disabled={isSearching}
                />
              ) : (
                <div className="mt-4 rounded-lg border border-dashed border-zinc-300 p-4 dark:border-zinc-700">
                  <input
                    ref={queryImageInputRef}
                    type="file"
                    accept="image/*"
                    onChange={(e) => {
                      setQueryImageFile(e.target.files?.[0] ?? null);
                    }}
                    className="w-full text-sm text-zinc-600 file:mr-4 file:rounded-md file:border-0 file:bg-zinc-900 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white dark:text-zinc-400 dark:file:bg-zinc-100 dark:file:text-zinc-900"
                    disabled={isSearching}
                  />
                  <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                    Upload one example image to find visually similar moments in this video.
                  </p>
                  {queryImageFile && (
                    <div className="mt-4 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                      {queryImagePreviewUrl && (
                        <div className="relative aspect-video overflow-hidden rounded-md bg-zinc-200 dark:bg-zinc-800">
                          <Image
                            src={queryImagePreviewUrl}
                            alt={queryImageFile.name}
                            fill
                            unoptimized
                            className="object-contain"
                          />
                        </div>
                      )}
                      <div className="mt-3 flex items-center justify-between gap-3">
                        <p className="truncate text-sm text-zinc-600 dark:text-zinc-400">
                          {queryImageFile.name}
                        </p>
                        <button
                          type="button"
                          onClick={() => {
                            setQueryImageFile(null);
                            if (queryImageInputRef.current) {
                              queryImageInputRef.current.value = "";
                            }
                          }}
                          className="text-xs font-medium text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                          disabled={isSearching}
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
              <button
                type="submit"
                className="mt-4 w-full px-4 py-3 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded-lg font-medium disabled:opacity-50"
                disabled={
                  isSearching ||
                  (searchMode === "text" ? !searchQuery.trim() : !queryImageFile)
                }
              >
                {isSearching ? "Searching..." : "Search"}
              </button>
            </form>

            {results.length > 0 && (
              <div>
                <p className="mt-6 text-xs text-center text-zinc-500 dark:text-zinc-400">
                  Results are AI-generated and may not always be accurate.
                </p>
                <div className="mt-2 grid grid-cols-3 gap-4">
                {results.map((result, index) => {
                  const timestampUrl = videoUrl
                    ? buildTimestampUrl(videoUrl, result.timestamp_s)
                    : null;
                  const canJumpToUpload = sourceType === "upload" && sourceUrl;

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
                      {canJumpToUpload && (
                        <button
                          type="button"
                          onClick={() => {
                            if (!videoRef.current) return;
                            videoRef.current.currentTime = result.timestamp_s;
                            void videoRef.current.play().catch(() => {});
                            videoRef.current.scrollIntoView({
                              behavior: "smooth",
                              block: "center",
                            });
                          }}
                          className="mt-1 block w-full text-xs text-center text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                        >
                          Jump to timestamp
                        </button>
                      )}
                    </div>
                  );
                })}
                </div>
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
    </div>
  );
}
