"use client";

import { useState, useEffect, use } from "react";
import Link from "next/link";

type VideoPageProps = {
  params: Promise<{ id: string }>;
};

type VideoStatus = "processing" | "ready" | "failed";

type SearchResult = {
  timestamp_s: number;
  thumbnail_url: string;
  score: number;
};

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export default function VideoPage({ params }: VideoPageProps) {
  const { id } = use(params);

  const [status, setStatus] = useState<VideoStatus>("processing");
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Poll for video status
  useEffect(() => {
    if (status !== "processing") return;

    const poll = async () => {
      try {
        const res = await fetch(`${apiUrl}/videos/${id}`);
        if (!res.ok) {
          throw new Error("Failed to fetch video status");
        }
        const data = await res.json();
        setStatus(data.status);
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    poll(); // Initial fetch
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [id, status, apiUrl]);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setError(null);
    setResults([]);

    try {
      const res = await fetch(`${apiUrl}/videos/${id}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query_text: searchQuery }),
      });

      if (!res.ok) {
        throw new Error("Search failed");
      }

      const data = await res.json();
      setResults(data.results);
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

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <h1 className="text-2xl font-bold mb-2">Video: {id}</h1>

      {status === "processing" && (
        <div className="text-center">
          <p className="text-zinc-600 dark:text-zinc-400 mb-4">
            Processing your video...
          </p>
          <div className="w-64 h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
            <div className="w-1/2 h-full bg-zinc-900 dark:bg-zinc-100 animate-pulse" />
          </div>
        </div>
      )}

      {status === "failed" && (
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 mb-4">
            Failed to process video
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

          {error && (
            <p className="mt-2 text-sm text-red-600 dark:text-red-400 text-center">
              {error}
            </p>
          )}

          {results.length > 0 && (
            <div className="mt-8 grid grid-cols-3 gap-4">
              {results.map((result, index) => (
                <div key={index} className="relative">
                  <div className="aspect-video bg-zinc-200 dark:bg-zinc-800 rounded overflow-hidden">
                    <img
                      src={result.thumbnail_url}
                      alt={`Result at ${formatTimestamp(result.timestamp_s)}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <p className="mt-1 text-sm text-center text-zinc-600 dark:text-zinc-400">
                    {formatTimestamp(result.timestamp_s)}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <Link
        href="/"
        className="mt-8 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
      >
        &larr; Back to home
      </Link>
    </main>
  );
}
