"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  SignInButton,
  SignedIn,
  SignedOut,
  useAuth,
} from "@clerk/nextjs";
import { API_URL, parseApiError } from "@/lib/api";

type VideoStatus = "queued" | "processing" | "ready" | "failed";

type VideoListItem = {
  id: string;
  youtube_url: string | null;
  status: VideoStatus;
  source_type: "youtube" | "upload";
  source_filename: string | null;
  created_at: string;
  error_message: string | null;
};

function formatCreatedAt(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function statusBadgeClass(status: VideoStatus): string {
  switch (status) {
    case "ready":
      return "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-200";
    case "failed":
      return "bg-red-100 text-red-700 dark:bg-red-500/15 dark:text-red-200";
    case "processing":
      return "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-200";
    default:
      return "bg-zinc-100 text-zinc-700 dark:bg-zinc-500/15 dark:text-zinc-200";
  }
}

export default function DashboardPage() {
  const { getToken, isLoaded, userId } = useAuth();
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isLoaded) return;
    if (!userId) {
      setIsLoading(false);
      setVideos([]);
      return;
    }

    let cancelled = false;

    async function loadVideos() {
      setIsLoading(true);
      setError(null);

      try {
        const token = await getToken();
        if (!token) {
          throw new Error("Please sign in to continue.");
        }

        const response = await fetch(`${API_URL}/users/me/videos`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error(await parseApiError(response, "Failed to load videos"));
        }

        const data: VideoListItem[] = await response.json();
        if (!cancelled) {
          setVideos(data);
        }
      } catch (err) {
        if (!cancelled) {
          if (err instanceof Error) {
            setError(err.message);
          } else {
            setError("An unexpected error occurred");
          }
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void loadVideos();

    return () => {
      cancelled = true;
    };
  }, [getToken, isLoaded, userId]);

  if (!isLoaded) {
    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <p className="text-zinc-600 dark:text-zinc-400">Loading authentication...</p>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 pb-16 pt-12">
      <div className="flex flex-col gap-2">
        <h1 className="font-heading text-3xl font-bold">My Videos</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Track processing status and jump back into search results.
        </p>
      </div>

      {error && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
          {error}
        </div>
      )}

      <SignedOut>
        <div className="mt-8 rounded-2xl border border-zinc-200 bg-surface-card p-8 text-center dark:border-zinc-800">
          <h2 className="font-heading text-xl font-semibold">Sign in to view your videos</h2>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            Your uploads and processing history live here once you sign in.
          </p>
          <div className="mt-6">
            <SignInButton mode="modal">
              <button className="rounded-lg bg-accent px-6 py-2 text-sm font-medium text-white">
                Sign In
              </button>
            </SignInButton>
          </div>
        </div>
      </SignedOut>

      <SignedIn>
        <div className="mt-8">
          {isLoading ? (
            <div className="flex items-center gap-3 text-sm text-zinc-600 dark:text-zinc-400">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" />
              Loading your videos...
            </div>
          ) : videos.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-zinc-300 bg-surface-card px-6 py-10 text-center text-sm text-zinc-600 dark:border-zinc-700 dark:text-zinc-400">
              <p>No videos yet.</p>
              <Link
                href="/"
                className="mt-3 inline-flex items-center justify-center rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white"
              >
                Process your first video
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {videos.map((video) => (
                <Link
                  key={video.id}
                  href={`/video/${video.id}`}
                  className="block rounded-2xl border border-zinc-200 bg-surface-card px-6 py-4 transition hover:border-accent dark:border-zinc-800"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="space-y-1">
                      <div className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        {video.youtube_url ?? video.source_filename ?? "Uploaded video"}
                      </div>
                      <div className="text-xs text-zinc-500 dark:text-zinc-400">
                        {formatCreatedAt(video.created_at)} · {video.source_type}
                      </div>
                      {video.status === "failed" && video.error_message && (
                        <div className="text-xs text-red-600 dark:text-red-400">
                          {video.error_message}
                        </div>
                      )}
                    </div>
                    <div
                      className={`inline-flex items-center justify-center rounded-full px-3 py-1 text-xs font-medium capitalize ${statusBadgeClass(
                        video.status
                      )}`}
                    >
                      {video.status}
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </SignedIn>
    </div>
  );
}
