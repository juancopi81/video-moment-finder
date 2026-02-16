"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  SignInButton,
  SignedIn,
  SignedOut,
  UserButton,
  useAuth,
} from "@clerk/nextjs";

export default function Home() {
  const router = useRouter();
  const { getToken, isLoaded } = useAuth();
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/videos`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ youtube_url: url }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        let message = "Failed to process video";
        if (Array.isArray(data?.detail)) {
          message = data.detail[0]?.msg || message;
        } else if (typeof data?.detail === "string") {
          message = data.detail;
        } else if (response.status === 401) {
          message = "Please sign in to process a video.";
        }
        throw new Error(message);
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

      <h1 className="text-4xl font-bold mb-4">Video Moment Finder</h1>
      <p className="text-zinc-600 dark:text-zinc-400 mb-8">
        Paste a YouTube URL to find specific moments
      </p>

      <SignedOut>
        <div className="w-full max-w-xl rounded-lg border border-zinc-300 dark:border-zinc-700 p-6 text-center">
          <p className="mb-4 text-zinc-600 dark:text-zinc-400">
            Sign in to start processing your videos.
          </p>
          <SignInButton mode="modal">
            <button className="w-full rounded-lg bg-zinc-900 px-4 py-3 font-medium text-white dark:bg-zinc-100 dark:text-zinc-900">
              Sign In
            </button>
          </SignInButton>
        </div>
      </SignedOut>

      <SignedIn>
        <form onSubmit={handleSubmit} className="w-full max-w-xl">
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
          {error && (
            <p className="mt-2 text-sm text-red-600 dark:text-red-400 text-center">
              {error}
            </p>
          )}
        </form>
      </SignedIn>
    </main>
  );
}
