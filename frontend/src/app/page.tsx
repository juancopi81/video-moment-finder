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

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const formData = new FormData();
      formData.append("file", uploadFile);

      const response = await new Promise<Response>((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open("POST", `${apiUrl}/videos/upload`);
        request.setRequestHeader("Authorization", `Bearer ${token}`);
        request.responseType = "text";

        request.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            setUploadProgress(Math.round((event.loaded / event.total) * 100));
          }
        };

        request.onload = () => {
          const response = new Response(request.responseText, {
            status: request.status,
            statusText: request.statusText,
          });
          resolve(response);
        };

        request.onerror = () => {
          reject(new TypeError("Network error"));
        };

        request.send(formData);
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        let message = "Failed to upload video";
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
        <div className="w-full max-w-xl">
          <div className="mb-4 flex gap-2">
            <button
              type="button"
              onClick={() => handleModeChange("youtube")}
              className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium ${
                mode === "youtube"
                  ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                  : "border border-zinc-300 text-zinc-700 dark:border-zinc-700 dark:text-zinc-200"
              }`}
              disabled={isLoading || isUploading}
            >
              YouTube URL
            </button>
            <button
              type="button"
              onClick={() => handleModeChange("upload")}
              className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium ${
                mode === "upload"
                  ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                  : "border border-zinc-300 text-zinc-700 dark:border-zinc-700 dark:text-zinc-200"
              }`}
              disabled={isLoading || isUploading}
            >
              Upload Video
            </button>
          </div>

          {mode === "youtube" ? (
            <form onSubmit={handleSubmit}>
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
            <form onSubmit={handleUpload}>
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
    </main>
  );
}
