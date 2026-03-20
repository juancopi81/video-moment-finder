"use client";

import { useEffect, useState } from "react";
import { API_URL, parseApiError } from "@/lib/api";

type ApiKeyCreateModalProps = {
  getToken: () => Promise<string | null>;
  onCreated: () => void;
  onClose: () => void;
};

export function ApiKeyCreateModal({
  getToken,
  onCreated,
  onClose,
}: ApiKeyCreateModalProps) {
  const [name, setName] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  async function handleCreate() {
    setError(null);
    setIsCreating(true);
    try {
      const token = await getToken();
      if (!token) {
        throw new Error("Please sign in to continue.");
      }
      const response = await fetch(`${API_URL}/api/v1/keys`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ name: name.trim() }),
      });
      if (!response.ok) {
        throw new Error(
          await parseApiError(response, "Failed to create API key."),
        );
      }
      const data = await response.json();
      setCreatedKey(data.key);
      onCreated();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to create API key.",
      );
    } finally {
      setIsCreating(false);
    }
  }

  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(id);
  }, [copied]);

  async function handleCopy() {
    if (!createdKey) return;
    try {
      await navigator.clipboard.writeText(createdKey);
      setCopied(true);
    } catch {
      // Clipboard API may fail in some contexts — ignore
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 animate-fade-in"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      role="presentation"
    >
      <div className="mx-4 w-full max-w-md animate-fade-in-up rounded-xl border border-zinc-200 bg-background p-6 shadow-xl dark:border-zinc-800">
        {createdKey ? (
          <>
            <h2 className="font-heading text-lg font-bold">
              API Key Created
            </h2>
            <p className="mt-2 text-sm text-amber-600 dark:text-amber-400">
              This key will not be shown again. Copy it now.
            </p>
            <div className="mt-3 rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-900">
              <code className="break-all text-sm">{createdKey}</code>
            </div>
            <div className="mt-4 flex gap-3">
              <button
                type="button"
                onClick={handleCopy}
                className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white"
              >
                {copied ? "Copied!" : "Copy to clipboard"}
              </button>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg border border-zinc-200 px-4 py-2 text-sm font-medium dark:border-zinc-700"
              >
                Done
              </button>
            </div>
          </>
        ) : (
          <>
            <h2 className="font-heading text-lg font-bold">
              Create API Key
            </h2>
            <label className="mt-4 block text-sm font-medium">
              Key name (optional)
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. my-agent"
                className="mt-1 block w-full rounded-lg border border-zinc-200 bg-background px-3 py-2 text-sm dark:border-zinc-700"
              />
            </label>
            {error && (
              <p className="mt-2 text-sm text-red-600 dark:text-red-400">
                {error}
              </p>
            )}
            <div className="mt-6 flex gap-3">
              <button
                type="button"
                onClick={handleCreate}
                disabled={isCreating}
                className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isCreating ? "Creating..." : "Create"}
              </button>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg border border-zinc-200 px-4 py-2 text-sm font-medium dark:border-zinc-700"
              >
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
