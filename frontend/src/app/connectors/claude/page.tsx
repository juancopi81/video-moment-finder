"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { SignInButton, useAuth, useUser } from "@clerk/nextjs";
import { AuthLoadingFallback } from "@/components/auth-loading-fallback";
import { ApiTrialNotice } from "@/components/api-trial-notice";
import { useApiBillingSummary } from "@/hooks/useApiBillingSummary";
import { API_URL, parseApiError } from "@/lib/api";

type ConnectorTool = {
  name: string;
  title: string;
  description: string;
  cost: string;
};

type ConnectorRequest = {
  request_id: string;
  client_id: string;
  resource: string;
  scope: string;
  scopes: string[];
  status: "pending" | "approved" | "denied" | "expired";
  expires_at: string | null;
  tools: ConnectorTool[];
};

function formatExpiresAt(value: string | null): string {
  if (!value) {
    return "soon";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div role="alert" className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
      {message}
    </div>
  );
}

function statusTone(status: ConnectorRequest["status"]): string {
  switch (status) {
    case "approved":
      return "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-200";
    case "denied":
      return "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200";
    case "expired":
      return "border-red-200 bg-red-50 text-red-800 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200";
    default:
      return "border-zinc-200 bg-surface-card text-zinc-900 dark:border-zinc-800 dark:text-zinc-100";
  }
}

function ConnectorContent() {
  const { getToken, isLoaded, userId } = useAuth();
  const { user } = useUser();
  const searchParams = useSearchParams();
  const requestId = searchParams.get("request_id");
  const [requestData, setRequestData] = useState<ConnectorRequest | null>(null);
  const [requestLoading, setRequestLoading] = useState(true);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [decisionLoading, setDecisionLoading] = useState<"approve" | "deny" | null>(
    null,
  );
  const [decisionError, setDecisionError] = useState<string | null>(null);

  useEffect(() => {
    if (!requestId) {
      setRequestLoading(false);
      setRequestData(null);
      setRequestError("This connection link is incomplete. Restart Connect in the app you came from.");
      return;
    }

    let cancelled = false;
    setRequestLoading(true);
    const currentRequestId = requestId;

    async function loadRequest(): Promise<void> {
      try {
        const response = await fetch(
          `${API_URL}/oauth/mcp/requests/${encodeURIComponent(currentRequestId)}`,
          { cache: "no-store" },
        );
        if (!response.ok) {
          throw new Error(
            await parseApiError(
              response,
              "Failed to load the Video Moment Finder connection request.",
            ),
          );
        }
        const payload = (await response.json()) as ConnectorRequest;
        if (!cancelled) {
          setRequestData(payload);
          setRequestError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setRequestData(null);
          setRequestError(
            error instanceof Error
              ? error.message
              : "Failed to load the Video Moment Finder connection request.",
          );
        }
      } finally {
        if (!cancelled) {
          setRequestLoading(false);
        }
      }
    }

    void loadRequest();

    return () => {
      cancelled = true;
    };
  }, [requestId]);
  const { apiBillingSummary, apiBillingSummaryError, isLoadingBalance, refreshBalance } =
    useApiBillingSummary({});

  async function handleDecision(action: "approve" | "deny") {
    if (!requestId) {
      setDecisionError("This connection link is incomplete. Restart Connect in the app you came from.");
      return;
    }
    setDecisionError(null);
    const token = await getToken();
    if (!token) {
      setDecisionError("Please sign in to continue.");
      return;
    }
    setDecisionLoading(action);
    try {
      const response = await fetch(
        `${API_URL}/oauth/mcp/requests/${encodeURIComponent(requestId)}/${action}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );
      if (!response.ok) {
        throw new Error(
          await parseApiError(
            response,
            `Failed to ${action} connector request.`,
          ),
        );
      }
      const payload = (await response.json()) as { redirect_url: string };
      window.location.assign(payload.redirect_url);
    } catch (error) {
      setDecisionError(
        error instanceof Error
          ? error.message
          : `Failed to ${action} connector request.`,
      );
      setDecisionLoading(null);
    }
  }

  if (!isLoaded) {
    return <AuthLoadingFallback />;
  }

  const isSignedIn = !!userId;
  const hasApiBalance =
    apiBillingSummary !== null && apiBillingSummary.api_units_balance > 0;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 pb-16 pt-12">
      <div className="rounded-3xl border border-zinc-200 bg-surface-card p-8 shadow-sm dark:border-zinc-800">
        <p className="text-sm font-medium uppercase tracking-[0.2em] text-accent">
          Account connection
        </p>
        <h1 className="mt-3 font-heading text-4xl font-bold">
          Connect Video Moment Finder
        </h1>
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Connect the app you came from to your Video Moment Finder account.
          Review access to your videos, transcripts, and frames and the API units
          each operation uses before approving.
        </p>

        {requestError && <ErrorBanner message={requestError} />}
        {apiBillingSummaryError && isSignedIn && <ErrorBanner message={apiBillingSummaryError} />}
        {decisionError && <ErrorBanner message={decisionError} />}

        {requestLoading ? (
          <div role="status" className="mt-6 flex items-center gap-3 text-sm text-zinc-600 dark:text-zinc-400">
            <div aria-hidden="true" className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" />
            Loading connector request...
          </div>
        ) : requestData ? (
          <>
            <div
              className={`mt-6 rounded-2xl border p-5 ${statusTone(requestData.status)}`}
            >
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-medium">
                    Request status:{" "}
                    <span className="capitalize">{requestData.status}</span>
                  </p>
                  <p className="mt-1 break-all text-sm">
                    Requesting app: <code>{requestData.client_id}</code>
                  </p>
                  <p className="mt-1 break-all text-sm">
                    Resource: <code>{requestData.resource}</code>
                  </p>
                </div>
                <p className="text-sm">
                  Expires: {formatExpiresAt(requestData.expires_at)}
                </p>
              </div>
            </div>

            <div className="mt-6">
              <h2 className="font-heading text-xl font-semibold">
                Tools this connector can use
              </h2>
              <div className="mt-4 space-y-3">
                {requestData.tools.map((tool) => (
                  <div
                    key={tool.name}
                    className="rounded-2xl border border-zinc-200 bg-white/80 p-4 dark:border-zinc-800 dark:bg-zinc-950/40"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="font-medium text-zinc-900 dark:text-zinc-100">
                          {tool.title}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                          <code>{tool.name}</code>
                        </p>
                      </div>
                      <span className="rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                        {tool.name === "upload_video" ? "Write" : "Read"}
                      </span>
                    </div>
                    <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
                      {tool.description}
                    </p>
                    <p className="mt-2 text-xs font-medium text-zinc-500 dark:text-zinc-400">
                      Cost: {tool.cost}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {!isSignedIn && requestData.status === "pending" && (
              <div className="mt-8 rounded-2xl border border-dashed border-zinc-300 p-6 text-center dark:border-zinc-700">
                <h2 className="font-heading text-xl font-semibold">
                  Sign in to continue
                </h2>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                  Use the Video Moment Finder account that holds your videos.
                  This connection only accesses that account. An API key is not needed.
                </p>
                <SignInButton mode="modal">
                  <button
                    type="button"
                    className="mt-4 rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white"
                  >
                    Sign in or create account
                  </button>
                </SignInButton>
              </div>
            )}

            {isSignedIn && requestData.status === "pending" && (
              <div className="mt-8 rounded-2xl border border-zinc-200 bg-white/80 p-6 dark:border-zinc-800 dark:bg-zinc-950/40">
                <h2 className="font-heading text-xl font-semibold">
                  Review your account access
                </h2>
                <p className="mt-2 break-words text-sm font-medium">
                  Signed in as {user?.primaryEmailAddress?.emailAddress ?? user?.fullName ?? "your Video Moment Finder account"}
                </p>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                  This app can use the tools above for your account, including uploading
                  a video. Indexing, searches, transcript retrieval, and frame retrieval
                  use API units. Video listing and status checks use no units. Approving
                  this connection does not itself start a video operation.
                </p>
                <div aria-live="polite" className="mt-4">
                  {isLoadingBalance && <p className="text-sm">Checking your account allowance...</p>}
                  {apiBillingSummary && (
                    <>
                      <p className="text-sm font-medium">
                        Current balance: {apiBillingSummary.api_units_balance.toLocaleString()} API units
                      </p>
                      <ApiTrialNotice summary={apiBillingSummary} />
                      {!hasApiBalance && (
                        <p id="connection-limit" className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                          This account has no API units available, so this connection cannot
                          be approved yet. Metered video operations pause when the balance
                          is exhausted. You can continue studying material already retrieved
                          in your chat. Reconnecting does not reset an allowance.
                        </p>
                      )}
                    </>
                  )}
                  {apiBillingSummaryError && <p id="connection-limit" className="text-sm">Account allowance could not be checked. Refresh before approving.</p>}
                </div>
                <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
                  Website video credits and API units are different balances. This
                  connection uses API units; its actual per-call rates are shown above.
                </p>
                <button
                  type="button"
                  onClick={refreshBalance}
                  disabled={isLoadingBalance || decisionLoading !== null}
                  className="mt-3 text-sm text-accent underline underline-offset-4 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Refresh account allowance
                </button>
                <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={() => void handleDecision("approve")}
                    disabled={decisionLoading !== null || !hasApiBalance}
                    aria-describedby={!hasApiBalance && !isLoadingBalance ? "connection-limit" : undefined}
                    className="rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {decisionLoading === "approve" ? "Approving..." : "Approve and continue"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleDecision("deny")}
                    disabled={decisionLoading !== null}
                    className="rounded-lg border border-zinc-300 px-5 py-2 text-sm font-medium text-zinc-900 hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-900"
                  >
                    {decisionLoading === "deny" ? "Declining..." : "Deny"}
                  </button>
                </div>
              </div>
            )}

            {requestData.status !== "pending" && (
              <div className="mt-8 rounded-2xl border border-zinc-200 bg-white/80 p-6 dark:border-zinc-800 dark:bg-zinc-950/40">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  This connector request is already {requestData.status}. Restart
                  the connection flow in your app if you need a new session.
                </p>
              </div>
            )}
          </>
        ) : null}

        <section className="mt-8 rounded-2xl border border-zinc-200 p-6 dark:border-zinc-800" aria-labelledby="first-result">
          <h2 id="first-result" className="font-heading text-xl font-semibold">Your first learning result</h2>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            After connecting, return to your app and ask: &ldquo;List my ready videos,
            then help me understand one key idea from a lecture, with a timestamp
            and a frame where available.&rdquo;
          </p>
          <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
            Choose a video in the connected account. If it is still processing,
            wait for it to be ready; if it failed or is unavailable, choose another.
            A public video URL alone does not make its content available to this
            connection. Upload only a file you own or are authorized to use, after
            reviewing the indexing cost. If a frame cannot be read, the answer should
            say so and use the transcript without inventing visual details.
          </p>
        </section>

        <div className="mt-10 flex flex-wrap gap-4 text-sm text-zinc-500">
          <Link href="/privacy" className="hover:text-zinc-900 dark:hover:text-zinc-100">
            Privacy
          </Link>
          <Link href="/support" className="hover:text-zinc-900 dark:hover:text-zinc-100">
            Support
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function ConnectorPage() {
  return (
    <Suspense fallback={<AuthLoadingFallback />}>
      <ConnectorContent />
    </Suspense>
  );
}
