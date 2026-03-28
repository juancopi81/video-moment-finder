"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { SignInButton, useAuth } from "@clerk/nextjs";
import { AuthLoadingFallback } from "@/components/auth-loading-fallback";
import { CheckoutStatusBanner } from "@/components/checkout-status-banner";
import { useApiBillingSummary } from "@/hooks/useApiBillingSummary";
import { API_URL, parseApiError } from "@/lib/api";
import { startApiCheckout } from "@/lib/api-billing";

type ConnectorTool = {
  name: string;
  title: string;
  description: string;
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

function ClaudeConnectorContent() {
  const { getToken, isLoaded, userId } = useAuth();
  const searchParams = useSearchParams();
  const requestId = searchParams.get("request_id");
  const checkoutStatus = searchParams.get("checkout");
  const [requestData, setRequestData] = useState<ConnectorRequest | null>(null);
  const [requestLoading, setRequestLoading] = useState(true);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [decisionLoading, setDecisionLoading] = useState<"approve" | "deny" | null>(
    null,
  );
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const [checkoutLoading, setCheckoutLoading] = useState(false);
  const [checkoutError, setCheckoutError] = useState<string | null>(null);

  useEffect(() => {
    if (!requestId) {
      setRequestLoading(false);
      setRequestData(null);
      setRequestError("Missing request_id. Restart Connect from Claude.");
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
              "Failed to load Claude connector request.",
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
              : "Failed to load Claude connector request.",
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
  const { apiBillingSummary, apiBillingSummaryError, isRefreshingBalance } =
    useApiBillingSummary({ checkoutStatus, enableCheckoutPolling: true });

  async function handleCheckout() {
    setCheckoutError(null);
    const token = await getToken();
    if (!token || !requestId) {
      setCheckoutError("Please sign in to continue.");
      return;
    }
    setCheckoutLoading(true);
    try {
      const returnPath = `/connectors/claude?request_id=${encodeURIComponent(requestId)}`;
      const url = await startApiCheckout(token, returnPath);
      window.location.assign(url);
    } catch (error) {
      setCheckoutError(
        error instanceof Error
          ? error.message
          : "Failed to start checkout.",
      );
      setCheckoutLoading(false);
    }
  }

  async function handleDecision(action: "approve" | "deny") {
    if (!requestId) {
      setDecisionError("Missing request_id. Restart Connect from Claude.");
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
          Claude Connector
        </p>
        <h1 className="mt-3 font-heading text-4xl font-bold">
          Connect Claude to Video Moment Finder
        </h1>
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Review the tools Claude will be able to use, confirm billing is ready,
          and approve the connection for your account.
        </p>

        <CheckoutStatusBanner
          status={checkoutStatus}
          successMessage="Developer Pack purchased. Your connector approval can continue below."
          cancelMessage="Checkout was canceled. Your connector request is still waiting for approval."
          className="mt-4"
        />

        {requestError && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
            {requestError}
          </div>
        )}

        {apiBillingSummaryError && isSignedIn && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
            {apiBillingSummaryError}
          </div>
        )}

        {checkoutError && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
            {checkoutError}
          </div>
        )}

        {decisionError && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
            {decisionError}
          </div>
        )}

        {requestLoading ? (
          <div className="mt-6 flex items-center gap-3 text-sm text-zinc-600 dark:text-zinc-400">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" />
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
                  <p className="mt-1 text-sm">
                    Scope: <code>{requestData.scope}</code>
                  </p>
                  <p className="mt-1 text-sm">
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
                  Use your Video Moment Finder account to approve this Claude connection.
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

            {isSignedIn && requestData.status === "pending" && !hasApiBalance && (
              <div className="mt-8 rounded-2xl border border-dashed border-zinc-300 p-6 dark:border-zinc-700">
                <h2 className="font-heading text-xl font-semibold">
                  Add API units before connecting
                </h2>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                  Claude connector usage bills against Developer Pack API units. Buy a pack, then return here to approve.
                </p>
                {isRefreshingBalance && (
                  <p className="mt-3 text-sm text-zinc-500">
                    Refreshing your balance...
                  </p>
                )}
                <div className="mt-5 flex flex-col gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={handleCheckout}
                    disabled={checkoutLoading}
                    className="rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {checkoutLoading ? "Opening checkout..." : "Buy Developer Pack"}
                  </button>
                  <Link
                    href="/dashboard/api"
                    className="rounded-lg border border-zinc-300 px-5 py-2 text-sm font-medium text-zinc-900 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-900"
                  >
                    Open API dashboard
                  </Link>
                </div>
              </div>
            )}

            {isSignedIn && requestData.status === "pending" && hasApiBalance && (
              <div className="mt-8 rounded-2xl border border-zinc-200 bg-white/80 p-6 dark:border-zinc-800 dark:bg-zinc-950/40">
                <h2 className="font-heading text-xl font-semibold">
                  Approve Claude access
                </h2>
                <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                  Claude will only be able to use the four tools shown above for your account. Uploads and searches consume API units from your Developer Pack balance.
                </p>
                {apiBillingSummary && (
                  <p className="mt-3 text-sm text-zinc-500">
                    Current balance: {apiBillingSummary.api_units_balance.toLocaleString()} units
                  </p>
                )}
                <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={() => void handleDecision("approve")}
                    disabled={decisionLoading !== null}
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
                  the connection flow from Claude if you need a new session.
                </p>
              </div>
            )}
          </>
        ) : null}

        <div className="mt-10 flex flex-wrap gap-4 text-sm text-zinc-500">
          <Link href="/developers" className="hover:text-zinc-900 dark:hover:text-zinc-100">
            Developers docs
          </Link>
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

export default function ClaudeConnectorPage() {
  return (
    <Suspense fallback={<AuthLoadingFallback />}>
      <ClaudeConnectorContent />
    </Suspense>
  );
}
