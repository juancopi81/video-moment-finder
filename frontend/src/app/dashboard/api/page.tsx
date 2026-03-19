"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useAuth } from "@clerk/nextjs";
import { AuthLoadingFallback } from "@/components/auth-loading-fallback";
import { ApiBalanceCard } from "@/components/api-balance-card";
import { ApiKeyCreateModal } from "@/components/api-key-create-modal";
import { CheckoutStatusBanner } from "@/components/checkout-status-banner";
import { useApiBillingSummary } from "@/hooks/useApiBillingSummary";
import { API_URL, parseApiError } from "@/lib/api";
import {
  startApiCheckout,
  fetchApiUsageEvents,
  ApiUsageEvent,
} from "@/lib/api-billing";

type ApiKeyItem = {
  id: string;
  name: string;
  key_prefix: string;
  created_at: string | null;
  last_used_at: string | null;
};

function formatDate(value: string | null): string {
  if (!value) return "Never";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

function ApiDashboardContent() {
  const { getToken, isLoaded, userId } = useAuth();
  const searchParams = useSearchParams();
  const checkoutStatus = searchParams.get("checkout");
  const { apiBillingSummary, apiBillingSummaryError, isRefreshingBalance } =
    useApiBillingSummary({ checkoutStatus, enableCheckoutPolling: true });

  const [keys, setKeys] = useState<ApiKeyItem[]>([]);
  const [keysLoading, setKeysLoading] = useState(true);
  const [keysError, setKeysError] = useState<string | null>(null);
  const [usageEvents, setUsageEvents] = useState<ApiUsageEvent[]>([]);
  const [usageLoading, setUsageLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [checkoutLoading, setCheckoutLoading] = useState(false);
  const [checkoutError, setCheckoutError] = useState<string | null>(null);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    const t = await getToken();
    if (!t) return;
    setToken(t);
    setKeysLoading(true);
    setUsageLoading(true);

    const keysPromise = fetch(`${API_URL}/api/v1/keys`, {
      headers: { Authorization: `Bearer ${t}` },
    })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await parseApiError(res, "Failed to load API keys."));
        }
        setKeys((await res.json()) as ApiKeyItem[]);
        setKeysError(null);
      })
      .catch((err: unknown) => {
        setKeysError(
          err instanceof Error ? err.message : "Failed to load API keys.",
        );
      })
      .finally(() => setKeysLoading(false));

    const usagePromise = fetchApiUsageEvents(t, { limit: 50 })
      .then((events) => setUsageEvents(events))
      .catch(() => {
        // Non-critical — silently ignore
      })
      .finally(() => setUsageLoading(false));

    await Promise.all([keysPromise, usagePromise]);
  }, [getToken]);

  useEffect(() => {
    if (!isLoaded || !userId) return;
    void loadData();
  }, [isLoaded, userId, loadData]);

  async function handleCheckout() {
    setCheckoutError(null);
    const t = await getToken();
    if (!t) {
      setCheckoutError("Please sign in to continue.");
      return;
    }
    setCheckoutLoading(true);
    try {
      const url = await startApiCheckout(t);
      window.location.assign(url);
    } catch (err) {
      setCheckoutError(
        err instanceof Error ? err.message : "Failed to start checkout.",
      );
      setCheckoutLoading(false);
    }
  }

  async function handleRevoke(keyId: string) {
    const t = await getToken();
    if (!t) return;
    setRevoking(keyId);
    try {
      await fetch(`${API_URL}/api/v1/keys/${keyId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${t}` },
      });
      setKeys((prev) => prev.filter((k) => k.id !== keyId));
    } catch {
      // Ignore — next refresh will show current state
    } finally {
      setRevoking(null);
    }
  }

  if (!isLoaded) {
    return <AuthLoadingFallback />;
  }

  const hasBalance =
    apiBillingSummary !== null && apiBillingSummary.api_units_balance > 0;
  const hasKeys = keys.length > 0;

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 pb-16 pt-12">
      <div className="flex flex-col gap-2">
        <h1 className="font-heading text-3xl font-bold">API Access</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Manage your API keys and developer billing.
        </p>
        <CheckoutStatusBanner
          status={checkoutStatus}
          successMessage="Developer Pack purchased. Balance refreshed below."
          cancelMessage="Checkout was canceled. No units were added."
          className="mt-1"
        />
      </div>

      {checkoutError && (
        <p className="mt-4 text-sm text-red-600 dark:text-red-400">
          {checkoutError}
        </p>
      )}

      {isRefreshingBalance && (
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          Refreshing your balance...
        </p>
      )}

      {apiBillingSummaryError && (
        <p className="mt-4 text-sm text-red-600 dark:text-red-400">
          {apiBillingSummaryError}
        </p>
      )}

      <div className="mt-8 space-y-6">
        {/* State 1: No balance — get started */}
        {!hasBalance && !keysLoading && (
          <div className="rounded-2xl border border-dashed border-zinc-300 bg-surface-card p-8 text-center dark:border-zinc-700">
            <h2 className="font-heading text-xl font-semibold">
              Get started with the API
            </h2>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Purchase a Developer Pack to get 10,000 API units. Then create a
              key to authenticate your requests.
            </p>
            <div className="mt-6 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
              <button
                type="button"
                onClick={handleCheckout}
                disabled={checkoutLoading}
                className="rounded-lg bg-accent px-6 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                {checkoutLoading
                  ? "Opening checkout..."
                  : "Buy Developer Pack"}
              </button>
              <Link
                href="/developers"
                className="text-sm text-zinc-600 hover:text-foreground dark:text-zinc-400"
              >
                Learn more
              </Link>
            </div>
          </div>
        )}

        {/* Balance card */}
        {apiBillingSummary && hasBalance && (
          <ApiBalanceCard summary={apiBillingSummary} />
        )}

        {/* State 2: Balance exists, no keys — create first key */}
        {hasBalance && !hasKeys && !keysLoading && (
          <div className="rounded-2xl border border-dashed border-zinc-300 bg-surface-card p-8 text-center dark:border-zinc-700">
            <h2 className="font-heading text-lg font-semibold">
              Create your first API key
            </h2>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              API keys authenticate your CLI and programmatic requests.
            </p>
            <button
              type="button"
              onClick={() => setShowCreateModal(true)}
              className="mt-4 rounded-lg bg-accent px-6 py-2 text-sm font-medium text-white"
            >
              Create API Key
            </button>
          </div>
        )}

        {/* State 3: Keys exist */}
        {hasKeys && (
          <>
            <div className="flex items-center justify-between">
              <h2 className="font-heading text-lg font-semibold">API Keys</h2>
              <button
                type="button"
                onClick={() => setShowCreateModal(true)}
                className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white"
              >
                Create Key
              </button>
            </div>

            {keysError && (
              <p className="text-sm text-red-600 dark:text-red-400">
                {keysError}
              </p>
            )}

            <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
                    <th className="px-4 py-2 text-left font-medium">Name</th>
                    <th className="px-4 py-2 text-left font-medium">Prefix</th>
                    <th className="px-4 py-2 text-left font-medium">
                      Created
                    </th>
                    <th className="px-4 py-2 text-left font-medium">
                      Last Used
                    </th>
                    <th className="px-4 py-2 text-right font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {keys.map((key) => (
                    <tr
                      key={key.id}
                      className="border-b border-zinc-100 last:border-0 dark:border-zinc-800"
                    >
                      <td className="px-4 py-2">{key.name || "Unnamed"}</td>
                      <td className="px-4 py-2">
                        <code className="text-xs">{key.key_prefix}...</code>
                      </td>
                      <td className="px-4 py-2 text-zinc-600 dark:text-zinc-400">
                        {formatDate(key.created_at)}
                      </td>
                      <td className="px-4 py-2 text-zinc-600 dark:text-zinc-400">
                        {formatDate(key.last_used_at)}
                      </td>
                      <td className="px-4 py-2 text-right">
                        <button
                          type="button"
                          onClick={() => handleRevoke(key.id)}
                          disabled={revoking === key.id}
                          className="text-xs text-red-600 hover:text-red-700 disabled:opacity-60 dark:text-red-400"
                        >
                          {revoking === key.id ? "Revoking..." : "Revoke"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* Usage table */}
        {usageEvents.length > 0 && !usageLoading && (
          <>
            <h2 className="font-heading text-lg font-semibold">
              Recent Usage
            </h2>
            <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
                    <th className="px-4 py-2 text-left font-medium">Type</th>
                    <th className="px-4 py-2 text-left font-medium">Units</th>
                    <th className="px-4 py-2 text-left font-medium">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {usageEvents.map((evt) => (
                    <tr
                      key={evt.id}
                      className="border-b border-zinc-100 last:border-0 dark:border-zinc-800"
                    >
                      <td className="px-4 py-2 capitalize">
                        {evt.event_type.replace("_", " ")}
                      </td>
                      <td className="px-4 py-2">
                        {evt.units > 0 ? `-${evt.units}` : `+${Math.abs(evt.units)}`}
                      </td>
                      <td className="px-4 py-2 text-zinc-600 dark:text-zinc-400">
                        {formatDate(evt.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* Buy more */}
        {hasBalance && (
          <div className="text-center">
            <button
              type="button"
              onClick={handleCheckout}
              disabled={checkoutLoading}
              className="rounded-lg border border-zinc-200 px-4 py-2 text-sm font-medium dark:border-zinc-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {checkoutLoading ? "Opening checkout..." : "Buy more units"}
            </button>
          </div>
        )}
      </div>

      {showCreateModal && token && (
        <ApiKeyCreateModal
          token={token}
          onCreated={() => void loadData()}
          onClose={() => setShowCreateModal(false)}
        />
      )}
    </div>
  );
}

export default function ApiDashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="flex flex-1 items-center justify-center p-8">
          <p className="text-zinc-600 dark:text-zinc-400">
            Loading API dashboard...
          </p>
        </div>
      }
    >
      <ApiDashboardContent />
    </Suspense>
  );
}
