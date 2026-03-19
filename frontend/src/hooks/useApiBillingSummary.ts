import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import {
  fetchApiBillingSummary,
  ApiBillingSummary,
} from "@/lib/api-billing";

const CHECKOUT_POLL_ATTEMPTS = 4;
const CHECKOUT_POLL_INTERVAL_MS = 2500;

type UseApiBillingSummaryOptions = {
  checkoutStatus: string | null;
  enableCheckoutPolling?: boolean;
};

type UseApiBillingSummaryResult = {
  apiBillingSummary: ApiBillingSummary | null;
  apiBillingSummaryError: string | null;
  isRefreshingBalance: boolean;
};

export function useApiBillingSummary({
  checkoutStatus,
  enableCheckoutPolling = false,
}: UseApiBillingSummaryOptions): UseApiBillingSummaryResult {
  const { getToken, isLoaded, userId } = useAuth();
  const [apiBillingSummary, setApiBillingSummary] =
    useState<ApiBillingSummary | null>(null);
  const [apiBillingSummaryError, setApiBillingSummaryError] = useState<
    string | null
  >(null);
  const [isRefreshingBalance, setIsRefreshingBalance] = useState(false);

  useEffect(() => {
    if (!isLoaded) return;
    if (!userId) {
      setApiBillingSummary(null);
      setApiBillingSummaryError(null);
      setIsRefreshingBalance(false);
      return;
    }

    let cancelled = false;
    const shouldPoll =
      enableCheckoutPolling && checkoutStatus === "success";
    const pollAttempts = shouldPoll ? CHECKOUT_POLL_ATTEMPTS : 1;
    setIsRefreshingBalance(shouldPoll);

    async function loadSummary(): Promise<void> {
      try {
        const token = await getToken();
        if (!token) {
          throw new Error("Please sign in to continue.");
        }
        const summary = await fetchApiBillingSummary(token);
        if (!cancelled) {
          setApiBillingSummary(summary);
          setApiBillingSummaryError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setApiBillingSummaryError(
            err instanceof Error
              ? err.message
              : "Failed to load API billing summary.",
          );
        }
      }
    }

    void loadSummary();

    if (pollAttempts <= 1) {
      setIsRefreshingBalance(false);
      return () => {
        cancelled = true;
      };
    }

    let runs = 1;
    const intervalId = window.setInterval(() => {
      runs += 1;
      void loadSummary();
      if (runs >= pollAttempts) {
        window.clearInterval(intervalId);
        if (!cancelled) {
          setIsRefreshingBalance(false);
        }
      }
    }, CHECKOUT_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [checkoutStatus, enableCheckoutPolling, getToken, isLoaded, userId]);

  return { apiBillingSummary, apiBillingSummaryError, isRefreshingBalance };
}
