import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { fetchBillingSummary, BillingSummary } from "@/lib/billing";

const CHECKOUT_POLL_ATTEMPTS = 4;
const CHECKOUT_POLL_INTERVAL_MS = 2500;

type UseBillingSummaryOptions = {
  checkoutStatus: string | null;
  enableCheckoutPolling?: boolean;
};

type UseBillingSummaryResult = {
  billingSummary: BillingSummary | null;
  billingSummaryError: string | null;
  isRefreshingBalance: boolean;
};

export function useBillingSummary({
  checkoutStatus,
  enableCheckoutPolling = false,
}: UseBillingSummaryOptions): UseBillingSummaryResult {
  const { getToken, isLoaded, userId } = useAuth();
  const [billingSummary, setBillingSummary] = useState<BillingSummary | null>(
    null,
  );
  const [billingSummaryError, setBillingSummaryError] = useState<string | null>(
    null,
  );
  const [isRefreshingBalance, setIsRefreshingBalance] = useState(false);

  useEffect(() => {
    if (!isLoaded) return;
    if (!userId) {
      setBillingSummary(null);
      setBillingSummaryError(null);
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
        const summary = await fetchBillingSummary(token);
        if (!cancelled) {
          setBillingSummary(summary);
          setBillingSummaryError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setBillingSummaryError(
            err instanceof Error
              ? err.message
              : "Failed to load billing summary.",
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

  return { billingSummary, billingSummaryError, isRefreshingBalance };
}
