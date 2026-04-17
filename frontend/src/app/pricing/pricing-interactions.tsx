"use client";

import { Suspense } from "react";
import { useAuth } from "@clerk/nextjs";
import { useSearchParams } from "next/navigation";
import { BillingSummaryCard } from "@/components/billing-summary-card";
import { CheckoutStatusBanner } from "@/components/checkout-status-banner";
import { useBillingSummary } from "@/hooks/useBillingSummary";

function PricingInteractionsInner() {
  const { userId } = useAuth();
  const searchParams = useSearchParams();
  const isSignedIn = !!userId;
  const checkoutStatus = searchParams.get("checkout");
  const { billingSummary, billingSummaryError, isRefreshingBalance } =
    useBillingSummary({ checkoutStatus, enableCheckoutPolling: true });

  return (
    <>
      <CheckoutStatusBanner
        status={checkoutStatus}
        successMessage="Checkout completed. Your latest credit balance is shown below."
        cancelMessage="Checkout was canceled. You can try again when ready."
        className="mt-3"
      />
      {isSignedIn && isRefreshingBalance && (
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          Refreshing your credit balance...
        </p>
      )}
      {isSignedIn && billingSummary && (
        <BillingSummaryCard
          summary={billingSummary}
          className="mt-4 text-left"
        />
      )}
      {isSignedIn && billingSummaryError && (
        <p className="mt-2 text-sm text-red-600 dark:text-red-400">
          {billingSummaryError}
        </p>
      )}
    </>
  );
}

export function PricingInteractions() {
  return (
    <Suspense fallback={null}>
      <PricingInteractionsInner />
    </Suspense>
  );
}
