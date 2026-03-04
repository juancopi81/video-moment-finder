"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { useSearchParams } from "next/navigation";
import { PricingCard } from "@/components/pricing-card";
import { API_URL, parseApiError } from "@/lib/api";
import { BillingSummary, fetchBillingSummary } from "@/lib/billing";

type BillingPlan = "starter" | "pro";
type TierId = "free" | BillingPlan;
type Tier = {
  id: TierId;
  name: string;
  price: string;
  description: string;
  features: string[];
  highlighted?: boolean;
  ctaHref: string;
};

const tiers: Tier[] = [
  {
    id: "free",
    name: "Free Trial",
    price: "$0",
    description: "Try it out with one video",
    features: [
      "1 video credit",
      "Up to 30-minute videos",
      "Text-based moment search",
      "Thumbnail previews",
    ],
    ctaHref: "/",
  },
  {
    id: "starter",
    name: "Starter",
    price: "$5",
    description: "5 video credits",
    features: [
      "5 video credits",
      "Up to 30-minute videos",
      "Text-based moment search",
      "Thumbnail previews",
      "YouTube URL & direct upload",
    ],
    highlighted: true,
    ctaHref: "/",
  },
  {
    id: "pro",
    name: "Pro",
    price: "$15",
    description: "20 video credits",
    features: [
      "20 video credits",
      "Up to 30-minute videos",
      "Text-based moment search",
      "Thumbnail previews",
      "YouTube URL & direct upload",
      "Priority processing",
    ],
    ctaHref: "/",
  },
];

function ctaLabel({
  isSignedIn,
  paidPlan,
  isCheckoutLoading,
}: {
  isSignedIn: boolean;
  paidPlan: BillingPlan | null;
  isCheckoutLoading: boolean;
}): string {
  if (!isSignedIn) {
    return "Get started";
  }
  if (!paidPlan) {
    return "Process a video";
  }
  if (isCheckoutLoading) {
    return "Opening checkout...";
  }
  return "Buy credits";
}

export default function PricingPage() {
  const { userId, getToken, isLoaded } = useAuth();
  const searchParams = useSearchParams();
  const [checkoutPlanLoading, setCheckoutPlanLoading] = useState<BillingPlan | null>(
    null,
  );
  const [checkoutError, setCheckoutError] = useState<string | null>(null);
  const [billingSummary, setBillingSummary] = useState<BillingSummary | null>(null);
  const [billingSummaryError, setBillingSummaryError] = useState<string | null>(null);
  const isSignedIn = !!userId;
  const checkoutStatus = searchParams.get("checkout");

  useEffect(() => {
    if (!isLoaded) {
      return;
    }
    if (!userId) {
      setBillingSummary(null);
      setBillingSummaryError(null);
      return;
    }

    let cancelled = false;
    const pollAttempts = checkoutStatus === "success" ? 4 : 1;
    const pollIntervalMs = 2500;

    const loadSummary = async (): Promise<void> => {
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
            err instanceof Error ? err.message : "Failed to load billing summary.",
          );
        }
      }
    };

    void loadSummary();

    if (pollAttempts <= 1) {
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
      }
    }, pollIntervalMs);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [checkoutStatus, getToken, isLoaded, userId]);

  async function startCheckout(plan: BillingPlan): Promise<void> {
    setCheckoutError(null);
    if (!isLoaded) {
      return;
    }

    const token = await getToken();
    if (!token) {
      setCheckoutError("Please sign in to buy credits.");
      return;
    }

    setCheckoutPlanLoading(plan);
    try {
      const response = await fetch(`${API_URL}/billing/checkout`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ plan }),
      });
      if (!response.ok) {
        throw new Error(
          await parseApiError(response, "Failed to start checkout. Please try again."),
        );
      }

      const payload = await response.json();
      if (typeof payload.checkout_url !== "string" || !payload.checkout_url) {
        throw new Error("Checkout URL missing in response.");
      }

      window.location.assign(payload.checkout_url);
    } catch (err) {
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setCheckoutError("Cannot connect to billing service. Please try again.");
      } else if (err instanceof Error) {
        setCheckoutError(err.message);
      } else {
        setCheckoutError("Failed to start checkout.");
      }
      setCheckoutPlanLoading(null);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="text-center">
        <h1 className="font-heading text-4xl font-bold">
          Simple, credit-based pricing
        </h1>
        <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
          Pay for what you use. Each credit processes one video up to 30 minutes.
        </p>
        {checkoutStatus === "success" && (
          <p className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-700 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-200">
            Checkout completed. Your latest credit balance is shown below.
          </p>
        )}
        {checkoutStatus === "cancel" && (
          <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-700 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
            Checkout was canceled. You can try again when ready.
          </p>
        )}
        {checkoutError && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400">
            {checkoutError}
          </p>
        )}
        {isSignedIn && billingSummary && (
          <div className="mt-4 rounded-xl border border-zinc-200 bg-surface-card px-4 py-3 text-left text-sm dark:border-zinc-800">
            <p className="font-medium text-zinc-900 dark:text-zinc-100">
              Credits available: {billingSummary.credits_balance}
            </p>
            <p className="mt-1 text-zinc-600 dark:text-zinc-400">
              {billingSummary.has_unlimited_access
                ? "Unlimited access enabled."
                : `Free trial remaining: ${billingSummary.free_videos_remaining}/${billingSummary.free_videos_limit}`}
            </p>
          </div>
        )}
        {isSignedIn && billingSummaryError && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400">
            {billingSummaryError}
          </p>
        )}
      </div>

      <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-3">
        {tiers.map((tier) => {
          const paidPlan = tier.id === "free" ? null : tier.id;
          const isCheckoutLoading = paidPlan !== null && checkoutPlanLoading === paidPlan;

          if (isSignedIn && paidPlan) {
            return (
              <PricingCard
                key={tier.name}
                name={tier.name}
                price={tier.price}
                description={tier.description}
                features={tier.features}
                highlighted={tier.highlighted}
                onCtaClick={() => startCheckout(paidPlan)}
                ctaDisabled={checkoutPlanLoading !== null}
                ctaLabel={ctaLabel({
                  isSignedIn,
                  paidPlan,
                  isCheckoutLoading,
                })}
              />
            );
          }

          return (
            <PricingCard
              key={tier.name}
              name={tier.name}
              price={tier.price}
              description={tier.description}
              features={tier.features}
              highlighted={tier.highlighted}
              ctaHref={tier.ctaHref}
              ctaLabel={ctaLabel({
                isSignedIn,
                paidPlan,
                isCheckoutLoading,
              })}
            />
          );
        })}
      </div>
    </div>
  );
}
