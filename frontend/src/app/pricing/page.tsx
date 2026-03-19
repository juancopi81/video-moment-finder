"use client";

import { Suspense, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { useSearchParams } from "next/navigation";
import { BillingSummaryCard } from "@/components/billing-summary-card";
import { CheckoutStatusBanner } from "@/components/checkout-status-banner";
import { PricingCard } from "@/components/pricing-card";
import { useBillingSummary } from "@/hooks/useBillingSummary";
import { API_URL, parseApiError } from "@/lib/api";
import { startApiCheckout } from "@/lib/api-billing";

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
      "Text & image moment search",
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
      "Text & image moment search",
      "Thumbnail previews",
      "Direct upload (recommended)",
      "YouTube import (not guaranteed)",
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
      "Text & image moment search",
      "Thumbnail previews",
      "Direct upload (recommended)",
      "YouTube import (not guaranteed)",
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

function ApiPricingCard({
  isSignedIn,
  getToken,
}: {
  isSignedIn: boolean;
  getToken: () => Promise<string | null>;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCheckout() {
    setError(null);
    const token = await getToken();
    if (!token) {
      setError("Please sign in to continue.");
      return;
    }
    setLoading(true);
    try {
      const url = await startApiCheckout(token);
      window.location.assign(url);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start checkout.",
      );
      setLoading(false);
    }
  }

  if (isSignedIn) {
    return (
      <>
        <PricingCard
          name="Developer Pack"
          price="$20"
          description="10,000 API units"
          features={[
            "500 units per indexed video",
            "1 unit per text query (launch pricing)",
            "Per-key usage dashboard",
            "CLI access",
          ]}
          onCtaClick={handleCheckout}
          ctaDisabled={loading}
          ctaLabel={loading ? "Opening checkout..." : "Buy Developer Pack"}
        />
        {error && (
          <p className="mt-2 text-center text-sm text-red-600 dark:text-red-400">
            {error}
          </p>
        )}
      </>
    );
  }

  return (
    <PricingCard
      name="Developer Pack"
      price="$20"
      description="10,000 API units"
      features={[
        "500 units per indexed video",
        "1 unit per text query (launch pricing)",
        "Per-key usage dashboard",
        "CLI access",
      ]}
      ctaHref="/developers"
      ctaLabel="Learn more"
    />
  );
}

function PricingPageContent() {
  const { userId, getToken, isLoaded } = useAuth();
  const searchParams = useSearchParams();
  const [checkoutPlanLoading, setCheckoutPlanLoading] = useState<BillingPlan | null>(
    null,
  );
  const [checkoutError, setCheckoutError] = useState<string | null>(null);
  const isSignedIn = !!userId;
  const checkoutStatus = searchParams.get("checkout");
  const { billingSummary, billingSummaryError, isRefreshingBalance } =
    useBillingSummary({ checkoutStatus, enableCheckoutPolling: true });

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
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          Direct upload is the reliable path. You can also use YouTube import
          for videos you own or are authorized to use, but it may not always
          work.
        </p>
        <p className="mt-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
          Beta pricing: current introductory rates may change as the product
          matures.
        </p>
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
        {checkoutError && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400">
            {checkoutError}
          </p>
        )}
        {isSignedIn && billingSummary && (
          <BillingSummaryCard summary={billingSummary} className="mt-4 text-left" />
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

      <div className="mt-16 border-t border-zinc-200 pt-12 dark:border-zinc-800">
        <h2 className="text-center font-heading text-2xl font-bold">
          API Access
        </h2>
        <p className="mt-2 text-center text-sm text-zinc-600 dark:text-zinc-400">
          Build on top of Video Moment Finder with our REST API and CLI.
        </p>
        <div className="mt-8 mx-auto max-w-sm">
          <ApiPricingCard
            isSignedIn={isSignedIn}
            getToken={getToken}
          />
        </div>
      </div>
    </div>
  );
}

export default function PricingPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-5xl px-4 py-16 text-center">
          <p className="text-zinc-600 dark:text-zinc-400">Loading pricing...</p>
        </div>
      }
    >
      <PricingPageContent />
    </Suspense>
  );
}
