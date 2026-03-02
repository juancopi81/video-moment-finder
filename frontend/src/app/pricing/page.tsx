"use client";

import { useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { PricingCard } from "@/components/pricing-card";
import { API_URL, parseApiError } from "@/lib/api";

type BillingPlan = "starter" | "pro";
type TierId = "free" | BillingPlan;

const tiers = [
  {
    id: "free" as TierId,
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
    id: "starter" as TierId,
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
    id: "pro" as TierId,
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

export default function PricingPage() {
  const { userId, getToken, isLoaded } = useAuth();
  const [checkoutPlanLoading, setCheckoutPlanLoading] = useState<BillingPlan | null>(
    null,
  );
  const [checkoutError, setCheckoutError] = useState<string | null>(null);
  const isSignedIn = !!userId;

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
        {checkoutError && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400">
            {checkoutError}
          </p>
        )}
      </div>

      <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-3">
        {tiers.map((tier) => {
          const paidPlan = tier.id === "free" ? null : tier.id;
          const isPaidTier = paidPlan !== null;
          const isCheckoutLoading = paidPlan !== null && checkoutPlanLoading === paidPlan;
          return (
            <PricingCard
              key={tier.name}
              {...tier}
              ctaHref={!isSignedIn || !isPaidTier ? tier.ctaHref : undefined}
              onCtaClick={
                isSignedIn && paidPlan
                  ? () => startCheckout(paidPlan)
                  : undefined
              }
              ctaDisabled={checkoutPlanLoading !== null}
              ctaLabel={
                isSignedIn
                  ? paidPlan
                    ? isCheckoutLoading
                      ? "Opening checkout..."
                      : "Buy credits"
                    : "Process a video"
                  : "Get started"
              }
            />
          );
        })}
      </div>
    </div>
  );
}
