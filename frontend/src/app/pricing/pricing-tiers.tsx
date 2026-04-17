"use client";

import { useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { usePostHog } from "posthog-js/react";
import { PricingCard } from "@/components/pricing-card";
import { API_URL, parseApiError } from "@/lib/api";
import type { BillingPlan, Tier } from "@/lib/pricing-constants";

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

export function PricingTiers({ tiers }: { tiers: ReadonlyArray<Tier> }) {
  const { userId, getToken, isLoaded } = useAuth();
  const posthog = usePostHog();
  const [checkoutPlanLoading, setCheckoutPlanLoading] =
    useState<BillingPlan | null>(null);
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
      const response = await fetch(
        `${API_URL}/api/v1/billing/credits/checkout`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ plan }),
        },
      );
      if (!response.ok) {
        throw new Error(
          await parseApiError(
            response,
            "Failed to start checkout. Please try again.",
          ),
        );
      }

      const payload = await response.json();
      if (
        typeof payload.checkout_url !== "string" ||
        !payload.checkout_url
      ) {
        throw new Error("Checkout URL missing in response.");
      }

      posthog?.capture("checkout_started_client", { plan });
      window.location.assign(payload.checkout_url);
    } catch (err) {
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setCheckoutError(
          "Cannot connect to billing service. Please try again.",
        );
      } else if (err instanceof Error) {
        setCheckoutError(err.message);
      } else {
        setCheckoutError("Failed to start checkout.");
      }
      setCheckoutPlanLoading(null);
    }
  }

  return (
    <>
      {checkoutError && (
        <p className="mt-4 text-center text-sm text-red-600 dark:text-red-400">
          {checkoutError}
        </p>
      )}
      <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-3">
        {tiers.map((tier) => {
          const paidPlan = tier.id === "free" ? null : tier.id;
          const isCheckoutLoading =
            paidPlan !== null && checkoutPlanLoading === paidPlan;

          if (!isLoaded && paidPlan) {
            return (
              <PricingCard
                key={tier.name}
                name={tier.name}
                price={tier.price}
                description={tier.description}
                features={tier.features}
                highlighted={tier.highlighted}
                onCtaClick={() => {}}
                ctaDisabled
                ctaLabel="Checking account..."
              />
            );
          }

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
    </>
  );
}
