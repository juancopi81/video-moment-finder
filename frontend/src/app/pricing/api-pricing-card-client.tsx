"use client";

import { useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { usePostHog } from "posthog-js/react";
import { PricingCard } from "@/components/pricing-card";
import { API_CARD } from "@/lib/pricing-constants";
import { startApiCheckout } from "@/lib/api-billing";

export function ApiPricingCardClient() {
  const { userId, getToken, isLoaded } = useAuth();
  const isSignedIn = !!userId;
  const posthog = usePostHog();
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
      posthog?.capture("checkout_started_client", { plan: "developer" });
      window.location.assign(url);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start checkout.",
      );
      setLoading(false);
    }
  }

  if (!isLoaded) {
    return (
      <PricingCard
        {...API_CARD}
        onCtaClick={() => {}}
        ctaDisabled
        ctaLabel="Checking account..."
      />
    );
  }

  if (isSignedIn) {
    return (
      <>
        <PricingCard
          {...API_CARD}
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
      {...API_CARD}
      ctaHref="/developers"
      ctaLabel="Learn more"
    />
  );
}
