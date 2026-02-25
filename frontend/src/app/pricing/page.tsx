"use client";

import { useAuth } from "@clerk/nextjs";
import { PricingCard } from "@/components/pricing-card";

const tiers = [
  {
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
  const { userId } = useAuth();
  const isSignedIn = !!userId;

  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="text-center">
        <h1 className="font-heading text-4xl font-bold">
          Simple, credit-based pricing
        </h1>
        <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
          Pay for what you use. Each credit processes one video up to 30 minutes.
        </p>
        {isSignedIn && (
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
            Paid checkout is being finalized. Join the waitlist for early access.
          </p>
        )}
      </div>

      <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-3">
        {tiers.map((tier) => (
          <PricingCard
            key={tier.name}
            {...tier}
            ctaHref={
              isSignedIn && tier.name !== "Free Trial"
                ? "/support"
                : tier.ctaHref
            }
            ctaLabel={
              isSignedIn
                ? tier.name === "Free Trial"
                  ? "Process a video"
                  : "Join waitlist"
                : "Get started"
            }
          />
        ))}
      </div>
    </div>
  );
}
