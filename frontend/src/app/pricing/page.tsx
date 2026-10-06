import type { Metadata } from "next";
import { tiers } from "@/lib/pricing-constants";
import { ApiPricingCardClient } from "./api-pricing-card-client";
import { PricingInteractions } from "./pricing-interactions";
import { PricingTiers } from "./pricing-tiers";

export const metadata: Metadata = {
  title: "Pricing",
  description:
    "Website video credits and API units for Video Moment Finder. Check your free allowance after sign-in. Starter: $5 for 5 website credits; Pro: $15 for 20.",
  alternates: { canonical: "/pricing" },
};

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="text-center">
        <h1 className="font-heading text-4xl font-bold">
          Simple, credit-based pricing
        </h1>
        <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
          Pay for what you use. Each website video credit processes one video up to 90
          minutes.
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
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Your signed-in summary shows the free allowance available to your account.
          An enrolled API trial shares its unit balance with website indexing;
          website video credits remain a separate balance.
        </p>

        <PricingInteractions />
      </div>

      <PricingTiers tiers={tiers} />

      <div className="mt-16 border-t border-zinc-200 pt-12 dark:border-zinc-800">
        <h2 className="text-center font-heading text-2xl font-bold">
          API Access
        </h2>
        <p className="mt-2 text-center text-sm text-zinc-600 dark:text-zinc-400">
          Build on top of Video Moment Finder with our REST API and CLI.
        </p>
        <div className="mt-8 mx-auto max-w-sm">
          <ApiPricingCardClient />
        </div>
      </div>
    </div>
  );
}
