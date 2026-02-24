import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Pricing - Video Moment Finder",
  description: "Simple credit-based pricing. Start free, upgrade as you need.",
};

export default function PricingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
