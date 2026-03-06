import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Pricing",
  description: "Simple credit-based pricing. Start free, upgrade as you need.",
  alternates: { canonical: "/pricing" },
};

export default function PricingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
