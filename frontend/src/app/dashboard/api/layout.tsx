import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "API Access",
  description: "Manage your API keys and usage",
  alternates: { canonical: "/dashboard/api" },
};

export default function ApiDashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
