import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Developers",
  description: "API and CLI for Video Moment Finder",
  alternates: { canonical: "/developers" },
};

export default function DevelopersLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
