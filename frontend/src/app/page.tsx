import type { Metadata } from "next";
import HomeContent from "./home-content";

export const metadata: Metadata = {
  title: "Search Your Video Archive",
  description:
    "Upload one lesson, webinar, workshop, or demo you own. Find the exact teaching, demo, or explanation moment by description or example image.",
  alternates: { canonical: "/" },
};

export default function Home() {
  return <HomeContent />;
}
