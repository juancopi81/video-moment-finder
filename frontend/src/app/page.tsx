import type { Metadata } from "next";
import HomeContent from "./home-content";

export const metadata: Metadata = {
  alternates: { canonical: "/" },
};

export default function Home() {
  return <HomeContent />;
}
