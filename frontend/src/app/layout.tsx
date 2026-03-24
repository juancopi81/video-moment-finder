import type { Metadata } from "next";
import { Outfit, Source_Sans_3 } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import { Header } from "@/components/header";
import { Footer } from "@/components/footer";
import { VercelAnalytics } from "@/components/vercel-analytics";
import { PHProvider } from "@/components/posthog-provider";
import { PostHogIdentify } from "@/components/posthog-identify";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-heading",
  display: "swap",
});

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://videomomentfinder.com"),
  title: {
    default: "Video Moment Finder",
    template: "%s | Video Moment Finder",
  },
  description: "Semantic video frame search. Find moments using text or images.",
  openGraph: {
    type: "website",
    siteName: "Video Moment Finder",
  },
  twitter: {
    card: "summary",
  },
};

const clerkPublishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider
      publishableKey={clerkPublishableKey}
      __internal_bypassMissingPublishableKey={!clerkPublishableKey}
    >
      <html lang="en">
        <body
          className={`${sourceSans.className} ${outfit.variable} ${sourceSans.variable} flex min-h-screen flex-col antialiased`}
        >
          <PHProvider>
            <Header />
            <main className="flex-1">{children}</main>
            <Footer />
            <PostHogIdentify />
            <VercelAnalytics />
          </PHProvider>
        </body>
      </html>
    </ClerkProvider>
  );
}
