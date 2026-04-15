import type { Metadata } from "next";
import { Outfit, Source_Sans_3 } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import { Header } from "@/components/header";
import { Footer } from "@/components/footer";
import { VercelAnalytics } from "@/components/vercel-analytics";
import { PHProvider } from "@/components/posthog-provider";
import { PostHogIdentify } from "@/components/posthog-identify";
import {
  OG_IMAGE_HEIGHT,
  OG_IMAGE_PATH,
  OG_IMAGE_WIDTH,
  SITE_NAME,
  SITE_TAGLINE,
  SITE_URL,
} from "@/lib/site-config";
import {
  organizationSchema,
  softwareApplicationSchema,
} from "@/lib/schema";
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
  metadataBase: new URL(SITE_URL),
  title: {
    default: SITE_NAME,
    template: `%s | ${SITE_NAME}`,
  },
  description: SITE_TAGLINE,
  openGraph: {
    type: "website",
    siteName: SITE_NAME,
    images: [
      {
        url: OG_IMAGE_PATH,
        width: OG_IMAGE_WIDTH,
        height: OG_IMAGE_HEIGHT,
        alt: SITE_NAME,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    images: [OG_IMAGE_PATH],
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
          <script type="application/ld+json">
            {JSON.stringify(organizationSchema)}
          </script>
          <script type="application/ld+json">
            {JSON.stringify(softwareApplicationSchema)}
          </script>
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
