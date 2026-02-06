import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video Moment Finder",
  description: "Semantic video frame search. Find moments using text or images.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
