"use client";

import { Analytics, type BeforeSendEvent } from "@vercel/analytics/next";
import { sanitizeAnalyticsUrl } from "@/lib/sanitize-url";

function beforeSend(event: BeforeSendEvent): BeforeSendEvent | null {
  try {
    if (typeof window !== "undefined" && localStorage.getItem("vmf_internal") === "1") {
      return null;
    }
    return {
      ...event,
      url: sanitizeAnalyticsUrl(event.url),
    };
  } catch {
    return event;
  }
}

export function VercelAnalytics() {
  return <Analytics beforeSend={beforeSend} />;
}
