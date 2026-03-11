"use client";

import { Analytics, type BeforeSendEvent } from "@vercel/analytics/next";

const VIDEO_DETAIL_PATH = /^\/video\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const DEFAULT_ANALYTICS_ORIGIN = "https://videomomentfinder.com";

function sanitizeAnalyticsUrl(rawUrl: string): string {
  const baseUrl = typeof window === "undefined" ? DEFAULT_ANALYTICS_ORIGIN : window.location.origin;
  const url = new URL(rawUrl, baseUrl);

  url.search = "";
  url.hash = "";

  if (VIDEO_DETAIL_PATH.test(url.pathname)) {
    url.pathname = "/video/[id]";
  }

  return url.toString();
}

function beforeSend(event: BeforeSendEvent): BeforeSendEvent {
  try {
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
