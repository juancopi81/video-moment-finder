"use client";

import posthog from "posthog-js";
import { PostHogProvider } from "posthog-js/react";
import { type ReactNode } from "react";
import { sanitizeAnalyticsUrl, sanitizePathname } from "@/lib/sanitize-url";

const posthogKey = process.env.NEXT_PUBLIC_POSTHOG_KEY;
const posthogHost = process.env.NEXT_PUBLIC_POSTHOG_HOST || "https://us.i.posthog.com";

function isInternalTraffic(): boolean {
  try {
    return localStorage.getItem("vmf_internal") === "1";
  } catch {
    return false;
  }
}

if (typeof window !== "undefined" && posthogKey) {
  posthog.init(posthogKey, {
    api_host: posthogHost,
    person_profiles: "identified_only",
    capture_pageview: true,
    capture_pageleave: true,
    before_send(event) {
      if (isInternalTraffic()) return null;
      return event;
    },
    sanitize_properties(properties) {
      if (properties["$current_url"]) {
        properties["$current_url"] = sanitizeAnalyticsUrl(properties["$current_url"]);
      }
      if (properties["$pathname"]) {
        properties["$pathname"] = sanitizePathname(properties["$pathname"]);
      }
      return properties;
    },
  });
}

export function PHProvider({ children }: { children: ReactNode }) {
  if (!posthogKey) return <>{children}</>;
  return <PostHogProvider client={posthog}>{children}</PostHogProvider>;
}
