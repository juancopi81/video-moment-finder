"use client";

import { useEffect } from "react";
import { useAuth } from "@clerk/nextjs";
import { usePostHog } from "posthog-js/react";

export function PostHogIdentify() {
  const { userId, isSignedIn } = useAuth();
  const posthog = usePostHog();

  useEffect(() => {
    if (!posthog) return;
    if (isSignedIn && userId) {
      posthog.identify(userId);
    } else {
      posthog.reset();
    }
  }, [posthog, isSignedIn, userId]);

  return null;
}
