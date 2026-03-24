"use client";

import { useEffect, useRef } from "react";
import { useAuth } from "@clerk/nextjs";
import { usePostHog } from "posthog-js/react";

export function PostHogIdentify() {
  const { userId, isSignedIn, isLoaded } = useAuth();
  const posthog = usePostHog();
  const wasSignedIn = useRef(false);
  const identifiedUserId = useRef<string | null>(null);

  useEffect(() => {
    if (!posthog || !isLoaded) return;

    if (isSignedIn && userId) {
      if (identifiedUserId.current !== userId) {
        posthog.identify(userId);
        identifiedUserId.current = userId;
      }
      wasSignedIn.current = true;
    } else if (wasSignedIn.current) {
      posthog.reset();
      wasSignedIn.current = false;
      identifiedUserId.current = null;
    }
  }, [posthog, isLoaded, isSignedIn, userId]);

  return null;
}
