"use client";

import { useEffect, useRef } from "react";
import { useAuth } from "@clerk/nextjs";
import { usePostHog } from "posthog-js/react";

export function PostHogIdentify() {
  const { userId, isSignedIn, isLoaded } = useAuth();
  const posthog = usePostHog();
  const wasSignedIn = useRef(false);

  useEffect(() => {
    if (!posthog || !isLoaded) return;

    if (isSignedIn && userId) {
      posthog.identify(userId);
      wasSignedIn.current = true;
    } else if (wasSignedIn.current) {
      // Real logout transition — clear identified state.
      posthog.reset();
      wasSignedIn.current = false;
    }
    // Initial anonymous mount: neither identify nor reset.
  }, [posthog, isLoaded, isSignedIn, userId]);

  return null;
}
