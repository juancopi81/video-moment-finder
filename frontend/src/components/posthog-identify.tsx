"use client";

import { useEffect } from "react";
import { useAuth } from "@clerk/nextjs";
import { usePostHog } from "posthog-js/react";

export function PostHogIdentify() {
  const { userId, isSignedIn } = useAuth();
  const posthog = usePostHog();

  useEffect(() => {
    if (!posthog || !isSignedIn || !userId) return;
    posthog.identify(userId);
  }, [posthog, isSignedIn, userId]);

  return null;
}
