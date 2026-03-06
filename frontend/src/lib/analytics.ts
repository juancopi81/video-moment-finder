import { API_URL } from "@/lib/api";

export function trackEvent(
  eventName: string,
  metadata?: Record<string, unknown>,
  token?: string | null
): void {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  fetch(`${API_URL}/analytics/event`, {
    method: "POST",
    headers,
    body: JSON.stringify({ event_name: eventName, metadata }),
  }).catch(() => {});
}
