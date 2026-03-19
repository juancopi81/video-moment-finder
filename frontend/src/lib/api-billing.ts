import { API_URL, parseApiError } from "@/lib/api";

export type ApiBillingSummary = {
  api_units_balance: number;
  unit_cost_index_video: number;
  unit_cost_text_query: number;
  approx_videos: number;
  approx_queries: number;
};

export type ApiUsageEvent = {
  id: string;
  api_key_id: string | null;
  event_type: string;
  units: number;
  video_id: string | null;
  created_at: string | null;
};

export async function fetchApiBillingSummary(
  token: string,
): Promise<ApiBillingSummary> {
  const response = await fetch(`${API_URL}/api/v1/billing/summary`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!response.ok) {
    throw new Error(
      await parseApiError(response, "Failed to load API billing summary."),
    );
  }
  return (await response.json()) as ApiBillingSummary;
}

export async function fetchApiUsageEvents(
  token: string,
  params?: { api_key_id?: string; limit?: number },
): Promise<ApiUsageEvent[]> {
  const searchParams = new URLSearchParams();
  if (params?.api_key_id) {
    searchParams.set("api_key_id", params.api_key_id);
  }
  if (params?.limit) {
    searchParams.set("limit", String(params.limit));
  }
  const qs = searchParams.toString();
  const url = `${API_URL}/api/v1/billing/usage${qs ? `?${qs}` : ""}`;
  const response = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!response.ok) {
    throw new Error(
      await parseApiError(response, "Failed to load API usage events."),
    );
  }
  return (await response.json()) as ApiUsageEvent[];
}

export async function startApiCheckout(token: string): Promise<string> {
  const response = await fetch(`${API_URL}/api/v1/billing/checkout`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ plan: "developer" }),
  });
  if (!response.ok) {
    throw new Error(
      await parseApiError(
        response,
        "Failed to start checkout. Please try again.",
      ),
    );
  }
  const payload = await response.json();
  if (typeof payload.checkout_url !== "string" || !payload.checkout_url) {
    throw new Error("Checkout URL missing in response.");
  }
  return payload.checkout_url as string;
}
