import { API_URL, parseApiError } from "@/lib/api";

export type BillingSummary = {
  credits_balance: number;
  free_videos_limit: number;
  free_videos_used: number;
  free_videos_remaining: number;
  has_unlimited_access: boolean;
};

export async function fetchBillingSummary(token: string): Promise<BillingSummary> {
  const response = await fetch(`${API_URL}/api/v1/billing/credits/summary`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!response.ok) {
    throw new Error(await parseApiError(response, "Failed to load billing summary."));
  }

  return (await response.json()) as BillingSummary;
}
