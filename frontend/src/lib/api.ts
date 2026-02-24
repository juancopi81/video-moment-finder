export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function parseApiError(
  response: Response,
  fallback: string
): Promise<string> {
  const data = await response.json().catch(() => null);
  if (Array.isArray(data?.detail)) {
    return data.detail[0]?.msg || fallback;
  }
  if (typeof data?.detail === "string") {
    return data.detail;
  }
  if (response.status === 401) {
    return "Please sign in to continue.";
  }
  return fallback;
}
