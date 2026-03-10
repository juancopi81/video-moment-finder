export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ApiErrorPayload = {
  code?: string;
  detail: string;
};

export async function parseApiErrorPayload(
  response: Response,
  fallback: string
): Promise<ApiErrorPayload> {
  const data = await response.json().catch(() => null);

  if (Array.isArray(data?.detail)) {
    return { detail: data.detail[0]?.msg || fallback };
  }

  if (typeof data?.detail === "object" && data?.detail !== null) {
    return {
      code: typeof data.detail.code === "string" ? data.detail.code : undefined,
      detail:
        typeof data.detail.message === "string" ? data.detail.message : fallback,
    };
  }

  if (typeof data?.detail === "string") {
    return { detail: data.detail };
  }

  if (response.status === 401) {
    return { detail: "Please sign in to continue." };
  }

  return { detail: fallback };
}

export async function parseApiError(
  response: Response,
  fallback: string
): Promise<string> {
  const payload = await parseApiErrorPayload(response, fallback);
  return payload.detail;
}
