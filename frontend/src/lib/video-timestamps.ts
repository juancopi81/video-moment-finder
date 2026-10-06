export function parseVideoTimestamp(value: string | string[] | undefined): number | null {
  if (typeof value !== "string" || !/^\d+(?:\.\d+)?$/.test(value)) return null;
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds <= Number.MAX_SAFE_INTEGER ? seconds : null;
}

export function boundedVideoTimestamp(seconds: number | null, duration: number): number | null {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return null;
  if (!Number.isFinite(duration) || duration <= 0) return null;
  return Math.min(seconds, duration);
}

export function buildTimestampUrl(baseUrl: string, seconds: number): string | null {
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  try {
    const url = new URL(baseUrl);
    if (url.protocol !== "https:" && url.protocol !== "http:") return null;
    url.searchParams.set("t", Math.floor(seconds).toString());
    return url.toString();
  } catch {
    return null;
  }
}
