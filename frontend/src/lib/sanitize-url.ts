const VIDEO_DETAIL_PATH = /^\/video\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const DEFAULT_ANALYTICS_ORIGIN = "https://videomomentfinder.com";

/** Mask video UUIDs in a pathname. */
export function sanitizePathname(pathname: string): string {
  return VIDEO_DETAIL_PATH.test(pathname) ? "/video/[id]" : pathname;
}

/** Strip query params, hash, and mask video UUIDs from a full URL. */
export function sanitizeAnalyticsUrl(rawUrl: string): string {
  const baseUrl = typeof window === "undefined" ? DEFAULT_ANALYTICS_ORIGIN : window.location.origin;
  const url = new URL(rawUrl, baseUrl);

  url.search = "";
  url.hash = "";
  url.pathname = sanitizePathname(url.pathname);

  return url.toString();
}
