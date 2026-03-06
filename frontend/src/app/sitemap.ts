import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const base = "https://videomomentfinder.com";
  return [
    { url: base, changeFrequency: "weekly", priority: 1.0 },
    { url: `${base}/pricing`, changeFrequency: "weekly", priority: 0.8 },
    { url: `${base}/support`, changeFrequency: "monthly", priority: 0.5 },
    { url: `${base}/privacy`, changeFrequency: "yearly", priority: 0.3 },
    { url: `${base}/terms`, changeFrequency: "yearly", priority: 0.3 },
  ];
}
