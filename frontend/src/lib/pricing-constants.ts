export type BillingPlan = "starter" | "pro";
export type TierId = "free" | BillingPlan;

export type Tier = {
  id: TierId;
  name: string;
  price: string;
  description: string;
  features: string[];
  highlighted?: boolean;
  ctaHref: string;
};

export const tiers: Tier[] = [
  {
    id: "free",
    name: "Free Trial",
    price: "$0",
    description: "Try it out with one video",
    features: [
      "1 video credit",
      "Up to 90-minute videos",
      "Text & image moment search",
      "Thumbnail previews",
    ],
    ctaHref: "/",
  },
  {
    id: "starter",
    name: "Starter",
    price: "$5",
    description: "5 video credits",
    features: [
      "5 video credits",
      "Up to 90-minute videos",
      "Text & image moment search",
      "Thumbnail previews",
      "Direct upload (recommended)",
      "YouTube import (not guaranteed)",
    ],
    highlighted: true,
    ctaHref: "/",
  },
  {
    id: "pro",
    name: "Pro",
    price: "$15",
    description: "20 video credits",
    features: [
      "20 video credits",
      "Up to 90-minute videos",
      "Text & image moment search",
      "Thumbnail previews",
      "Direct upload (recommended)",
      "YouTube import (not guaranteed)",
    ],
    ctaHref: "/",
  },
];

export type ApiCard = {
  name: string;
  price: string;
  description: string;
  features: string[];
};

export const API_CARD: ApiCard = {
  name: "Developer Pack",
  price: "$20",
  description: "10,000 API units",
  features: [
    "500 units per indexed video",
    "1 unit per text query (launch pricing)",
    "Per-key usage dashboard",
    "CLI access",
  ],
};
