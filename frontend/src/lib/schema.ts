import {
  GITHUB_URL,
  OG_IMAGE_HEIGHT,
  OG_IMAGE_PATH,
  OG_IMAGE_WIDTH,
  SITE_DESCRIPTION,
  SITE_NAME,
  SITE_TAGLINE,
  SITE_URL,
  SUPPORT_EMAIL,
} from "./site-config";

type Faq = { question: string; answer: string };

export const organizationSchema = {
  "@context": "https://schema.org",
  "@type": "Organization",
  "@id": `${SITE_URL}/#organization`,
  name: SITE_NAME,
  url: `${SITE_URL}/`,
  logo: {
    "@type": "ImageObject",
    url: `${SITE_URL}/icon-512.png`,
    width: 512,
    height: 512,
  },
  image: {
    "@type": "ImageObject",
    url: `${SITE_URL}${OG_IMAGE_PATH}`,
    width: OG_IMAGE_WIDTH,
    height: OG_IMAGE_HEIGHT,
  },
  description: SITE_TAGLINE,
  sameAs: [GITHUB_URL],
  contactPoint: {
    "@type": "ContactPoint",
    email: SUPPORT_EMAIL,
    contactType: "customer support",
    availableLanguage: "en",
  },
};

export const softwareApplicationSchema = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "@id": `${SITE_URL}/#software`,
  name: SITE_NAME,
  url: `${SITE_URL}/`,
  description: SITE_DESCRIPTION,
  applicationCategory: "MultimediaApplication",
  applicationSubCategory: "Video Search",
  operatingSystem: "Any",
  browserRequirements: "Requires a modern browser with JavaScript enabled",
  featureList: [
    "Semantic text-to-video search",
    "Image-to-video similarity search",
    "Multimodal embeddings (Qwen3-VL)",
    "Direct video upload (up to 30 minutes)",
    "YouTube URL import for owned videos",
    "Row-level security for user data",
  ],
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
    description:
      "Free tier includes 1 credit (1 video up to 30 minutes). Paid credit packs available; credits do not expire.",
  },
  license: "https://www.gnu.org/licenses/agpl-3.0.html",
  codeRepository: GITHUB_URL,
  publisher: { "@id": `${SITE_URL}/#organization` },
};

export const personSchema = {
  "@context": "https://schema.org",
  "@type": "Person",
  "@id": `${SITE_URL}/about#person`,
  name: "Juan Piñeros",
  url: `${SITE_URL}/about`,
  image: "https://github.com/juancopi81.png",
  jobTitle: "Creator, Video Moment Finder",
  worksFor: { "@id": `${SITE_URL}/#organization` },
  sameAs: [
    "https://github.com/juancopi81",
    "https://www.linkedin.com/in/REPLACE_ME",
    "https://x.com/REPLACE_ME",
  ],
};

export function faqPageSchema(faqs: ReadonlyArray<Faq>) {
  return {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    url: `${SITE_URL}/support`,
    mainEntity: faqs.map((faq) => ({
      "@type": "Question",
      name: faq.question,
      acceptedAnswer: {
        "@type": "Answer",
        text: faq.answer,
      },
    })),
  };
}
