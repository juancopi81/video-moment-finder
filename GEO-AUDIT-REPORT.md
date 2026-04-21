# GEO Audit Report: Video Moment Finder

**Audit Date:** 2026-04-13
**URL:** https://www.videomomentfinder.com
**Business Type:** SaaS — AI-powered semantic video moment search (indie, open-source under AGPL-3.0)
**Pages Analyzed:** 5 canonical (`/`, `/pricing`, `/support`, `/privacy`, `/terms`) + `/developers`, plus the agent-facing surfaces `/llms.txt`, `/skill.md`, `/connectors/claude`, `/sitemap.xml`, `/robots.txt`.

---

## Executive Summary

**Overall GEO Score: 37/100 (Critical)**

Video Moment Finder has an unusually mature *agent-integration* layer for a pre-launch indie SaaS — `llms.txt`, `skill.md`, a published MCP connector, OpenAPI + Swagger UI, and permissive AI-crawler access are all in place and are roughly two years ahead of the typical SaaS site. The score is dragged into the Critical band by the opposite problem: essentially zero *AI-citation* readiness. There is no JSON-LD schema anywhere on the site, near-zero brand authority on the platforms that LLMs use for entity recognition (Reddit, HN, Wikipedia, YouTube, Product Hunt), the pricing page is client-rendered behind a "Loading pricing…" fallback that is invisible to AI crawlers, and there is no author/about page, so E-E-A-T signals are thin. Six or seven targeted fixes — none of which require new content strategy — will move the score into the mid-60s within a week.

### Score Breakdown

| Category | Score | Weight | Weighted |
|---|---|---|---|
| AI Citability | 55/100 | 25% | 13.75 |
| Brand Authority | 8/100 | 20% | 1.60 |
| Content E-E-A-T | 38/100 | 20% | 7.60 |
| Technical GEO (incl. crawler access, llms.txt) | 74/100 | 15% | 11.10 |
| Schema & Structured Data | 3/100 | 10% | 0.30 |
| Platform Optimization | 24/100 | 10% | 2.40 |
| **Overall GEO Score** | | | **36.75 → 37/100** |

Scores above are the baseline snapshot from 2026-04-13 and are deliberately preserved as historical reference. See **Progress Update** below for what has shipped since.

---

## Progress Update — 2026-04-15

Week 1 of the 30-Day Action Plan shipped in PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72) on branch `feat/geo-improvements`. Six of the seven Quick Wins are addressed; Quick Win #7 (Show HN / Product Hunt / YouTube demo) is a Week 3 brand-authority launch tracked separately.

**Shipped**

- **Critical #1 — Zero JSON-LD.** `Organization` + `SoftwareApplication` JSON-LD injected in the root layout (present on every page). `FAQPage` JSON-LD on `/support`, built from the existing FAQ array so schema cannot drift from visible content. Centralized in `frontend/src/lib/schema.ts`.
- **Critical #2 — `/pricing` client-only.** `/pricing` is now fully prerendered (`○ Static`). Every tier name, price, and feature appears in the initial HTML that crawlers see. The signed-in UX is unchanged: clicking "Buy credits" on a paid tier still opens Stripe.
- **Critical #3 — www ↔ apex canonical-flip.** `metadataBase`, `sitemap.ts`, `robots.ts`, and `public/llms.txt` are all aligned to `https://www.videomomentfinder.com` to match the existing Vercel apex→www edge redirect.
- **High #4 — FAQPage schema.** Covered by the Critical #1 rollout.
- **High #6 — Security headers.** Four added in `next.config.ts`: `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy: camera=(), microphone=(), geolocation=()`, `X-Frame-Options: DENY`. Strict CSP is intentionally deferred (see below).
- **High #7 — Sitemap incomplete.** Sitemap expanded from 5 → 9 URLs (added `/developers`, `/connectors/claude`, `/skill.md`, `/llms.txt`). Every entry now carries `<lastmod>`.
- **Medium #9 — `llms.txt` under-described.** Link annotations expanded from one-word labels to short factual sentences; `/connectors/claude` added under `## Links`. `/skill.md` was already listed.
- **Medium #11 — No `og:image` / `twitter:image`.** `og:image` set to the existing `preview-whiteboard-teaching.jpg` (1200×630); Twitter card upgraded from `summary` to `summary_large_image`.

**Deferred (intentional, with reasons)**

- **Strict Content-Security-Policy.** Clerk, Stripe, PostHog, and Vercel Analytics each need explicit source allowlists — a wrong CSP breaks auth in production. Plan: ship first as `Content-Security-Policy-Report-Only` to catch violations, then enforce.
- **`vercel.json` 301 formalization** of apex → www. The existing Vercel edge 307 already satisfies the canonical-alignment goal; tightening to 301 is low-value follow-up.
- **`llms-full.txt`.** Existing `llms.txt` + `skill.md` + SSR `/support` already surface the same content; adding a third variant is diminishing returns.

**Still open (Week 2+)**

- **High #8 — Brand footprint.** External/marketing (Show HN, Product Hunt, Reddit, YouTube demo, dev.to, LinkedIn). Week 3.
- **Medium #10** — per-bot `Allow` stanzas in `robots.txt` (Week 4).
- **Medium #13** — edge `X-Robots-Tag: noindex` on `/dashboard/*` and `/video/*` (Week 4).
- **Medium #14** — thicker SSR homepage copy outside the auth boundary (Week 2+).
- **Low #15–#18** — freshness signals, `WebSite` + `SearchAction`, `speakable` — optimize-when-possible.

**Impact estimate (not re-audited).** Week 1 alone should lift Schema & Structured Data (3 → ~75), Technical GEO (74 → ~85), and AI Citability (55 → ~65). A full re-audit is best run after `/about` lands in Week 2, so Brand Authority and E-E-A-T can be measured in the same pass.

---

## Progress Update — 2026-04-21

Week 2 is now closed out across two PRs.

**Shipped in PR [#73](https://github.com/juancopi81/video-moment-finder/pull/73)**

- **High #5 — `/about` page with `Person` schema.** Server-rendered creator page with bio, avatar, and `sameAs` links to GitHub, LinkedIn, and X. `Person` JSON-LD is linked to the Organization via `worksFor` `@id`, so the entity graph resolves. Linked from the footer on every page; added to `sitemap.ts`.

**Shipped on branch `feat/how-it-works-and-faq-dates`** (this change)

- **Medium #12 — Qwen3-VL buried in FAQ.** New server-rendered `/how-it-works` page covering the pipeline (upload → frame extraction → Qwen3-VL embeddings → vector store → search), an explicit "why Qwen3-VL" section, and an accuracy-tradeoffs section framing where the approach beats transcript search and where it doesn't. `TechArticle` JSON-LD with `author` pointing to the Person `@id` and `publisher` pointing to the Organization `@id`. Linked from the footer, added to `sitemap.ts` and `llms.txt`.
- **Low #15 (partial) — Freshness signals on `/support`.** Every FAQ entry now carries a `dateModified` field, rendered as a small "Updated 2026-04-13" timestamp under each answer and echoed into each `Answer` in the `FAQPage` JSON-LD. Future FAQ edits bump the per-entry date.

**Still deferred**

- **Strict Content-Security-Policy.** Unchanged from Week 1 — needs a Report-Only rollout with real prod monitoring for Clerk/Stripe/PostHog/Vercel Analytics, handled as a standalone PR after Week 3.
- **`vercel.json` 301.** Unchanged — the existing Vercel edge 307 still satisfies the canonical goal.
- **`llms-full.txt`.** Unchanged — diminishing returns over the existing `llms.txt` + `skill.md` + SSR surfaces.

**Next up (Week 3 — no code).** Brand-authority launch: Show HN, Product Hunt, r/SideProject + r/MachineLearning, 3-minute YouTube demo, dev.to post on the MCP connector, LinkedIn company page. `/about` and `/how-it-works` now exist as substantive pages to link to.

---

## Critical Issues (Fix Immediately)

1. **Zero JSON-LD structured data anywhere on the site.** All four audited pages (`/`, `/support`, `/developers`, `/pricing`) contain zero `application/ld+json` blocks, zero Microdata, zero RDFa. For a prerendered Next.js site this is a pure implementation gap. Highest-leverage fix in the entire audit — see the three ready-to-paste snippets in §6 below. **Affects: every AI platform that uses entity graphs (Google AIO, ChatGPT, Perplexity, Gemini, Bing Copilot).**

2. **`/pricing` is fully client-rendered with no SSR fallback.** Raw HTML contains a client-side-rendering bailout template and a "Loading pricing…" placeholder. GPTBot / ClaudeBot / PerplexityBot / Googlebot see no plan tiers, no prices, no credit-pack information. When a user asks an AI "How much does Video Moment Finder cost?", there is nothing to quote.

3. **www ↔ apex canonical-flip.** Both `www.videomomentfinder.com` and `videomomentfinder.com` serve identical HTML (same ETag), but apex 307-redirects to www, while `<link rel="canonical">` and `sitemap.xml` both point to apex. The canonical target itself redirects away from where it claims to be. Pick one host (www, since that is the redirect target) and align canonical + sitemap + `llms.txt` link hrefs to match.

---

## High Priority Issues (Fix Within 1 Week)

4. **`/support` FAQ has 9 verbatim Q&A pairs but no `FAQPage` schema.** The content is already server-rendered — marking it up is a zero-new-content win that feeds ChatGPT, Perplexity, Claude, and Gemini directly (Google ended FAQ rich results in 2023, but the schema still drives AI extraction). Ready-to-paste JSON-LD in §6.

5. **No author / about / creator page.** The maintainer's name (Juan Piñeros, `juancopi81`) and credentials are only on GitHub, not on the site. This is the single biggest E-E-A-T drag. Add `/about` with a bio, photo, relevant background, and `Person` schema with `sameAs` pointing to GitHub/LinkedIn/X.

6. **Missing security headers.** Only HSTS is set at the edge. Add via `next.config.js` `headers()`: `x-content-type-options: nosniff`, `referrer-policy: strict-origin-when-cross-origin`, `permissions-policy: camera=(), microphone=(), geolocation=()`, `x-frame-options: DENY`, and a baseline CSP. Vercel's defaults are minimal.

7. **Sitemap is incomplete.** Only 5 URLs; omits `/developers` (a ~650-word SSR content asset), `/skill.md`, `/llms.txt`, and `/connectors/claude`. No `<lastmod>` dates on any entry.

8. **Brand footprint is effectively zero.** No Wikipedia, no HN, no Reddit thread, no Product Hunt listing, no YouTube demo, no LinkedIn company page, 1 GitHub star. AI models have nothing outside the site itself to cross-corroborate the brand's existence. A single Show HN + Product Hunt launch + r/SideProject post would move Brand Authority from 8 toward 20-25 within days.

---

## Medium Priority Issues (Fix Within 1 Month)

9. **`llms.txt` is good but not great (72/100).** Does not list `/skill.md` (the single richest AI-agent signal on the site); link annotations are one-word; no `llms-full.txt` variant inlining the FAQ + skill content; no mention of the MCP endpoint or `/connectors/claude`.

10. **No explicit AI-bot stanzas in `robots.txt`.** Wildcard `Allow: /` currently covers GPTBot / ClaudeBot / PerplexityBot / OAI-SearchBot / Google-Extended, but adding explicit per-bot `Allow: /` lines removes ambiguity and signals intent — useful as CDN defaults trend toward default-deny for AI bots.

11. **No `og:image` / `twitter:image`.** Homepage ships no social-preview image and uses `summary` (small) instead of `summary_large_image`. Low cost, affects social sharing + some AI-card renderers.

12. **Qwen3-VL expertise signal is buried in FAQ.** The specific model + pipeline is a real expertise credential; surface it on a dedicated `/how-it-works` page (not just as an FAQ answer).

13. **`/dashboard` and `/video/*` return 200 despite robots.txt Disallow.** They are gated by client-side Clerk auth, not `noindex` meta or `X-Robots-Tag` at the edge. Well-behaved crawlers skip via robots.txt, but defense-in-depth calls for an edge `X-Robots-Tag: noindex` header on these paths.

14. **Homepage copy is a thin shell for crawlers.** Meta description is a single sentence; the hero content sits behind Clerk auth, so non-JS crawlers see `Loading authentication...` as the primary content. SSR a 3–4-paragraph feature description outside the auth boundary.

---

## Low Priority (Optimize When Possible)

15. No freshness signals (no `<lastmod>`, no "Last updated" stamps on FAQ, no changelog, no dated blog posts). 2026 copyright alone is weak.
16. Twitter card is `summary` — upgrade to `summary_large_image` once an `og:image` exists.
17. Add `WebSite` + `SearchAction` JSON-LD on homepage for potential sitelinks search box.
18. Consider `speakable` properties on `/support` Q&A answers as an explicit AI-assistant readability hint.

---

## Category Deep Dives

### AI Citability (55/100)

**Strengths.** The `/support` FAQ is the single strongest citable asset on the site — nine self-contained Q&A pairs with concrete numbers (~7-8 min processing for a 30-min video, 1 credit per video, Qwen3-VL embeddings, row-level security, 30-min video length cap). These are directly quotable by an LLM answering a user question. `skill.md` is exceptionally citable for agent-oriented queries — named tools, OAuth2+PKCE details, working curl examples — and is rare on a site this small.

**Weaknesses.** Homepage is a 16.8 KB marketing shell with a single-sentence meta and no extended prose an AI could quote. `/pricing` renders client-side and is effectively invisible. `/developers` has dense technical content but most of the value sits in interactive Swagger components rather than quotable prose. There is no JSON-LD FAQ/Product/SoftwareApplication schema anywhere to reinforce entity semantics.

### Brand Authority (8/100)

**Floor-level score, confirmed empirically.** Wikipedia: no article (API + direct 404). HN Algolia: 0 hits for "Video Moment Finder" or "videomomentfinder.com". Reddit: 0 relevant mentions. Product Hunt: no listing. YouTube: no branded demo or review (the "Grace AI" top hit is a different product). LinkedIn company page: not discoverable. GitHub: 1 star, 1 fork, 1 open issue, created 2026-01-19. The 8 points reflect (a) a live canonical domain, (b) a public open-source repo, (c) self-published entity signals (`llms.txt`, `skill.md`, MCP connector) that AI crawlers can ingest even without third-party validation.

**This is the category with the largest addressable gap.** A single coordinated launch day (Show HN + Product Hunt + r/SideProject + r/MachineLearning + a dev.to post) realistically moves this from 8 toward 20-25. Sustained Reddit + Twitter + a short YouTube demo push it past 40.

### Content E-E-A-T (38/100)

| Dimension | Score | Rationale |
|---|---|---|
| Experience | 48/100 | FAQ reads firsthand with product-specific numbers (30-min cap, 7-8 min processing, credit mechanics); /developers demonstrates implementation experience via real curl/CLI examples. |
| Expertise | 42/100 | Qwen3-VL model choice is a genuine expertise signal, but it is buried in a single FAQ answer rather than surfaced on a dedicated technical page. No author credentials shown. |
| Authoritativeness | 12/100 | No third-party citations, no press, no Wikipedia entry, 1 GitHub star. This is the hard part — it cannot be forced. |
| Trustworthiness | 52/100 | HTTPS, Privacy + Terms present, support email shown, AGPL-3.0 transparent, data-handling claims specific ("frames extracted, embeddings stored, source files deleted"). Drag: no visible owner identity on the site itself. |

**Largest single lift: a named `/about` page** with creator bio, photo, relevant background, `Person` schema with `sameAs` to GitHub/LinkedIn/X. Raises Expertise + Authoritativeness + Trustworthiness simultaneously.

### Technical GEO (74/100)

**Sub-scores:** Crawlability & indexability 78, Canonicalization & sitemap 55 (canonical-flip penalty), SSR / JS-dependency 70 (`/pricing` gap), Security & HTTPS 65 (HSTS only, no CSP/nosniff/referrer-policy), Performance signals 90 (Next.js on Vercel edge, HTML 16.8 KB, `x-vercel-cache: HIT`, `x-nextjs-prerender: 1`), Meta / OG / Twitter 80 (present but thin, no `og:image`). AI crawler access scored separately at 80 (permissive wildcard, appropriate `/dashboard` + `/video/` blocks, no explicit per-bot allowlists). `llms.txt` quality 72 (present, valid, under-described).

### Schema & Structured Data (3/100)

A raw-HTML scan of `/`, `/support`, `/developers`, and `/pricing` confirmed zero `application/ld+json` blocks, zero Microdata, zero RDFa. The 3 points are a participation floor for having basic Open Graph tags. Every audited route is prerendered, so JSON-LD can be injected via Next.js App Router static generation with zero runtime cost.

### Platform Optimization (24/100)

| Platform | Readiness | Why |
|---|---|---|
| Google AI Overviews | 22/100 | Thin content, zero schema, no backlink authority; technical SEO basics present but nothing for AIO to extract. |
| ChatGPT Search | 30/100 | `/support` FAQ is highly quotable and SSR; MCP connector is a meaningful forward signal; entity recognition near-zero. |
| Perplexity | 32/100 | SSR FAQ + fresh 2026 dates + permissive robots = best-structured surface; no Reddit/forum validation is the ceiling. |
| Google Gemini | 20/100 | Google-Extended unset (default permissive for live retrieval), but no YouTube/Knowledge Graph/news footprint. |
| Bing Copilot | 18/100 | No IndexNow, no `msvalidate.01`, no LinkedIn/GitHub org authority; 5-URL sitemap gives Bing almost nothing to crawl. |

**The MCP + `skill.md` + `llms.txt` + OpenAPI stack is genuinely ahead of the curve** — it represents direct-integration readiness (agents using the tool), which is a different axis from citation readiness. It will not move AIO or Copilot rankings in 2026, but it positions the product well for agent-native discovery channels as they mature.

---

## Quick Wins (Implement This Week)

1. **Ship FAQPage + SoftwareApplication + Organization JSON-LD.** Three blocks of ~30 lines each. Copy-paste snippets in §6 below. Single biggest-ROI change in the entire audit — lifts Citability, Schema, E-E-A-T, and every platform score.

2. **SSR the pricing page.** Move plan tiles into a server component that reads pricing constants; keep Stripe checkout client-side. One-hour change that eliminates the biggest crawler blind spot.

3. **Pick a canonical host and redirect the other.** Add a `Hosts` rule in `vercel.json` (301), and update `<link rel="canonical">`, `sitemap.xml`, and `llms.txt` link hrefs to match. Recommend sticking with `www` since that is where apex already redirects.

4. **Expand `sitemap.xml`** to include `/developers`, `/skill.md`, `/llms.txt`, `/connectors/claude`. Add `<lastmod>` to every entry.

5. **Add `/skill.md` to `llms.txt`** under `## Links`, and expand the one-word descriptions. Consider shipping `llms-full.txt` that inlines the FAQ and skill.md body.

6. **Add five security headers** via `next.config.js` `headers()`: `x-content-type-options`, `referrer-policy`, `permissions-policy`, `x-frame-options`, baseline CSP.

7. **Launch coordinated brand day:** Show HN post, Product Hunt listing, r/SideProject + r/MachineLearning posts, one short YouTube demo. Move Brand Authority from 8 → 20+ in a week.

---

## 30-Day Action Plan

### Week 1 — Schema, SSR, Canonical Cleanup
- [x] Deploy Organization + SoftwareApplication JSON-LD on homepage (§6.1, §6.2) — _shipped in PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72)_
- [x] Deploy FAQPage JSON-LD on `/support` with verbatim Q&A (§6.3) — _PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72)_
- [x] SSR `/pricing` with static plan tiles — _PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72)_
- [x] Pick canonical host (www), align canonical + sitemap + llms.txt — _PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72); 301 formalization deferred — existing Vercel 307 is sufficient_
- [x] Add `<lastmod>` and missing URLs to `sitemap.xml` — _PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72)_
- [x] Add security headers in `next.config.js` — _PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72); strict CSP deferred to follow-up_

### Week 2 — Author Identity & E-E-A-T
- [x] Create `/about` with creator bio, photo, credentials, and `Person` JSON-LD (`sameAs`: GitHub, LinkedIn, X) — _PR [#73](https://github.com/juancopi81/video-moment-finder/pull/73)_
- [x] Link `/about` from header or footer on every page — _PR [#73](https://github.com/juancopi81/video-moment-finder/pull/73)_
- [x] Add `og:image` / `twitter:image` (1200×630) and upgrade Twitter card to `summary_large_image` — _shipped early in PR [#72](https://github.com/juancopi81/video-moment-finder/pull/72)_
- [x] Add `/how-it-works` page surfacing Qwen3-VL, frame pipeline, accuracy tradeoffs — _branch `feat/how-it-works-and-faq-dates`_
- [x] Add "Last updated" timestamps to `/support` answers — _branch `feat/how-it-works-and-faq-dates`_

### Week 3 — Brand Authority Launch
- [ ] Show HN submission ("Show HN: Video Moment Finder — open-source semantic search for video, with MCP connector")
- [ ] Product Hunt launch
- [ ] r/SideProject, r/MachineLearning, r/selfhosted posts
- [ ] One 3-minute YouTube demo (also embed on homepage)
- [ ] Publish a dev.to post walking through the MCP connector architecture
- [ ] Create LinkedIn company page with GitHub + site `sameAs`

### Week 4 — Agent Polish & Freshness Cadence
- [ ] Ship `llms-full.txt` inlining FAQ + skill.md body
- [ ] Add explicit per-bot `User-agent: GPTBot / ClaudeBot / PerplexityBot / OAI-SearchBot / Google-Extended` + `Allow: /` stanzas to `robots.txt`
- [ ] Add edge `X-Robots-Tag: noindex` for `/dashboard/*` and `/video/*`
- [ ] Add BreadcrumbList JSON-LD on `/support`, `/developers`, `/pricing`
- [ ] Start a `/changelog` page (even 3 entries is enough to create a freshness signal)

---

## §6 — Ready-to-Deploy JSON-LD Snippets

**Recommended Next.js 15 (App Router) pattern** — use the built-in `next/script` component, which escapes JSON safely:

```tsx
import Script from 'next/script'

const orgSchema = { /* see §6.1 below */ }

export default function Layout({ children }) {
  return (
    <>
      <Script
        id="org-schema"
        type="application/ld+json"
        strategy="beforeInteractive"
      >
        {JSON.stringify(orgSchema)}
      </Script>
      {children}
    </>
  )
}
```

Alternatively, emit the block from a Server Component directly in the JSX (`<script type="application/ld+json">{JSON.stringify(orgSchema)}</script>`) — in a React Server Component the body is treated as text content, not HTML, so it is XSS-safe for static JSON.

### §6.1 Organization (homepage `<head>`)

```json
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "@id": "https://www.videomomentfinder.com/#organization",
  "name": "Video Moment Finder",
  "url": "https://www.videomomentfinder.com/",
  "logo": {
    "@type": "ImageObject",
    "url": "https://www.videomomentfinder.com/icon.png",
    "width": 32,
    "height": 32
  },
  "description": "Semantic video frame search. Find moments using text or images.",
  "sameAs": [
    "https://github.com/juancopi81/video-moment-finder"
  ],
  "contactPoint": {
    "@type": "ContactPoint",
    "email": "support@videomomentfinder.com",
    "contactType": "customer support",
    "availableLanguage": "en"
  }
}
```

### §6.2 SoftwareApplication (homepage `<head>`)

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "@id": "https://www.videomomentfinder.com/#software",
  "name": "Video Moment Finder",
  "url": "https://www.videomomentfinder.com/",
  "description": "AI-powered semantic video frame search. Process videos up to 30 minutes, extract frames, generate multimodal embeddings with Qwen3-VL, and search by text query or example image to find specific moments.",
  "applicationCategory": "MultimediaApplication",
  "applicationSubCategory": "Video Search",
  "operatingSystem": "Any",
  "browserRequirements": "Requires a modern browser with JavaScript enabled",
  "featureList": [
    "Semantic text-to-video search",
    "Image-to-video similarity search",
    "Multimodal embeddings (Qwen3-VL)",
    "Direct video upload (up to 30 minutes)",
    "YouTube URL import for owned videos",
    "Row-level security for user data"
  ],
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD",
    "description": "Free tier includes 1 credit (1 video up to 30 minutes). Paid credit packs available; credits do not expire."
  },
  "license": "https://www.gnu.org/licenses/agpl-3.0.html",
  "codeRepository": "https://github.com/juancopi81/video-moment-finder",
  "publisher": { "@id": "https://www.videomomentfinder.com/#organization" }
}
```

### §6.3 FAQPage (`/support` `<head>`, verbatim Q&A)

```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "url": "https://www.videomomentfinder.com/support",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What video sources work best?",
      "acceptedAnswer": { "@type": "Answer", "text": "Direct upload is the reliable path. You can upload video files you own or are authorized to use, and videos must be 30 minutes or shorter. YouTube URL import is also available for owned videos, but server-side restrictions can block it." }
    },
    {
      "@type": "Question",
      "name": "What should I do if YouTube import fails?",
      "acceptedAnswer": { "@type": "Answer", "text": "Upload the video file directly instead. If this is your own YouTube video, use official download paths such as YouTube Studio or Google Takeout, then upload the file here. We do not recommend relying on third-party downloader sites." }
    },
    {
      "@type": "Question",
      "name": "How does the search work?",
      "acceptedAnswer": { "@type": "Answer", "text": "We use AI-powered semantic search. When you process a video, we extract frames and create embeddings using a multimodal AI model (Qwen3-VL). When you search, your text query or uploaded example image is embedded into the same space to find the most visually relevant moments." }
    },
    {
      "@type": "Question",
      "name": "How long does processing take?",
      "acceptedAnswer": { "@type": "Answer", "text": "Processing time depends on the video length. A 30-minute video typically takes around 7-8 minutes. You can leave the page and come back — processing continues in the background." }
    },
    {
      "@type": "Question",
      "name": "What happens to my video data?",
      "acceptedAnswer": { "@type": "Answer", "text": "We extract frames and generate AI embeddings for search. Thumbnail images and search vectors are stored to enable search. Uploaded source files are automatically cleaned up after processing, and submitted YouTube URLs are only used to attempt import and processing. We do not share your video data with third parties." }
    },
    {
      "@type": "Question",
      "name": "How do credits work?",
      "acceptedAnswer": { "@type": "Answer", "text": "Each credit allows you to process one video (up to 30 minutes). Free accounts include 1 credit to try the service. Paid credit packs are available on the pricing page for signed-in users. Credits do not expire." }
    },
    {
      "@type": "Question",
      "name": "How accurate are the search results?",
      "acceptedAnswer": { "@type": "Answer", "text": "Search results are AI-generated and may not always be perfectly accurate. The quality depends on video content, query phrasing, and how closely an example image matches the scene you want. Try different wording or a different image if the first search doesn't find what you need." }
    },
    {
      "@type": "Question",
      "name": "Can I get a refund?",
      "acceptedAnswer": { "@type": "Answer", "text": "Credits that have been used for processing are non-refundable. If you experience a technical issue that consumes a credit without delivering results, contact us and we will review your case." }
    },
    {
      "@type": "Question",
      "name": "Is my data secure?",
      "acceptedAnswer": { "@type": "Answer", "text": "Yes. All connections use HTTPS encryption. Your data is scoped to your authenticated account with row-level security. We use industry-standard infrastructure providers for storage and processing." }
    }
  ]
}
```

---

## Appendix: Pages Analyzed

| URL | Status | SSR? | Notes |
|---|---|---|---|
| `/` (homepage) | 200 | Yes (prerendered) | Title/meta/OG present, no JSON-LD, hero content behind Clerk auth |
| `/pricing` | 200 | **No (client-only)** | "Loading pricing…" fallback — GEO blind spot |
| `/support` | 200 | Yes | 9-item FAQ, ~650 words, no FAQPage schema |
| `/developers` | 200 | Yes | MCP + OpenAPI docs, ~650-700 words, missing from sitemap |
| `/privacy` | 200 | Yes | Standard legal |
| `/terms` | 200 | Yes | Standard legal |
| `/skill.md` | 200 | n/a | Agent bootstrap file — excellent |
| `/llms.txt` | 200 | n/a | Present, valid, 1382 bytes |
| `/robots.txt` | 200 | n/a | Wildcard allow, blocks `/dashboard` + `/video/` |
| `/sitemap.xml` | 200 | n/a | Only 5 URLs, missing `/developers` et al. |
| apex `videomomentfinder.com` | 307 → www | n/a | Canonical-flip: sitemap and canonical point to apex, which redirects to www |

---

*Audit performed via the `/geo audit` skill with 5 parallel specialist subagents (AI visibility, platform optimization, technical GEO, content E-E-A-T, schema). Methodology and scoring weights are documented in `~/.claude/skills/geo-audit/SKILL.md`.*
