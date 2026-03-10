import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "Terms of Service for Video Moment Finder.",
  alternates: { canonical: "/terms" },
};

export default function TermsPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="font-heading text-4xl font-bold">
        Terms of Service
      </h1>
      <p className="mt-2 text-sm text-zinc-500">Last updated: March 10, 2026</p>

      <div className="mt-8 space-y-8 text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
        <section>
          <h2 className="text-lg font-semibold text-foreground">1. Acceptance of Terms</h2>
          <p className="mt-2">
            By accessing or using Video Moment Finder (&quot;the Service&quot;), you agree to be
            bound by these Terms of Service. If you do not agree to these terms, do not use the
            Service.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">2. Description of Service</h2>
          <p className="mt-2">
            Video Moment Finder is a web-based tool that allows users to search for specific
            moments within videos using AI-powered semantic search. Users primarily upload video
            files and may also submit YouTube URLs as a best-effort convenience for content they
            own or are authorized to use. The Service processes submitted videos to enable
            text-based or image-based frame search.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">3. User Accounts</h2>
          <p className="mt-2">
            You must create an account to use the Service. You are responsible for maintaining the
            security of your account credentials. You must not share your account with others or
            allow unauthorized access.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">4. Acceptable Use</h2>
          <p className="mt-2">You agree not to:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Upload or process content that you do not have the right to use</li>
            <li>Submit YouTube URLs for content you do not own or are not authorized to use</li>
            <li>Use the Service to infringe on intellectual property rights</li>
            <li>Attempt to reverse-engineer, exploit, or abuse the Service</li>
            <li>Upload content that is illegal, harmful, or violates third-party rights</li>
            <li>Use automated tools to scrape or overload the Service</li>
          </ul>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">5. Prohibited Content</h2>
          <p className="mt-2">
            You may not submit, upload, or process any content that falls into the following
            categories. We reserve the right to remove such content without notice and to suspend
            or terminate accounts that violate this policy.
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              <strong>Sexually explicit or NSFW material:</strong> Pornography, nudity, or any
              content intended to be sexually gratifying
            </li>
            <li>
              <strong>Violence and gore:</strong> Graphic depictions of real-world violence, injury,
              or death
            </li>
            <li>
              <strong>Illegal content:</strong> Material that violates any applicable law, including
              child sexual abuse material (CSAM), non-consensual intimate imagery, or content that
              promotes illegal activity
            </li>
            <li>
              <strong>Hate speech and harassment:</strong> Content that promotes hatred, discrimination,
              or violence against individuals or groups based on protected characteristics
            </li>
            <li>
              <strong>Profanity and abusive language:</strong> Queries or content primarily consisting of
              obscene, vulgar, or abusive language
            </li>
            <li>
              <strong>Terrorism and extremism:</strong> Content that promotes, glorifies, or recruits for
              terrorist organizations or extremist ideologies
            </li>
          </ul>
          <p className="mt-2">
            If you encounter prohibited content on the Service, please report it to{" "}
            <a
              href="mailto:support@videomomentfinder.com"
              className="text-accent hover:underline"
            >
              support@videomomentfinder.com
            </a>
            .
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">6. Credit-Based Billing</h2>
          <p className="mt-2">
            The Service uses a credit-based pricing model. Each credit allows processing of one
            video. Credits are non-refundable once used. Unused credits do not expire. Pricing
            and credit allocations may change with notice.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">7. Intellectual Property</h2>
          <p className="mt-2">
            You retain all rights to the videos you submit. We do not claim ownership of your
            content. The Service, its design, code, and AI models are the intellectual property of
            Video Moment Finder. The source code for Video Moment Finder is licensed under the GNU
            Affero General Public License v3.0 (AGPLv3), while these Terms continue to govern your
            use of our hosted service at videomomentfinder.com.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">8. Data Processing</h2>
          <p className="mt-2">
            Videos you submit are processed to extract frames and generate embeddings for search.
            Processed data (embeddings and thumbnails) is stored to enable search functionality.
            Example images submitted for search are processed transiently and are not retained
            after the request. See our Privacy Policy for details on data handling.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">9. Limitation of Liability</h2>
          <p className="mt-2">
            The Service is provided &quot;as is&quot; without warranties of any kind. We are not
            liable for any indirect, incidental, or consequential damages arising from your use of
            the Service. Our total liability is limited to the amount you paid in the 12 months
            preceding the claim.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">10. Termination</h2>
          <p className="mt-2">
            We may suspend or terminate your access to the Service at any time for violations of
            these terms. You may delete your account at any time. Upon termination, your processed
            data will be deleted in accordance with our data retention policy.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">11. Changes to Terms</h2>
          <p className="mt-2">
            We may update these terms from time to time. We will notify users of material changes
            via email or through the Service. Continued use after changes constitutes acceptance.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">12. Contact</h2>
          <p className="mt-2">
            For questions about these terms, contact us at{" "}
            <a
              href="mailto:support@videomomentfinder.com"
              className="text-accent hover:underline"
            >
              support@videomomentfinder.com
            </a>
            .
          </p>
        </section>
      </div>
    </div>
  );
}
