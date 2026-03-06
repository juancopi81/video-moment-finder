import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy - Video Moment Finder",
  description: "Privacy Policy for Video Moment Finder.",
};

export default function PrivacyPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="font-heading text-4xl font-bold">
        Privacy Policy
      </h1>
      <p className="mt-2 text-sm text-zinc-500">Last updated: March 6, 2026</p>

      <div className="mt-8 space-y-8 text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
        <section>
          <h2 className="text-lg font-semibold text-foreground">1. Information We Collect</h2>
          <p className="mt-2">We collect the following types of information:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              <strong>Account information:</strong> Email address and authentication data provided
              through our identity provider (Clerk).
            </li>
            <li>
              <strong>Video data:</strong> YouTube URLs you submit and video files you upload for
              processing.
            </li>
            <li>
              <strong>Usage data:</strong> Search queries, optional example images submitted for
              search, processing history, and credit usage.
            </li>
          </ul>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">2. How We Use Your Information</h2>
          <p className="mt-2">We use your information to:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Process videos and enable semantic search functionality</li>
            <li>Manage your account and credit balance</li>
            <li>Improve the Service and develop new features</li>
            <li>Communicate with you about your account and the Service</li>
          </ul>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">3. Video Data Handling</h2>
          <p className="mt-2">
            When you submit a video, we extract frames and generate AI embeddings (numerical
            representations) for search. We store:
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Thumbnail images of extracted frames</li>
            <li>Embedding vectors for search indexing</li>
            <li>Metadata (timestamps, video ID, processing status)</li>
          </ul>
          <p className="mt-2">
            Original uploaded video files are stored temporarily for processing and are subject to
            automatic cleanup via storage lifecycle rules.
          </p>
          <p className="mt-2">
            Example images uploaded for search are processed transiently to generate a query
            embedding and are not retained after the request completes.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">4. Third-Party Services</h2>
          <p className="mt-2">We use the following third-party services to operate:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              <strong>Clerk:</strong> Authentication and user management
            </li>
            <li>
              <strong>Supabase:</strong> Database and job queue
            </li>
            <li>
              <strong>Modal:</strong> GPU compute for AI processing
            </li>
            <li>
              <strong>Qdrant:</strong> Vector database for search indexing
            </li>
            <li>
              <strong>Cloudflare R2:</strong> Object storage for thumbnails and uploaded files
            </li>
            <li>
              <strong>Lemon Squeezy:</strong> Payment processing
            </li>
          </ul>
          <p className="mt-2">
            Each provider processes data in accordance with their own privacy policies.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">5. Data Retention</h2>
          <p className="mt-2">
            Processed video data (embeddings and thumbnails) is retained as long as your account
            is active. Uploaded source video files are automatically deleted after a short
            retention period. You can request deletion of all your data by contacting us.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">6. Cookies</h2>
          <p className="mt-2">
            We use essential cookies for authentication session management. We do not use
            tracking or advertising cookies.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">7. Security</h2>
          <p className="mt-2">
            We implement industry-standard security measures including encrypted connections
            (HTTPS), authenticated API access, and row-level security on database records. Access
            to your data is scoped to your authenticated account.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">8. Your Rights</h2>
          <p className="mt-2">You have the right to:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Access the personal data we hold about you</li>
            <li>Request correction of inaccurate data</li>
            <li>Request deletion of your data and account</li>
            <li>Export your data in a portable format</li>
          </ul>
          <p className="mt-2">
            To exercise these rights, contact us at the address below.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">9. Changes to This Policy</h2>
          <p className="mt-2">
            We may update this Privacy Policy from time to time. We will notify users of material
            changes via email or through the Service.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">10. Contact</h2>
          <p className="mt-2">
            For privacy-related inquiries, contact us at{" "}
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
