import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "Privacy Policy for Video Moment Finder.",
  alternates: { canonical: "/privacy" },
};

export default function PrivacyPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="font-heading text-4xl font-bold">
        Privacy Policy
      </h1>
      <p className="mt-2 text-sm text-zinc-500">Last updated: November 12, 2026</p>

      <div className="mt-8 space-y-8 text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
        <section>
          <h2 className="text-lg font-semibold text-foreground">1. Information We Collect</h2>
          <p className="mt-2">We collect the following types of information:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              <strong>Account information:</strong> Email address, internal account identifier,
              and authentication and verification state provided through our identity provider (Clerk).
            </li>
            <li>
              <strong>Video data:</strong> Video files you upload for processing and optional
              YouTube URLs you submit for import, original filenames, and the images, speech,
              text, and other information contained in that material. Your content and filenames
              may include personal information about you or other people.
            </li>
            <li>
              <strong>Usage data:</strong> Search queries, optional example images submitted for
              search, processing history, website credit and API-unit usage, and
              one-time trial grant records. Operational records associate your account with
              video identifiers, operation types, timestamps, processing status, and units used.
            </li>
            <li>
              <strong>Website payment records:</strong> For purchases on the separate VMF
              website, Lemon Squeezy sends payment notifications that VMF stores to apply
              credits and prevent duplicate grants. These can include customer name and
              email, customer/order/subscription identifiers, purchased items, amounts,
              currency, tax, payment/refund status, event timestamps, receipt links, and
              card brand and last four digits when supplied by the processor. VMF does not
              receive full payment-card numbers. The plugin does not initiate purchases or
              return these payment records.
            </li>
            <li>
              <strong>Connected app authorization:</strong> OAuth client and
              connection records, approved permissions, authorization requests,
              token hashes, identifiers and timestamps for these records, and expiration or
              revocation state used to authenticate connected apps. Account passwords and
              authentication tokens must not be pasted into chat.
            </li>
            <li>
              <strong>Traffic analytics:</strong> We use Vercel Web Analytics to measure page
              visits and visitor trends across the website. We use this to understand which pages
              are useful, detect site regressions, and prioritize improvements to the hosted
              product.
            </li>
            <li>
              <strong>Product analytics:</strong> We use PostHog to record client-side product
              analytics such as page views, feature usage, and button clicks. We also record
              limited server-side product events, such as signup completion, processing
              outcomes, searches and checkout outcomes, in our own database. These records
              can contain the account and video identifiers, operation type, result count,
              and event time. Website analytics can include browser/device information,
              referrer and campaign information, and cookie-based visitor/session identifiers.
              Signed-in website events are associated with your internal account identifier.
              Hosting and analytics providers also receive network information such as IP addresses.
            </li>
            <li>
              <strong>Operational diagnostics:</strong> Hosting and processing logs may contain
              request paths, status codes, durations, error messages, timestamps, and account
              or video references used to diagnose failures. Sentry receives exception messages,
              stack traces, service/version labels, event timestamps, and related account,
              job or video identifiers for error monitoring. Raw diagnostic messages and
              internal request, trace, session, and account identifiers are not returned by
              the plugin&apos;s video tools.
            </li>
            <li>
              <strong>Prepared learning content:</strong> When you ask to display a guide,
              flashcards, a Playground, or slides, the connected app sends that prepared
              content and its selected source references to VMF for validation and display.
              This operation does not store a learner profile or a copy in VMF&apos;s database.
            </li>
          </ul>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">2. How We Use Your Information</h2>
          <p className="mt-2">We use your information to:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Process videos and enable semantic search functionality</li>
            <li>Manage your account, connected app permissions, credit and API-unit balances, and trial eligibility</li>
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
            <li>Transcripts with segment timestamps, when available</li>
            <li>Metadata (timestamps, video ID, processing status)</li>
          </ul>
          <p className="mt-2">
            Original uploaded video files are stored temporarily for processing, playback,
            and high-resolution frame retrieval. A storage lifecycle rule schedules their
            deletion 30 days after upload. Account-owned originals and thumbnails are
            delivered through temporary signed links rather than a public bucket URL.
          </p>
          <p className="mt-2">
            If you submit a YouTube URL, we use that URL to attempt a server-side import for the
            corresponding video. Direct upload remains the primary supported ingest path.
          </p>
          <p className="mt-2">
            Example images uploaded for search are used to generate a query embedding and
            are not added to your video library or saved in VMF&apos;s database or object
            storage. Our processing provider, Modal, can retain function inputs and outputs
            for up to seven days, as explained below.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">4. Third-Party Services</h2>
          <p className="mt-2">
            We use the following third-party services to operate. Because Video Moment Finder is
            an open-source product, we want to be explicit about which providers help us run the
            hosted service and why.
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              <strong>Vercel:</strong> Frontend hosting and web analytics
            </li>
            <li>
              <strong>Railway:</strong> API and worker hosting, including operational logs
            </li>
            <li>
              <strong>Sentry:</strong> API and worker error monitoring and diagnostics
            </li>
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
            <li>
              <strong>PostHog:</strong> Product analytics and funnel measurement
            </li>
          </ul>
          <p className="mt-2">
            Each provider processes data in accordance with their own privacy policies.
          </p>
          <p className="mt-2">
            When you authorize a connected app such as ChatGPT, its approved tools can receive
            the video identifier needed to select your source, its filename or YouTube source
            reference, source type, and processing status; timestamped search matches,
            similarity scores, transcript text and language, and requested frame images;
            your available API-unit balance and the cost of an operation; and the prepared
            learning content you ask to display. These categories also apply when the data
            appears inside a list, search result, source reference, or learning view.
          </p>
          <p className="mt-2">
            The workspace also receives temporary signed playback and thumbnail links in
            component-only tool metadata to load the private media. Starting an explicitly
            requested upload returns a temporary signed upload link and transfer instructions
            so the client can send the file. These links grant temporary access to the
            specified object; they must not be included in exported learning materials.
            Video identifiers allow follow-up tools to select the same source; filenames help
            you recognize it.
            Tools do not return your account email or account identifier, VMF authentication
            tokens, trial/grant history, record-creation timestamps, or raw backend diagnostics.
            Personal information present in the requested video, filename, transcript, or
            learning content is still part of that content.
          </p>
          <p className="mt-2">
            The receiving app processes and retains tool results and conversation content
            under its own terms and privacy policy. VMF does not request your full chat history.
            Selecting a passage or a learning action shares only the selected video reference,
            timestamp, and relevant excerpt, requested frame, or learning-view context with
            the connected conversation.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">5. Data Retention</h2>
          <p className="mt-2">
            Processed video data (embeddings, thumbnails, and transcripts) is retained as long as your account
            is active. Uploaded source video files are scheduled for automatic deletion
            30 days after upload; cleanup may complete later. Playback and high-resolution
            frames become unavailable when the original is deleted, while stored transcripts
            and thumbnails remain available. Temporary link expiration does not delete the
            stored video. Already-deleted originals are not restored by this retention policy.
            You can request deletion of all your data by contacting us.
          </p>
          <p className="mt-2">
            Account records, completed authorization/connection records, usage and billing
            history, and first-party product events currently have no fixed automatic deletion
            period in VMF&apos;s database. Website product analytics also currently have no
            scheduled automatic event deletion in our PostHog project. You can request deletion
            through the contact below. Disconnecting a plugin does not delete these records
            or your video library; contact support for help revoking remaining authorization.
          </p>
          <p className="mt-2">
            Pending authorization requests expire after one hour and expired pending requests
            are cleaned up when a new authorization starts. Authorization codes expire after
            ten minutes, access tokens after one hour, and refresh tokens after thirty days.
            Expiration or revocation prevents use of a credential; it does not itself delete
            the associated authorization or accounting records.
          </p>
          <p className="mt-2">
            Modal processes search text, example search images, extracted video frames,
            audio, and transcript text through remote functions. Modal&apos;s published
            retention policy states that function inputs and outputs, including generated
            embeddings and transcription results, can be retained for up to seven days
            before deletion. App, function, and container metadata remains for the lifetime
            of our Modal account. On our current Starter plan, Modal retains app and
            container logs for one day.
            See{" "}
            <a href="https://modal.com/docs/guide/security" className="text-accent hover:underline">
              Modal&apos;s security and privacy documentation
            </a>
            . This provider retention is separate from the copies of indexed video data
            that VMF keeps for your library.
          </p>
          <p className="mt-2">
            Prepared learning views are not persisted in VMF&apos;s database; copies in chats
            or exported files remain subject to the receiving app&apos;s retention and your
            own file controls. Our current Sentry Developer plan retains new error events
            for thirty days; events received under a different plan keep the retention
            period that applied when they were received. Infrastructure providers retain technical logs and backups
            according to their configured service policies, separately from VMF&apos;s
            application records. A provider&apos;s log viewing window does not necessarily
            mean all copies are permanently deleted at its end. Deleting active application
            records does not immediately remove copies in provider backups.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-foreground">6. Cookies</h2>
          <p className="mt-2">
            We use essential cookies for authentication session management. We also use Vercel Web
            Analytics and PostHog to understand overall page traffic and whether our public site
            is helping users discover and trust the product. PostHog uses a cookie for session
            attribution. We use this analytics tooling for product improvement and reliability,
            not for advertising or cross-site retargeting.
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
            <li>Disconnect VMF in the connected app&apos;s settings and request authorization revocation from support</li>
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
