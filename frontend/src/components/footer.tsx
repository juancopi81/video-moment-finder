import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-zinc-200 bg-surface-card dark:border-zinc-800">
      <div className="mx-auto max-w-5xl px-4 py-10">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
          <div>
            <p className="font-heading font-bold">
              Video Moment Finder
            </p>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Find any moment in any video with AI-powered semantic search.
            </p>
          </div>

          <div>
            <p className="text-sm font-semibold">Product</p>
            <ul className="mt-2 space-y-1 text-sm text-zinc-600 dark:text-zinc-400">
              <li>
                <Link href="/about" className="hover:text-foreground">
                  About
                </Link>
              </li>
              <li>
                <Link href="/pricing" className="hover:text-foreground">
                  Pricing
                </Link>
              </li>
              <li>
                <Link href="/how-it-works" className="hover:text-foreground">
                  How it works
                </Link>
              </li>
              <li>
                <Link href="/support" className="hover:text-foreground">
                  Support
                </Link>
              </li>
              <li>
                <Link href="/developers" className="hover:text-foreground">
                  Developers
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <p className="text-sm font-semibold">Legal</p>
            <ul className="mt-2 space-y-1 text-sm text-zinc-600 dark:text-zinc-400">
              <li>
                <Link href="/terms" className="hover:text-foreground">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link href="/privacy" className="hover:text-foreground">
                  Privacy Policy
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 border-t border-zinc-200 pt-6 text-center text-xs text-zinc-500 dark:border-zinc-800">
          <p>
            &copy; {new Date().getFullYear()} Video Moment Finder. Open-source under AGPLv3.
          </p>
          <p className="mt-1">
            Source code:{" "}
            <a
              href="https://github.com/juancopi81/video-moment-finder"
              target="_blank"
              rel="noreferrer"
              className="hover:text-foreground"
            >
              github.com/juancopi81/video-moment-finder
            </a>
          </p>
          <p className="mt-1">
            Contact:{" "}
            <a
              href="mailto:support@videomomentfinder.com"
              className="hover:text-foreground"
            >
              support@videomomentfinder.com
            </a>
          </p>
        </div>
      </div>
    </footer>
  );
}
