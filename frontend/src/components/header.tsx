"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SignInButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";

const navLinks = [
  { href: "/pricing", label: "Pricing" },
  { href: "/support", label: "Support" },
];

function MobileMenu({ onClose }: { onClose: () => void }) {
  return (
    <nav className="border-t border-zinc-200 bg-background px-4 py-3 dark:border-zinc-800 md:hidden">
      <div className="flex flex-col gap-3">
        {navLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className="text-sm text-zinc-600 hover:text-foreground dark:text-zinc-400"
            onClick={onClose}
          >
            {link.label}
          </Link>
        ))}
        <SignedOut>
          <SignInButton mode="modal">
            <button className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white">
              Sign In
            </button>
          </SignInButton>
        </SignedOut>
        <SignedIn>
          <UserButton afterSignOutUrl="/" />
        </SignedIn>
      </div>
    </nav>
  );
}

export function Header() {
  const [menuOpenOnPath, setMenuOpenOnPath] = useState<string | null>(null);
  const pathname = usePathname();
  const menuOpen = menuOpenOnPath === pathname;

  function toggleMenu() {
    setMenuOpenOnPath(menuOpen ? null : pathname);
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-200 bg-background/80 backdrop-blur-md dark:border-zinc-800">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <Link
          href="/"
          className="font-heading text-lg font-bold tracking-tight"
        >
          Video Moment Finder
        </Link>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-6 md:flex">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="text-sm text-zinc-600 hover:text-foreground dark:text-zinc-400"
            >
              {link.label}
            </Link>
          ))}
          <SignedOut>
            <SignInButton mode="modal">
              <button className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white">
                Sign In
              </button>
            </SignInButton>
          </SignedOut>
          <SignedIn>
            <UserButton afterSignOutUrl="/" />
          </SignedIn>
        </nav>

        {/* Mobile hamburger */}
        <button
          type="button"
          className="md:hidden p-2 text-zinc-600 dark:text-zinc-400"
          onClick={toggleMenu}
          aria-label="Toggle menu"
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            {menuOpen ? (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            ) : (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            )}
          </svg>
        </button>
      </div>

      {menuOpen && (
        <MobileMenu onClose={() => setMenuOpenOnPath(null)} />
      )}
    </header>
  );
}
