import { BillingSummary } from "@/lib/billing";

type BillingSummaryCardProps = {
  summary: BillingSummary;
  className?: string;
};

export function BillingSummaryCard({
  summary,
  className = "",
}: BillingSummaryCardProps) {
  return (
    <div
      className={`rounded-xl border border-zinc-200 bg-surface-card px-4 py-3 text-sm dark:border-zinc-800 ${className}`.trim()}
    >
      <p className="font-medium text-zinc-900 dark:text-zinc-100">
        Credits available: {summary.credits_balance}
      </p>
      <p className="mt-1 text-zinc-600 dark:text-zinc-400">
        {summary.has_unlimited_access
          ? "Unlimited access enabled."
          : `Free trial remaining: ${summary.free_videos_remaining}/${summary.free_videos_limit}`}
      </p>
    </div>
  );
}
