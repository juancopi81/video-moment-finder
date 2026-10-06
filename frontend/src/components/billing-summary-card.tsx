import { BillingSummary } from "@/lib/billing";
import { ApiTrialNotice } from "@/components/api-trial-notice";

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
        Website video credits: {summary.credits_balance}
      </p>
      <p className="mt-1 text-zinc-600 dark:text-zinc-400">
        {summary.has_unlimited_access
          ? "Unlimited access enabled."
          : summary.trial_status === "granted" || summary.trial_status === "exhausted"
            ? `Shared API balance: ${(summary.api_units_balance ?? 0).toLocaleString()} units remaining. Website indexing uses ${summary.unit_cost_index_video?.toLocaleString() ?? "the configured number of"} units when sufficient units remain, then falls back to an available website video credit.`
            : `Free website videos remaining: ${summary.free_videos_remaining}/${summary.free_videos_limit}`}
      </p>
      <ApiTrialNotice summary={summary} />
    </div>
  );
}
