import { ApiBillingSummary } from "@/lib/api-billing";

type ApiBalanceCardProps = {
  summary: ApiBillingSummary;
  className?: string;
};

export function ApiBalanceCard({
  summary,
  className = "",
}: ApiBalanceCardProps) {
  return (
    <div
      className={`rounded-xl border border-zinc-200 bg-surface-card px-4 py-3 text-sm dark:border-zinc-800 ${className}`.trim()}
    >
      <p className="font-medium text-zinc-900 dark:text-zinc-100">
        {summary.api_units_balance.toLocaleString()} units remaining
      </p>
      <p className="mt-1 text-zinc-600 dark:text-zinc-400">
        Approximately {summary.approx_videos.toLocaleString()} video
        {summary.approx_videos !== 1 ? "s" : ""} or{" "}
        {summary.approx_queries.toLocaleString()} quer
        {summary.approx_queries !== 1 ? "ies" : "y"}
      </p>
    </div>
  );
}
