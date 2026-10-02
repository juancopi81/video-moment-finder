import { apiUnitTrialMessage, type ApiUnitTrialSummary } from "@/lib/api-unit-trial";

export function ApiTrialNotice({ summary }: { summary: ApiUnitTrialSummary }) {
  const message = apiUnitTrialMessage(summary);
  if (!message) return null;

  return (
    <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
      {message}
    </p>
  );
}
