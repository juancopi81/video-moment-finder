export type ApiUnitTrialSummary = {
  trial_enabled?: boolean;
  trial_status?:
    | "disabled"
    | "verification_required"
    | "verification_unavailable"
    | "granted"
    | "exhausted";
  trial_units_granted?: number;
  trial_allowance_units?: number;
  trial_legacy_units_offset?: number;
};

export function apiUnitTrialMessage(summary: ApiUnitTrialSummary): string | null {
  // Only a reported grant proves enrollment. Keep it visible after rollout is
  // paused, without treating the current total balance as trial-only units.
  if (
    (summary.trial_status === "granted" || summary.trial_status === "exhausted") &&
    typeof summary.trial_units_granted === "number" &&
    summary.trial_units_granted >= 0
  ) {
    const offset = summary.trial_legacy_units_offset ?? 0;
    return `One-time trial: ${summary.trial_units_granted.toLocaleString()} API units granted to this account.${offset > 0 ? ` Prior free website processing used ${offset.toLocaleString()} units of the allowance.` : ""} Reconnecting does not add another trial.`;
  }

  if (!summary.trial_enabled) return null;

  if (summary.trial_status === "verification_required") {
    return "Verify your primary email address in your Video Moment Finder account, then refresh to check trial eligibility.";
  }

  if (summary.trial_status === "verification_unavailable") {
    return "Account verification is temporarily unavailable. Refresh later to check trial eligibility.";
  }

  return null;
}
