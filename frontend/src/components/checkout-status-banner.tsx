type CheckoutStatusBannerProps = {
  status: string | null;
  successMessage: string;
  cancelMessage: string;
  className?: string;
};

export function CheckoutStatusBanner({
  status,
  successMessage,
  cancelMessage,
  className = "",
}: CheckoutStatusBannerProps) {
  if (status !== "success" && status !== "cancel") {
    return null;
  }

  const toneClass =
    status === "success"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-200"
      : "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200";
  const message = status === "success" ? successMessage : cancelMessage;

  return (
    <p
      className={`rounded-lg border px-4 py-2 text-sm ${toneClass} ${className}`.trim()}
    >
      {message}
    </p>
  );
}
