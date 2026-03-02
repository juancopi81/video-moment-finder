import Link from "next/link";

type PricingCardBaseProps = {
  name: string;
  price: string;
  description: string;
  features: string[];
  highlighted?: boolean;
  ctaLabel: string;
};

type PricingCardLinkProps = PricingCardBaseProps & {
  ctaHref: string;
  onCtaClick?: never;
  ctaDisabled?: never;
};

type PricingCardButtonProps = PricingCardBaseProps & {
  ctaHref?: never;
  onCtaClick: () => void | Promise<void>;
  ctaDisabled?: boolean;
};

type PricingCardProps = PricingCardLinkProps | PricingCardButtonProps;

export function PricingCard(props: PricingCardProps) {
  const {
    name,
    price,
    description,
    features,
    highlighted = false,
    ctaLabel,
  } = props;
  const ctaClassName = `mt-6 block rounded-lg px-4 py-2.5 text-center text-sm font-medium ${
    highlighted
      ? "bg-accent text-white"
      : "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
  }`;

  return (
    <div
      className={`relative flex flex-col rounded-xl border p-6 ${
        highlighted
          ? "border-accent shadow-lg"
          : "border-zinc-200 dark:border-zinc-800"
      }`}
    >
      {highlighted && (
        <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-accent px-3 py-0.5 text-xs font-semibold text-white">
          Popular
        </span>
      )}

      <h3 className="font-heading text-lg font-bold">
        {name}
      </h3>
      <p className="mt-2 text-3xl font-bold">{price}</p>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        {description}
      </p>

      <ul className="mt-6 flex-1 space-y-2 text-sm text-zinc-700 dark:text-zinc-300">
        {features.map((feature) => (
          <li key={feature} className="flex items-start gap-2">
            <svg
              className="mt-0.5 h-4 w-4 shrink-0 text-accent"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
            {feature}
          </li>
        ))}
      </ul>

      {"onCtaClick" in props ? (
        <button
          type="button"
          onClick={props.onCtaClick}
          disabled={props.ctaDisabled ?? false}
          className={`${ctaClassName} disabled:cursor-not-allowed disabled:opacity-60`}
        >
          {ctaLabel}
        </button>
      ) : (
        <Link href={props.ctaHref} className={ctaClassName}>
          {ctaLabel}
        </Link>
      )}
    </div>
  );
}
