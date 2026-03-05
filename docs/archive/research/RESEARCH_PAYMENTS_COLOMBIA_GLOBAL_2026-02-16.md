# Payments Research: Colombia-Based SaaS Selling Globally

Last updated: 2026-02-16

## Goal

Document realistic payment options for a founder based in Colombia who wants to accept payments from users worldwide (for example Europe, US, LATAM), and identify the easiest and most standard path.

## Executive Summary

- Direct Stripe account creation in Colombia is not currently available for standard Stripe payments accounts, based on Stripe's global availability list.
- The fastest standard way to launch globally from Colombia is usually a Merchant of Record (MoR) platform (for example Paddle, Lemon Squeezy, FastSpring, or 2Checkout/Verifone), because tax/compliance burden is shifted to the provider.
- A local PSP-first path (Wompi or PayU) can work for Colombia + some international cards, but global tax, global payment method breadth, and subscription maturity can be harder.
- A US-entity path (for example Stripe Atlas + Stripe) is a strong long-term option, but has setup costs and legal/accounting overhead.
- Wise is helpful for payouts and FX operations, but by itself does not make Stripe payments account availability in Colombia possible.

## Constraints and Product Requirements

### Hard Constraints

- Founder/operator is based in Colombia.
- Product is SaaS and needs recurring billing support.
- Customers can be anywhere globally.

### Desired Capabilities

- Global card acceptance plus local payment methods where possible.
- Subscription lifecycle support: trials, upgrades/downgrades, dunning, retries, cancellations.
- Tax handling for cross-border digital services (especially VAT/GST).
- Reliable payouts to Colombia-based accounts (or Wise-backed account details where accepted).
- Reasonable fraud/chargeback tooling.

## Option 1: Merchant of Record (MoR) Platforms

### Why this is usually the easiest global launch path

- MoR providers generally collect/remit indirect taxes and act as seller of record for checkout transactions.
- This removes a large part of global tax and compliance work from your team.
- Integration is typically a hosted checkout + webhook model, which is fast to implement.

Inference: For a small SaaS team in an unsupported Stripe country, MoR is often the shortest path to "start charging globally" with manageable legal and operational overhead.

### 1A) Paddle

Key facts:
- Managed Payments pricing lists a base fee plus extra fees for international cards, PayPal, and subscription payments.
- Seller onboarding is country-restricted via an unsupported-country list (Colombia is not listed there).
- Payouts are supported via Stripe or PayPal, with fee details documented.

Notes:
- Strong fit if you want explicit MoR + subscription-oriented workflows.
- Confirm final payout rail and fee impact for Colombia before committing.

Sources:
- [Paddle pricing](https://www.paddle.com/pricing)
- [Paddle supported countries](https://www.paddle.com/help/sell/getting-started/supported-countries-and-entities)
- [Paddle payout methods and fees](https://www.paddle.com/help/manage/get-paid/how-do-i-get-paid)

### 1B) Lemon Squeezy (by Stripe)

Key facts:
- Positions itself as Merchant of Record.
- Documentation states support for businesses in many countries, including Colombia.
- Supports subscriptions with interval options and billing controls.
- Payouts documented via Stripe/PayPal with thresholds and payout-fee details.

Notes:
- Strong "launch quickly" option for software products with recurring billing.
- Verify any current onboarding limits and reserve policies during signup review.

Sources:
- [Lemon Squeezy Merchant of Record model](https://docs.lemonsqueezy.com/help/getting-started/merchant-of-record)
- [Lemon Squeezy supported countries](https://docs.lemonsqueezy.com/help/getting-started/supported-countries)
- [Lemon Squeezy subscriptions](https://docs.lemonsqueezy.com/help/products/subscriptions)
- [Lemon Squeezy getting paid](https://docs.lemonsqueezy.com/help/getting-paid/getting-paid)

### 1C) FastSpring

Key facts:
- Marketed as Merchant of Record for software/SaaS.
- Promotes broad global methods/currencies and subscription billing support.
- Has seller-country payout restrictions listed in docs.

Notes:
- Enterprise-friendly and globally focused; implementation effort can be moderate.
- Validate Colombia payout support and contract terms early.

Sources:
- [FastSpring documentation](https://fastspring.com/docs/)
- [FastSpring support article (payout country restrictions)](https://fastspring.com/docs/classic/what-countries-can-i-pay-out-to/)
- [FastSpring support article (supported currencies)](https://fastspring.com/docs/classic/what-currencies-can-i-use-for-pricing-and-payouts/)

### 1D) 2Checkout / Verifone

Key facts:
- Positions itself as a global processor with broad country/method coverage.
- Documentation includes supported seller country lists (including Colombia on the referenced list).

Notes:
- Useful MoR/processor candidate when evaluating alternatives to Paddle/Lemon/FastSpring.
- Verify exact subscription/tax/reseller-of-record model for your plan before selection.

Sources:
- [2Checkout features](https://verifone.cloud/2checkout/features/)
- [2Checkout supported seller countries](https://verifone.cloud/docs/2checkout/Documentation/01Start_here/01Onboarding_and_set_up/Supported_seller_countries)

## Option 2: Colombia-First Direct PSP Path

### 2A) Wompi (Colombia)

Key facts:
- Supports multiple payment methods in Colombia and card rails.
- Offers recurring flow capabilities through payment sources (3RI flow).
- Public integration docs and plugin pages emphasize COP-centric operations.

Notes:
- Good for local launch and Colombia payment familiarity.
- For worldwide SaaS acquisition, validate acceptance rates for non-LATAM cards and non-COP buyer experience.
- You remain responsible for global tax/compliance stack (unless paired with external tax tooling).

Sources:
- [Wompi plugins overview](https://wompi.com/en/co/plugins)
- [Wompi recurring payments guidance](https://docs.wompi.co/en/docs/colombia/pagos-recurrentes/)
- [Wompi docs](https://docs.wompi.co/en/docs/colombia/)

### 2B) PayU (Colombia integration)

Key facts:
- Payments API for Colombia covers cards, wallets, cash, and bank transfer methods.
- Tokenization is available for storing/reusing cards.
- Legacy recurring API docs mark recurring methods as deprecated in that module.

Notes:
- Viable direct-processor path if you want local coverage plus selected international methods.
- Subscription lifecycle may require newer product paths or custom retry/dunning logic depending on plan.
- Like Wompi path, tax/compliance obligations remain on your side.

Sources:
- [PayU Payments API (Colombia)](https://developers.payulatam.com/latam/en/docs/integrations/api-integration/payments-api-colombia.html)
- [PayU tokenization](https://developers.payulatam.com/latam/en/docs/integrations/api-integration/payments-api-colombia.html#operation/TOKENIZATION)
- [PayU recurring docs (legacy/deprecated note)](https://developers.payulatam.com/latam/en/docs/getting-started/technical-documentation/recurring-payments.html)

### 2C) PayPal Checkout (direct)

Key facts:
- Colombia-facing PayPal materials position it for selling internationally.
- Fee schedules for international sales and withdrawals are documented.

Notes:
- Useful as a fallback/add-on channel, especially where buyer preference is PayPal.
- Card acceptance and subscription ergonomics can be less unified than full MoR stacks.

Sources:
- [PayPal Colombia business fee page](https://www.paypal.com/co/business/paypal-business-fees)
- [PayPal Subscriptions overview](https://developer.paypal.com/docs/subscriptions/)

## Option 3: US Entity + Stripe (Atlas Path)

Key facts:
- Stripe Atlas documentation covers incorporation workflow and Stripe onboarding for Atlas companies.
- Stripe Atlas pricing is publicly listed (setup + annual state filing handling fee).
- Atlas docs emphasize global charging capabilities once the Stripe account is set up.

Notes:
- This is the closest path to a "native Stripe stack" when local Stripe account availability is blocked.
- Expect legal/accounting overhead and ongoing compliance obligations beyond payments integration.
- Wise can help with banking/payout operations, but does not replace entity/eligibility requirements.

Sources:
- [Stripe global availability](https://stripe.com/global)
- [Stripe Atlas docs](https://docs.stripe.com/atlas)
- [Stripe Atlas pricing](https://stripe.com/atlas/pricing)
- [Atlas: accept payments](https://docs.stripe.com/atlas/accept-payments)
- [Wise get paid page (platform payout compatibility guidance)](https://wise.com/gb/business/getpaid)

## Comparison Matrix (Practical View)

Scoring scale: 1 (weak) to 5 (strong), based on documented capabilities plus implementation inference for this repo.

| Option | Time to first charge | Global buyer coverage | Subscription support | Tax/compliance burden on us | Ops complexity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Paddle (MoR) | 4 | 4 | 4 | 5 | 3 | Strong standard SaaS path from unsupported Stripe countries |
| Lemon Squeezy (MoR) | 5 | 4 | 4 | 5 | 2 | Very fast implementation profile for small SaaS teams |
| FastSpring (MoR) | 3 | 4 | 4 | 5 | 3 | Strong global model; onboarding/contract fit should be validated |
| 2Checkout/Verifone | 3 | 4 | 3 | 4 | 3 | Broad coverage; verify final MoR/tax behavior in chosen plan |
| Wompi direct | 4 | 2 | 3 | 1 | 3 | Great Colombia fit; weaker as sole global stack |
| PayU direct | 3 | 3 | 2 | 1 | 4 | Broad local methods; recurring path needs careful product validation |
| PayPal direct | 4 | 3 | 3 | 1 | 3 | Good supplemental channel, less complete as sole billing core |
| US entity + Stripe | 2 | 5 | 5 | 2 | 5 | Powerful long-term stack, highest legal/operational overhead initially |

Inference caveat:
- Scores are decision support, not provider guarantees. Confirm with trial onboarding + sandbox + pricing review.

## Recommended Path for This Repository

### Recommendation (standard + easiest)

1. Choose a MoR-first path now (Paddle or Lemon Squeezy first shortlist).
2. Keep direct local PSP (Wompi/PayU) as optional phase-two optimization for Colombia/LATAM conversion if needed.
3. Revisit US entity + Stripe after product traction if deeper Stripe ecosystem control is needed.

Why:
- Fastest way to start charging global users from Colombia with lower legal/tax friction.
- Keeps engineering scope small for Phase 4.2.
- Avoids blocking launch on cross-border corporate setup.

## Due Diligence Checklist Before Committing

- Confirm provider explicitly supports your legal entity profile and payout destination.
- Verify payouts to your current bank/Wise details and all payout fees.
- Validate subscription events needed by backend (`created`, `renewed`, `failed`, `canceled`, `refunded`, `chargeback`).
- Validate webhook retry semantics and signature verification.
- Confirm support for EU SCA/3DS paths and fraud tooling.
- Confirm VAT/GST handling boundaries (who is responsible for filing in each model).
- Run a full sandbox flow: checkout -> webhook -> credit grant -> cancellation/refund.
- Validate buyer experience in US + EU + Colombia test scenarios.

## Implementation Impact on Current Codebase

Phase 4.2 should add:
- Payment provider integration module (hosted checkout session creation).
- Webhook endpoint with signature verification and idempotency.
- Credit ledger updates tied to payment status.
- Subscription state model if recurring credits are offered.

This repo should avoid provider lock-in by defining a small internal billing interface (provider adapter pattern) before wiring a single provider.

## Sources (Primary)

- [Stripe global availability](https://stripe.com/global)
- [Stripe Atlas documentation](https://docs.stripe.com/atlas)
- [Stripe Atlas pricing](https://stripe.com/atlas/pricing)
- [Stripe Atlas accept payments](https://docs.stripe.com/atlas/accept-payments)
- [Paddle pricing](https://www.paddle.com/pricing)
- [Paddle supported countries](https://www.paddle.com/help/sell/getting-started/supported-countries-and-entities)
- [Paddle payout methods](https://www.paddle.com/help/manage/get-paid/how-do-i-get-paid)
- [Lemon Squeezy Merchant of Record](https://docs.lemonsqueezy.com/help/getting-started/merchant-of-record)
- [Lemon Squeezy supported countries](https://docs.lemonsqueezy.com/help/getting-started/supported-countries)
- [Lemon Squeezy subscriptions](https://docs.lemonsqueezy.com/help/products/subscriptions)
- [Lemon Squeezy getting paid](https://docs.lemonsqueezy.com/help/getting-paid/getting-paid)
- [FastSpring docs](https://fastspring.com/docs/)
- [FastSpring payout country support](https://fastspring.com/docs/classic/what-countries-can-i-pay-out-to/)
- [FastSpring currency support](https://fastspring.com/docs/classic/what-currencies-can-i-use-for-pricing-and-payouts/)
- [2Checkout features](https://verifone.cloud/2checkout/features/)
- [2Checkout supported seller countries](https://verifone.cloud/docs/2checkout/Documentation/01Start_here/01Onboarding_and_set_up/Supported_seller_countries)
- [Wompi plugins](https://wompi.com/en/co/plugins)
- [Wompi recurring payments docs](https://docs.wompi.co/en/docs/colombia/pagos-recurrentes/)
- [Wompi docs landing](https://docs.wompi.co/en/docs/colombia/)
- [PayU Payments API Colombia](https://developers.payulatam.com/latam/en/docs/integrations/api-integration/payments-api-colombia.html)
- [PayU recurring payments docs](https://developers.payulatam.com/latam/en/docs/getting-started/technical-documentation/recurring-payments.html)
- [PayPal Colombia business fees](https://www.paypal.com/co/business/paypal-business-fees)
- [PayPal subscriptions docs](https://developer.paypal.com/docs/subscriptions/)
- [Wise get paid (platform payout guidance)](https://wise.com/gb/business/getpaid)

## Non-Legal Advice Disclaimer

This is product/implementation research, not legal or tax advice. Before launch, validate tax and corporate implications with a Colombia-qualified accountant/tax advisor.
