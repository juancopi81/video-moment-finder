-- Billing events table + atomic/idempotent credit grants for Lemon Squeezy webhooks.

create table if not exists public.billing_credit_events (
  id uuid primary key default gen_random_uuid(),
  provider text not null,
  event_id text not null,
  event_type text not null,
  user_id text not null,
  credits integer not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

alter table public.billing_credit_events add column if not exists provider text;
alter table public.billing_credit_events add column if not exists event_id text;
alter table public.billing_credit_events add column if not exists event_type text;
alter table public.billing_credit_events add column if not exists user_id text;
alter table public.billing_credit_events add column if not exists credits integer;
alter table public.billing_credit_events add column if not exists payload jsonb not null default '{}'::jsonb;
alter table public.billing_credit_events add column if not exists created_at timestamptz not null default now();
alter table public.billing_credit_events alter column provider set not null;
alter table public.billing_credit_events alter column event_id set not null;
alter table public.billing_credit_events alter column event_type set not null;
alter table public.billing_credit_events alter column user_id set not null;
alter table public.billing_credit_events alter column credits set not null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'billing_credit_events_credits_positive'
      and conrelid = 'public.billing_credit_events'::regclass
  ) then
    alter table public.billing_credit_events
      add constraint billing_credit_events_credits_positive
      check (credits > 0);
  end if;
end
$$;

create unique index if not exists billing_credit_events_provider_event_id_unique
  on public.billing_credit_events (provider, event_id);

create index if not exists billing_credit_events_user_id_created_at_idx
  on public.billing_credit_events (user_id, created_at desc);

alter table public.billing_credit_events enable row level security;

drop policy if exists billing_credit_events_service_role_all on public.billing_credit_events;
create policy billing_credit_events_service_role_all
on public.billing_credit_events
for all
to service_role
using (true)
with check (true);

drop policy if exists billing_credit_events_owner_select on public.billing_credit_events;
create policy billing_credit_events_owner_select
on public.billing_credit_events
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

create or replace function public.apply_billing_credit_grant(
  p_provider text,
  p_event_id text,
  p_event_type text,
  p_user_id text,
  p_credits integer,
  p_payload jsonb default '{}'::jsonb
)
returns boolean
language plpgsql
set search_path = public
as $$
declare
  inserted_count integer;
begin
  if p_credits <= 0 then
    raise exception 'p_credits must be > 0';
  end if;

  insert into public.billing_credit_events (
    provider,
    event_id,
    event_type,
    user_id,
    credits,
    payload
  )
  values (
    p_provider,
    p_event_id,
    p_event_type,
    p_user_id,
    p_credits,
    coalesce(p_payload, '{}'::jsonb)
  )
  on conflict (provider, event_id) do nothing;

  get diagnostics inserted_count = row_count;
  if inserted_count = 0 then
    return false;
  end if;

  insert into public.credits (user_id, balance)
  values (p_user_id, p_credits)
  on conflict (user_id)
  do update set balance = public.credits.balance + excluded.balance;

  return true;
end;
$$;
