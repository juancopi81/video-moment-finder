-- API billing: separate credit pool, billing events, and usage ledger for API access.

-- ---------------------------------------------------------------------------
-- api_credits — parallel to credits, unit-based
-- ---------------------------------------------------------------------------

create table if not exists public.api_credits (
  id uuid primary key default gen_random_uuid(),
  user_id text not null unique,
  balance integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'api_credits_balance_non_negative'
      and conrelid = 'public.api_credits'::regclass
  ) then
    alter table public.api_credits
      add constraint api_credits_balance_non_negative
      check (balance >= 0);
  end if;
end
$$;

-- Reuse the existing set_updated_at trigger function.
drop trigger if exists api_credits_set_updated_at on public.api_credits;
create trigger api_credits_set_updated_at
  before update on public.api_credits
  for each row
  execute function public.set_updated_at();

alter table public.api_credits enable row level security;

drop policy if exists api_credits_service_role_all on public.api_credits;
create policy api_credits_service_role_all
on public.api_credits
for all
to service_role
using (true)
with check (true);

drop policy if exists api_credits_owner_select on public.api_credits;
create policy api_credits_owner_select
on public.api_credits
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

-- ---------------------------------------------------------------------------
-- api_billing_events — parallel to billing_credit_events
-- ---------------------------------------------------------------------------

create table if not exists public.api_billing_events (
  id uuid primary key default gen_random_uuid(),
  provider text not null,
  event_id text not null,
  event_type text not null,
  user_id text not null,
  credits integer not null,
  grant_target text not null default 'api',
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'api_billing_events_credits_positive'
      and conrelid = 'public.api_billing_events'::regclass
  ) then
    alter table public.api_billing_events
      add constraint api_billing_events_credits_positive
      check (credits > 0);
  end if;

  if not exists (
    select 1
    from pg_constraint
    where conname = 'api_billing_events_grant_target_valid'
      and conrelid = 'public.api_billing_events'::regclass
  ) then
    alter table public.api_billing_events
      add constraint api_billing_events_grant_target_valid
      check (grant_target in ('web', 'api'));
  end if;
end
$$;

create unique index if not exists api_billing_events_provider_event_id_unique
  on public.api_billing_events (provider, event_id);

create index if not exists api_billing_events_user_id_created_at_idx
  on public.api_billing_events (user_id, created_at desc);

alter table public.api_billing_events enable row level security;

drop policy if exists api_billing_events_service_role_all on public.api_billing_events;
create policy api_billing_events_service_role_all
on public.api_billing_events
for all
to service_role
using (true)
with check (true);

drop policy if exists api_billing_events_owner_select on public.api_billing_events;
create policy api_billing_events_owner_select
on public.api_billing_events
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

-- ---------------------------------------------------------------------------
-- api_usage_events — usage ledger
-- ---------------------------------------------------------------------------

create table if not exists public.api_usage_events (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  api_key_id uuid references public.api_keys(id),
  event_type text not null,
  units integer not null,
  video_id uuid,
  request_id text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'api_usage_events_event_type_valid'
      and conrelid = 'public.api_usage_events'::regclass
  ) then
    alter table public.api_usage_events
      add constraint api_usage_events_event_type_valid
      check (event_type in ('index_video', 'text_query', 'compensation'));
  end if;
end
$$;

create index if not exists api_usage_events_user_id_created_at_idx
  on public.api_usage_events (user_id, created_at desc);

create index if not exists api_usage_events_api_key_id_created_at_idx
  on public.api_usage_events (api_key_id, created_at desc);

create unique index if not exists api_usage_events_request_id_unique
  on public.api_usage_events (request_id) where request_id is not null;

alter table public.api_usage_events enable row level security;

drop policy if exists api_usage_events_service_role_all on public.api_usage_events;
create policy api_usage_events_service_role_all
on public.api_usage_events
for all
to service_role
using (true)
with check (true);

drop policy if exists api_usage_events_owner_select on public.api_usage_events;
create policy api_usage_events_owner_select
on public.api_usage_events
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

-- ---------------------------------------------------------------------------
-- RPC: apply_api_credit_grant — idempotent credit grant for API billing
-- ---------------------------------------------------------------------------

create or replace function public.apply_api_credit_grant(
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

  insert into public.api_billing_events (
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

  insert into public.api_credits (user_id, balance)
  values (p_user_id, p_credits)
  on conflict (user_id)
  do update set balance = public.api_credits.balance + excluded.balance;

  return true;
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: consume_api_units — atomic unit deduction with idempotency
-- ---------------------------------------------------------------------------

create or replace function public.consume_api_units(
  p_user_id text,
  p_api_key_id uuid,
  p_event_type text,
  p_units integer,
  p_video_id uuid default null,
  p_request_id text default null,
  p_metadata jsonb default '{}'::jsonb
)
returns table (
  allowed boolean,
  remaining_balance integer
)
language plpgsql
set search_path = public
as $$
declare
  updated_balance integer;
  current_balance integer;
begin
  if p_user_id is null or btrim(p_user_id) = '' then
    raise exception 'p_user_id must be non-empty';
  end if;
  if p_units <= 0 then
    raise exception 'p_units must be > 0';
  end if;

  -- Idempotency: if request_id already exists, return the previous result
  -- without re-deducting.
  if p_request_id is not null then
    if exists (
      select 1 from public.api_usage_events
      where request_id = p_request_id
    ) then
      select balance into current_balance
      from public.api_credits
      where user_id = p_user_id;

      return query select true, coalesce(current_balance, 0);
      return;
    end if;
  end if;

  -- Attempt atomic deduction.
  update public.api_credits
  set balance = balance - p_units
  where user_id = p_user_id
    and balance >= p_units
  returning balance into updated_balance;

  if found then
    insert into public.api_usage_events (
      user_id, api_key_id, event_type, units, video_id, request_id, metadata
    )
    values (
      p_user_id, p_api_key_id, p_event_type, p_units,
      p_video_id, p_request_id, coalesce(p_metadata, '{}'::jsonb)
    );

    return query select true, updated_balance;
    return;
  end if;

  -- Insufficient balance.
  select balance into current_balance
  from public.api_credits
  where user_id = p_user_id;

  return query select false, coalesce(current_balance, 0);
end;
$$;

-- ---------------------------------------------------------------------------
-- RPC: compensate_api_units — refund units (idempotent on request_id)
-- ---------------------------------------------------------------------------

create or replace function public.compensate_api_units(
  p_user_id text,
  p_units integer,
  p_video_id uuid default null,
  p_request_id text default null,
  p_metadata jsonb default '{}'::jsonb
)
returns void
language plpgsql
set search_path = public
as $$
begin
  if p_user_id is null or btrim(p_user_id) = '' then
    raise exception 'p_user_id must be non-empty';
  end if;
  if p_units <= 0 then
    raise exception 'p_units must be > 0';
  end if;

  -- Idempotency on request_id.
  if p_request_id is not null then
    if exists (
      select 1 from public.api_usage_events
      where request_id = p_request_id
    ) then
      return;
    end if;
  end if;

  -- Increment balance.
  insert into public.api_credits (user_id, balance)
  values (p_user_id, p_units)
  on conflict (user_id)
  do update set balance = public.api_credits.balance + excluded.balance;

  -- Record compensation event with negative units.
  insert into public.api_usage_events (
    user_id, api_key_id, event_type, units, video_id, request_id, metadata
  )
  values (
    p_user_id, null, 'compensation', -p_units,
    p_video_id, p_request_id, coalesce(p_metadata, '{}'::jsonb)
  );
end;
$$;
