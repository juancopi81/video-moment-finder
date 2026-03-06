-- Product analytics events table for first-party behavioral tracking.

create table if not exists public.analytics_events (
  id uuid primary key default gen_random_uuid(),
  event_name text not null,
  user_id text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists analytics_events_event_name_created_at_idx
  on public.analytics_events (event_name, created_at);

create index if not exists analytics_events_user_id_idx
  on public.analytics_events (user_id)
  where user_id is not null;

-- Partial unique index: one signup_complete per user (idempotent).
create unique index if not exists analytics_events_signup_complete_user_unique
  on public.analytics_events (event_name, user_id)
  where event_name = 'signup_complete';

alter table public.analytics_events enable row level security;

drop policy if exists analytics_events_service_role_all on public.analytics_events;
create policy analytics_events_service_role_all
on public.analytics_events
for all
to service_role
using (true)
with check (true);
