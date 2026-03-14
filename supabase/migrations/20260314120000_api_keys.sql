-- API keys table for programmatic authentication.

create table if not exists public.api_keys (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  name text not null default '',
  key_hash text not null,
  key_prefix text not null,
  revoked_at timestamptz,
  last_used_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Auth lookup: find active key by hash.
create unique index if not exists api_keys_key_hash_active_unique
  on public.api_keys (key_hash) where revoked_at is null;

-- List keys for a user, newest first.
create index if not exists api_keys_user_id_created_at_idx
  on public.api_keys (user_id, created_at desc);

alter table public.api_keys enable row level security;

drop policy if exists api_keys_service_role_all on public.api_keys;
create policy api_keys_service_role_all
on public.api_keys
for all
to service_role
using (true)
with check (true);

drop policy if exists api_keys_owner_select on public.api_keys;
create policy api_keys_owner_select
on public.api_keys
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

drop policy if exists api_keys_owner_delete on public.api_keys;
create policy api_keys_owner_delete
on public.api_keys
for delete
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));
