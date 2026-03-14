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

-- Atomic key creation with per-user cap (avoids count-then-insert race).
create or replace function public.create_api_key_atomic(
  p_user_id text,
  p_name text,
  p_key_hash text,
  p_key_prefix text,
  p_max_keys integer default 10
)
returns uuid
language plpgsql
set search_path = public
as $$
declare
  active_count integer;
  new_id uuid;
begin
  -- Lock active keys for this user to prevent concurrent inserts.
  select count(*) into active_count
    from public.api_keys
   where user_id = p_user_id
     and revoked_at is null
     for update;

  if active_count >= p_max_keys then
    raise exception 'Maximum of % active API keys per user', p_max_keys;
  end if;

  insert into public.api_keys (user_id, name, key_hash, key_prefix)
  values (p_user_id, p_name, p_key_hash, p_key_prefix)
  returning id into new_id;

  return new_id;
end;
$$;
