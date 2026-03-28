-- OAuth persistence for the Claude MCP connector.

-- ---------------------------------------------------------------------------
-- mcp_oauth_authorization_requests
-- ---------------------------------------------------------------------------

create table if not exists public.mcp_oauth_authorization_requests (
  id uuid primary key default gen_random_uuid(),
  client_id text not null,
  redirect_uri text not null,
  redirect_uri_provided_explicitly boolean not null default false,
  state text,
  scopes text[] not null default array[]::text[],
  code_challenge text not null,
  resource text not null,
  user_id text,
  status text not null default 'pending',
  expires_at timestamptz not null,
  approved_at timestamptz,
  denied_at timestamptz,
  resolved_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'mcp_oauth_authorization_requests_status_valid'
      and conrelid = 'public.mcp_oauth_authorization_requests'::regclass
  ) then
    alter table public.mcp_oauth_authorization_requests
      add constraint mcp_oauth_authorization_requests_status_valid
      check (status in ('pending', 'approved', 'denied'));
  end if;
end
$$;

create index if not exists mcp_oauth_authorization_requests_expires_at_idx
  on public.mcp_oauth_authorization_requests (expires_at);

drop trigger if exists mcp_oauth_authorization_requests_set_updated_at on public.mcp_oauth_authorization_requests;
create trigger mcp_oauth_authorization_requests_set_updated_at
  before update on public.mcp_oauth_authorization_requests
  for each row
  execute function public.set_updated_at();

alter table public.mcp_oauth_authorization_requests enable row level security;

drop policy if exists mcp_oauth_authorization_requests_service_role_all on public.mcp_oauth_authorization_requests;
create policy mcp_oauth_authorization_requests_service_role_all
on public.mcp_oauth_authorization_requests
for all
to service_role
using (true)
with check (true);

-- ---------------------------------------------------------------------------
-- mcp_oauth_authorization_codes
-- ---------------------------------------------------------------------------

create table if not exists public.mcp_oauth_authorization_codes (
  id uuid primary key default gen_random_uuid(),
  authorization_request_id uuid references public.mcp_oauth_authorization_requests(id) on delete set null,
  user_id text not null,
  client_id text not null,
  code_hash text not null,
  redirect_uri text not null,
  redirect_uri_provided_explicitly boolean not null default false,
  scopes text[] not null default array[]::text[],
  code_challenge text not null,
  resource text not null,
  expires_at timestamptz not null,
  used_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz not null default now()
);

create unique index if not exists mcp_oauth_authorization_codes_active_hash_unique
  on public.mcp_oauth_authorization_codes (code_hash)
  where used_at is null and revoked_at is null;

create index if not exists mcp_oauth_authorization_codes_expires_at_idx
  on public.mcp_oauth_authorization_codes (expires_at);

alter table public.mcp_oauth_authorization_codes enable row level security;

drop policy if exists mcp_oauth_authorization_codes_service_role_all on public.mcp_oauth_authorization_codes;
create policy mcp_oauth_authorization_codes_service_role_all
on public.mcp_oauth_authorization_codes
for all
to service_role
using (true)
with check (true);

-- ---------------------------------------------------------------------------
-- mcp_oauth_access_tokens
-- ---------------------------------------------------------------------------

create table if not exists public.mcp_oauth_access_tokens (
  id uuid primary key default gen_random_uuid(),
  connection_id uuid not null,
  user_id text not null,
  client_id text not null,
  token_hash text not null,
  scopes text[] not null default array[]::text[],
  resource text not null,
  expires_at timestamptz not null,
  revoked_at timestamptz,
  created_at timestamptz not null default now()
);

create unique index if not exists mcp_oauth_access_tokens_active_hash_unique
  on public.mcp_oauth_access_tokens (token_hash)
  where revoked_at is null;

create index if not exists mcp_oauth_access_tokens_connection_id_idx
  on public.mcp_oauth_access_tokens (connection_id);

create index if not exists mcp_oauth_access_tokens_expires_at_idx
  on public.mcp_oauth_access_tokens (expires_at);

alter table public.mcp_oauth_access_tokens enable row level security;

drop policy if exists mcp_oauth_access_tokens_service_role_all on public.mcp_oauth_access_tokens;
create policy mcp_oauth_access_tokens_service_role_all
on public.mcp_oauth_access_tokens
for all
to service_role
using (true)
with check (true);

-- ---------------------------------------------------------------------------
-- mcp_oauth_refresh_tokens
-- ---------------------------------------------------------------------------

create table if not exists public.mcp_oauth_refresh_tokens (
  id uuid primary key default gen_random_uuid(),
  connection_id uuid not null,
  user_id text not null,
  client_id text not null,
  token_hash text not null,
  scopes text[] not null default array[]::text[],
  resource text not null,
  expires_at timestamptz not null,
  revoked_at timestamptz,
  created_at timestamptz not null default now()
);

create unique index if not exists mcp_oauth_refresh_tokens_active_hash_unique
  on public.mcp_oauth_refresh_tokens (token_hash)
  where revoked_at is null;

create index if not exists mcp_oauth_refresh_tokens_connection_id_idx
  on public.mcp_oauth_refresh_tokens (connection_id);

create index if not exists mcp_oauth_refresh_tokens_expires_at_idx
  on public.mcp_oauth_refresh_tokens (expires_at);

alter table public.mcp_oauth_refresh_tokens enable row level security;

drop policy if exists mcp_oauth_refresh_tokens_service_role_all on public.mcp_oauth_refresh_tokens;
create policy mcp_oauth_refresh_tokens_service_role_all
on public.mcp_oauth_refresh_tokens
for all
to service_role
using (true)
with check (true);
