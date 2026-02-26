-- Per-user overrides for free video limits (admin/support entitlements).

create table if not exists public.video_access_overrides (
  user_id text primary key,
  unlimited_videos boolean not null default false,
  note text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.video_access_overrides add column if not exists user_id text;
alter table public.video_access_overrides add column if not exists unlimited_videos boolean;
alter table public.video_access_overrides add column if not exists note text;
alter table public.video_access_overrides add column if not exists created_at timestamptz not null default now();
alter table public.video_access_overrides add column if not exists updated_at timestamptz not null default now();
update public.video_access_overrides set unlimited_videos = false where unlimited_videos is null;
alter table public.video_access_overrides alter column user_id set not null;
alter table public.video_access_overrides alter column unlimited_videos set default false;
alter table public.video_access_overrides alter column unlimited_videos set not null;

drop trigger if exists video_access_overrides_set_updated_at on public.video_access_overrides;
create trigger video_access_overrides_set_updated_at
before update on public.video_access_overrides
for each row
execute function public.set_updated_at();

create index if not exists video_access_overrides_unlimited_idx
on public.video_access_overrides (unlimited_videos)
where unlimited_videos = true;

alter table public.video_access_overrides enable row level security;

drop policy if exists video_access_overrides_service_role_all on public.video_access_overrides;
create policy video_access_overrides_service_role_all
on public.video_access_overrides
for all
to service_role
using (true)
with check (true);

drop policy if exists video_access_overrides_owner_select on public.video_access_overrides;
create policy video_access_overrides_owner_select
on public.video_access_overrides
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));
