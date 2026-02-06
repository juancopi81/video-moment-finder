-- Core schema for Video Moment Finder.
-- Idempotent: safe to apply multiple times.

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.videos (
  id uuid primary key default gen_random_uuid(),
  user_id text,
  youtube_url text not null,
  status text not null default 'queued',
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.videos add column if not exists user_id text;
alter table public.videos add column if not exists youtube_url text;
alter table public.videos add column if not exists status text;
alter table public.videos add column if not exists error_message text;
alter table public.videos add column if not exists created_at timestamptz not null default now();
alter table public.videos add column if not exists updated_at timestamptz not null default now();
alter table public.videos alter column youtube_url set not null;
alter table public.videos alter column status set default 'queued';
update public.videos set status = 'queued' where status is null;
update public.videos set status = 'failed' where status not in ('queued', 'processing', 'ready', 'failed');
alter table public.videos alter column status set not null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'videos_status_check'
      and conrelid = 'public.videos'::regclass
  ) then
    alter table public.videos
      add constraint videos_status_check
      check (status in ('queued', 'processing', 'ready', 'failed'));
  end if;
end
$$;

drop trigger if exists videos_set_updated_at on public.videos;
create trigger videos_set_updated_at
before update on public.videos
for each row
execute function public.set_updated_at();

create table if not exists public.credits (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  balance integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.credits add column if not exists user_id text;
alter table public.credits add column if not exists balance integer;
alter table public.credits add column if not exists created_at timestamptz not null default now();
alter table public.credits add column if not exists updated_at timestamptz not null default now();
update public.credits set balance = 0 where balance is null;
alter table public.credits alter column user_id set not null;
alter table public.credits alter column balance set default 0;
alter table public.credits alter column balance set not null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'credits_balance_non_negative'
      and conrelid = 'public.credits'::regclass
  ) then
    alter table public.credits
      add constraint credits_balance_non_negative
      check (balance >= 0);
  end if;
end
$$;

create unique index if not exists credits_user_id_unique on public.credits (user_id);

drop trigger if exists credits_set_updated_at on public.credits;
create trigger credits_set_updated_at
before update on public.credits
for each row
execute function public.set_updated_at();

create table if not exists public.video_jobs (
  id uuid primary key default gen_random_uuid(),
  video_id uuid not null references public.videos(id) on delete cascade,
  status text not null default 'queued',
  worker_id text,
  attempt_count integer not null default 0,
  error_message text,
  locked_at timestamptz,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.video_jobs add column if not exists video_id uuid;
alter table public.video_jobs add column if not exists status text;
alter table public.video_jobs add column if not exists worker_id text;
alter table public.video_jobs add column if not exists attempt_count integer;
alter table public.video_jobs add column if not exists error_message text;
alter table public.video_jobs add column if not exists locked_at timestamptz;
alter table public.video_jobs add column if not exists started_at timestamptz;
alter table public.video_jobs add column if not exists completed_at timestamptz;
alter table public.video_jobs add column if not exists created_at timestamptz not null default now();
alter table public.video_jobs add column if not exists updated_at timestamptz not null default now();
alter table public.video_jobs alter column status set default 'queued';
update public.video_jobs set status = 'queued' where status is null;
update public.video_jobs set status = 'failed' where status not in ('queued', 'processing', 'completed', 'failed');
alter table public.video_jobs alter column status set not null;
alter table public.video_jobs alter column attempt_count set default 0;
update public.video_jobs set attempt_count = 0 where attempt_count is null;
alter table public.video_jobs alter column attempt_count set not null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'video_jobs_status_check'
      and conrelid = 'public.video_jobs'::regclass
  ) then
    alter table public.video_jobs
      add constraint video_jobs_status_check
      check (status in ('queued', 'processing', 'completed', 'failed'));
  end if;
end
$$;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'video_jobs_attempt_count_non_negative'
      and conrelid = 'public.video_jobs'::regclass
  ) then
    alter table public.video_jobs
      add constraint video_jobs_attempt_count_non_negative
      check (attempt_count >= 0);
  end if;
end
$$;

create unique index if not exists video_jobs_video_id_unique on public.video_jobs (video_id);
create index if not exists video_jobs_status_created_at_idx on public.video_jobs (status, created_at);
create index if not exists videos_status_created_at_idx on public.videos (status, created_at);

drop trigger if exists video_jobs_set_updated_at on public.video_jobs;
create trigger video_jobs_set_updated_at
before update on public.video_jobs
for each row
execute function public.set_updated_at();
