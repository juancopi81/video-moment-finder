-- Security hardening: explicit function search_path + RLS policies.

create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

alter table public.videos enable row level security;
alter table public.credits enable row level security;
alter table public.video_jobs enable row level security;

drop policy if exists videos_owner_select on public.videos;
drop policy if exists videos_owner_insert on public.videos;
drop policy if exists videos_owner_update on public.videos;
drop policy if exists videos_owner_delete on public.videos;
drop policy if exists videos_service_role_all on public.videos;

create policy videos_owner_select
on public.videos
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

create policy videos_owner_insert
on public.videos
for insert
to authenticated
with check (user_id = (auth.jwt() ->> 'sub'));

create policy videos_owner_update
on public.videos
for update
to authenticated
using (user_id = (auth.jwt() ->> 'sub'))
with check (user_id = (auth.jwt() ->> 'sub'));

create policy videos_owner_delete
on public.videos
for delete
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

create policy videos_service_role_all
on public.videos
for all
to service_role
using (true)
with check (true);

drop policy if exists credits_owner_select on public.credits;
drop policy if exists credits_owner_insert on public.credits;
drop policy if exists credits_owner_update on public.credits;
drop policy if exists credits_owner_delete on public.credits;
drop policy if exists credits_service_role_all on public.credits;

create policy credits_owner_select
on public.credits
for select
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

create policy credits_owner_insert
on public.credits
for insert
to authenticated
with check (user_id = (auth.jwt() ->> 'sub'));

create policy credits_owner_update
on public.credits
for update
to authenticated
using (user_id = (auth.jwt() ->> 'sub'))
with check (user_id = (auth.jwt() ->> 'sub'));

create policy credits_owner_delete
on public.credits
for delete
to authenticated
using (user_id = (auth.jwt() ->> 'sub'));

create policy credits_service_role_all
on public.credits
for all
to service_role
using (true)
with check (true);

drop policy if exists video_jobs_owner_select on public.video_jobs;
drop policy if exists video_jobs_owner_insert on public.video_jobs;
drop policy if exists video_jobs_owner_update on public.video_jobs;
drop policy if exists video_jobs_owner_delete on public.video_jobs;
drop policy if exists video_jobs_service_role_all on public.video_jobs;

create policy video_jobs_owner_select
on public.video_jobs
for select
to authenticated
using (
  exists (
    select 1
    from public.videos v
    where v.id = video_jobs.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_jobs_owner_insert
on public.video_jobs
for insert
to authenticated
with check (
  exists (
    select 1
    from public.videos v
    where v.id = video_jobs.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_jobs_owner_update
on public.video_jobs
for update
to authenticated
using (
  exists (
    select 1
    from public.videos v
    where v.id = video_jobs.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
)
with check (
  exists (
    select 1
    from public.videos v
    where v.id = video_jobs.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_jobs_owner_delete
on public.video_jobs
for delete
to authenticated
using (
  exists (
    select 1
    from public.videos v
    where v.id = video_jobs.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_jobs_service_role_all
on public.video_jobs
for all
to service_role
using (true)
with check (true);
