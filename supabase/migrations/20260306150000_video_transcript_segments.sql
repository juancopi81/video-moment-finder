-- Store transcript segments for transcript-backed video search.

create table if not exists public.video_transcript_segments (
  id uuid primary key default gen_random_uuid(),
  video_id uuid not null references public.videos(id) on delete cascade,
  segment_index integer not null,
  start_s double precision not null,
  end_s double precision not null,
  text text not null,
  language_code text,
  created_at timestamptz not null default now()
);

create unique index if not exists video_transcript_segments_video_id_segment_index_unique
  on public.video_transcript_segments (video_id, segment_index);

create index if not exists video_transcript_segments_video_id_start_s_idx
  on public.video_transcript_segments (video_id, start_s);

create index if not exists video_transcript_segments_search_vector_idx
  on public.video_transcript_segments
  using gin (to_tsvector('simple', text));

alter table public.video_transcript_segments enable row level security;

drop policy if exists video_transcript_segments_owner_select on public.video_transcript_segments;
drop policy if exists video_transcript_segments_owner_insert on public.video_transcript_segments;
drop policy if exists video_transcript_segments_owner_update on public.video_transcript_segments;
drop policy if exists video_transcript_segments_owner_delete on public.video_transcript_segments;
drop policy if exists video_transcript_segments_service_role_all on public.video_transcript_segments;

create policy video_transcript_segments_owner_select
on public.video_transcript_segments
for select
to authenticated
using (
  exists (
    select 1
    from public.videos v
    where v.id = video_transcript_segments.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_transcript_segments_owner_insert
on public.video_transcript_segments
for insert
to authenticated
with check (
  exists (
    select 1
    from public.videos v
    where v.id = video_transcript_segments.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_transcript_segments_owner_update
on public.video_transcript_segments
for update
to authenticated
using (
  exists (
    select 1
    from public.videos v
    where v.id = video_transcript_segments.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
)
with check (
  exists (
    select 1
    from public.videos v
    where v.id = video_transcript_segments.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_transcript_segments_owner_delete
on public.video_transcript_segments
for delete
to authenticated
using (
  exists (
    select 1
    from public.videos v
    where v.id = video_transcript_segments.video_id
      and v.user_id = (auth.jwt() ->> 'sub')
  )
);

create policy video_transcript_segments_service_role_all
on public.video_transcript_segments
for all
to service_role
using (true)
with check (true);

create or replace function public.search_video_transcript_segments(
  p_video_id uuid,
  p_query text,
  p_limit integer default 5
)
returns table (
  video_id uuid,
  segment_index integer,
  start_s double precision,
  end_s double precision,
  text text,
  language_code text,
  score real
)
language sql
stable
set search_path = public
as $$
  with query as (
    select websearch_to_tsquery('simple', trim(p_query)) as terms
    where nullif(trim(p_query), '') is not null
  )
  select
    s.video_id,
    s.segment_index,
    s.start_s,
    s.end_s,
    s.text,
    s.language_code,
    ts_rank_cd(to_tsvector('simple', s.text), q.terms)::real as score
  from public.video_transcript_segments s
  cross join query q
  where s.video_id = p_video_id
    and to_tsvector('simple', s.text) @@ q.terms
  order by score desc, s.start_s asc
  limit least(greatest(coalesce(p_limit, 5), 1), 20);
$$;
