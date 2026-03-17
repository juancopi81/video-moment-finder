-- Prevent double billing on retried v1 video submissions.
--
-- YouTube: unique partial index on (user_id, youtube_url) for non-failed videos
-- ensures that concurrent INSERT attempts for the same URL are serialized by
-- Postgres, and only one wins.
--
-- Uploads with Idempotency-Key: the existing PK on videos.id already
-- serializes concurrent INSERTs when the caller supplies a deterministic UUID.

-- Historical cleanup: before this migration, duplicate active YouTube rows
-- could exist for the same authenticated user. Keep the "best" active row
-- (ready > processing > queued, then newest) and mark the rest failed so the
-- unique index can be created safely on existing databases.
with ranked_active_youtube_rows as (
  select
    id,
    row_number() over (
      partition by user_id, youtube_url
      order by
        case status
          when 'ready' then 0
          when 'processing' then 1
          when 'queued' then 2
          else 3
        end,
        created_at desc,
        id desc
    ) as rn
  from public.videos
  where user_id is not null
    and youtube_url is not null
    and status != 'failed'
)
update public.videos v
set status = 'failed',
    error_message = case
      when coalesce(v.error_message, '') = ''
        then 'Deduplicated before videos_user_youtube_url_active_unique migration'
      else v.error_message || E'\nDeduplicated before videos_user_youtube_url_active_unique migration'
    end
from ranked_active_youtube_rows r
where v.id = r.id
  and r.rn > 1;

-- YouTube URL dedup index.
create unique index if not exists videos_user_youtube_url_active_unique
  on public.videos (user_id, youtube_url)
  where status != 'failed' and youtube_url is not null;

-- Atomic insert-or-get for YouTube video creation.
-- Returns the video row and whether it was newly created.
create or replace function public.insert_youtube_video_idempotent(
  p_youtube_url text,
  p_user_id text,
  out row_data jsonb,
  out was_created boolean
)
language plpgsql
set search_path = public
as $$
declare
  r record;
begin
  insert into public.videos (youtube_url, status, source_type, user_id)
  values (p_youtube_url, 'queued', 'youtube', p_user_id)
  on conflict (user_id, youtube_url)
     where status != 'failed' and youtube_url is not null
  do nothing
  returning * into r;

  if found then
    row_data := to_jsonb(r);
    was_created := true;
    return;
  end if;

  select to_jsonb(v.*) into row_data
    from public.videos v
   where v.user_id = p_user_id
     and v.youtube_url = p_youtube_url
     and v.status != 'failed'
   order by v.created_at desc
   limit 1;

  was_created := false;
end;
$$;

-- Atomic insert-or-get for uploaded video with a predetermined ID.
-- The caller derives the ID from an Idempotency-Key so retries collide on PK.
create or replace function public.insert_uploaded_video_idempotent(
  p_video_id uuid,
  p_user_id text,
  p_source_r2_key text,
  p_source_filename text,
  out row_data jsonb,
  out was_created boolean
)
language plpgsql
set search_path = public
as $$
declare
  r record;
  existing_user_id text;
begin
  insert into public.videos (id, status, source_type, user_id, source_r2_key, source_filename)
  values (p_video_id, 'queued', 'upload', p_user_id, p_source_r2_key, p_source_filename)
  on conflict (id) do nothing
  returning * into r;

  if found then
    row_data := to_jsonb(r);
    was_created := true;
    return;
  end if;

  select to_jsonb(v.*), v.user_id into row_data, existing_user_id
    from public.videos v
   where v.id = p_video_id;

  if existing_user_id != p_user_id then
    raise exception 'Video ID conflict with different user';
  end if;

  was_created := false;
end;
$$;
