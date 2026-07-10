-- Persist probed source duration on videos so frame-clamping can use the
-- actual video length instead of only the global VIDEO_MAX_DURATION_S
-- ceiling. Populated at upload admission on both upload paths (direct
-- multipart upload and presigned-upload-complete), which already ffprobe
-- the duration before insert. Nullable: older rows and YouTube-sourced
-- videos (duration validated via yt-dlp metadata, not ffprobe) leave this
-- null and continue to fall back to the global cap.

alter table public.videos
  add column if not exists duration_s double precision;

-- ---------------------------------------------------------------------------
-- RPC: insert_uploaded_video_idempotent — extended to accept the probed
-- duration. p_duration_s defaults to null so callers that omit it keep
-- working. The old 4-arg signature must be dropped first: CREATE OR REPLACE
-- only replaces a function with an identical input signature, so without the
-- drop this would create a second overload and make RPC calls that omit
-- p_duration_s ambiguous ("could not choose the best candidate function").
-- ---------------------------------------------------------------------------

drop function if exists public.insert_uploaded_video_idempotent(uuid, text, text, text);

create or replace function public.insert_uploaded_video_idempotent(
  p_video_id uuid,
  p_user_id text,
  p_source_r2_key text,
  p_source_filename text,
  p_duration_s double precision default null,
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
  insert into public.videos (
    id, status, source_type, user_id, source_r2_key, source_filename, duration_s
  )
  values (
    p_video_id, 'queued', 'upload', p_user_id, p_source_r2_key, p_source_filename, p_duration_s
  )
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
