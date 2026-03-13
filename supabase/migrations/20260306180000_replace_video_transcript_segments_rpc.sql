-- Atomically replace transcript segments for one video.

create or replace function public.replace_video_transcript_segments(
  p_video_id uuid,
  p_segments jsonb default '[]'::jsonb
)
returns integer
language plpgsql
set search_path = public
as $$
declare
  normalized_segments jsonb;
  inserted_count integer;
begin
  if p_video_id is null then
    raise exception 'p_video_id must be non-null';
  end if;

  normalized_segments := coalesce(p_segments, '[]'::jsonb);
  if jsonb_typeof(normalized_segments) <> 'array' then
    raise exception 'p_segments must be a JSON array';
  end if;

  delete from public.video_transcript_segments
  where video_id = p_video_id;

  insert into public.video_transcript_segments (
    video_id,
    segment_index,
    start_s,
    end_s,
    text,
    language_code
  )
  select
    p_video_id,
    segment_index,
    start_s,
    end_s,
    text,
    language_code
  from jsonb_to_recordset(normalized_segments) as segment(
    segment_index integer,
    start_s double precision,
    end_s double precision,
    text text,
    language_code text
  );

  get diagnostics inserted_count = row_count;
  return inserted_count;
end;
$$;
