-- Add source metadata for uploaded videos (production ingest path).

alter table public.videos add column if not exists source_type text;
alter table public.videos add column if not exists source_r2_key text;
alter table public.videos add column if not exists source_filename text;

alter table public.videos alter column youtube_url drop not null;

update public.videos set source_type = 'youtube' where source_type is null;

alter table public.videos alter column source_type set default 'youtube';
alter table public.videos alter column source_type set not null;

alter table public.videos
  drop constraint if exists videos_source_check;
alter table public.videos
  add constraint videos_source_check
  check (
    (source_type = 'youtube' and youtube_url is not null)
    or (source_type = 'upload' and source_r2_key is not null)
  );
