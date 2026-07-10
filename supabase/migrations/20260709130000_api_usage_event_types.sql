-- Allow the retrieval billing event types introduced with the transcript and
-- frames endpoints. The original constraint whitelisted only the launch-era
-- event types, so billed transcript/frame calls failed with a check violation
-- in any environment that actually writes billing rows (found via a live
-- get_transcript connector call; unit tests mock the DB and could not catch
-- this). Compensation rows reuse the already-allowed 'compensation' type.

alter table public.api_usage_events
  drop constraint if exists api_usage_events_event_type_valid;

alter table public.api_usage_events
  add constraint api_usage_events_event_type_valid
  check (event_type in (
    'index_video',
    'text_query',
    'compensation',
    'transcript_fetch',
    'frames_thumb',
    'frames_high'
  ));
