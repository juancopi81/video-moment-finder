-- Native reference-image searches use the existing atomic debit/refund path.
-- Preserve every existing event type; no balances or grant settings change.
alter table public.api_usage_events
  drop constraint if exists api_usage_events_event_type_valid;

alter table public.api_usage_events
  add constraint api_usage_events_event_type_valid
  check (event_type in (
    'index_video', 'text_query', 'image_query', 'compensation',
    'transcript_fetch', 'frames_thumb', 'frames_high'
  ));
