-- Aggregate-only reporting RPC for internal metrics.
--
-- The local reporting role is intentionally not granted broad table access:
-- RLS should keep raw user/video rows private. This SECURITY DEFINER function
-- exposes only coarse aggregate counts needed for standups and funnel checks.

create or replace function public.reporting_metrics(p_days integer default 7)
returns jsonb
language sql
security definer
set search_path = public
as $$
with params as (
  select greatest(coalesce(p_days, 7), 1)::text || ' days' as lookback
),
analytics_event_breakdown as (
  select coalesce(
    jsonb_agg(
      jsonb_build_object(
        'event_name', event_name,
        'events', events,
        'users', users
      )
      order by events desc, event_name
    ),
    '[]'::jsonb
  ) as rows
  from (
    select event_name, count(*) as events, count(distinct user_id) as users
    from public.analytics_events, params
    where created_at >= now() - params.lookback::interval
    group by event_name
    order by events desc, event_name
    limit 20
  ) t
),
video_breakdown as (
  select coalesce(
    jsonb_agg(
      jsonb_build_object(
        'status', status,
        'source_type', source_type,
        'videos', videos,
        'users', users,
        'newest', newest
      )
      order by newest desc nulls last, videos desc
    ),
    '[]'::jsonb
  ) as rows
  from (
    select
      status,
      coalesce(source_type, 'unknown') as source_type,
      count(*) as videos,
      count(distinct user_id) as users,
      max(created_at) as newest
    from public.videos, params
    where created_at >= now() - params.lookback::interval
    group by status, coalesce(source_type, 'unknown')
  ) t
),
job_breakdown as (
  select coalesce(
    jsonb_agg(
      jsonb_build_object(
        'status', status,
        'jobs', jobs,
        'avg_processing_s', avg_processing_s,
        'newest', newest
      )
      order by newest desc nulls last, jobs desc
    ),
    '[]'::jsonb
  ) as rows
  from (
    select
      status,
      count(*) as jobs,
      round(avg(extract(epoch from (completed_at - started_at))) filter (
        where completed_at is not null and started_at is not null
      )) as avg_processing_s,
      max(created_at) as newest
    from public.video_jobs, params
    where created_at >= now() - params.lookback::interval
    group by status
  ) t
),
api_key_summary as (
  select jsonb_build_object(
    'created', count(*) filter (where created_at >= now() - params.lookback::interval),
    'used', count(*) filter (where last_used_at >= now() - params.lookback::interval),
    'users_created', count(distinct user_id) filter (where created_at >= now() - params.lookback::interval),
    'users_used', count(distinct user_id) filter (where last_used_at >= now() - params.lookback::interval)
  ) as row
  from public.api_keys, params
),
api_usage_breakdown as (
  select coalesce(
    jsonb_agg(
      jsonb_build_object(
        'event_type', event_type,
        'events', events,
        'units', units,
        'users', users
      )
      order by events desc, event_type
    ),
    '[]'::jsonb
  ) as rows
  from (
    select event_type, count(*) as events, coalesce(sum(units), 0) as units, count(distinct user_id) as users
    from public.api_usage_events, params
    where created_at >= now() - params.lookback::interval
    group by event_type
  ) t
),
billing_breakdown as (
  select coalesce(
    jsonb_agg(
      jsonb_build_object(
        'event_type', event_type,
        'events', events,
        'credits', credits,
        'users', users
      )
      order by events desc, event_type
    ),
    '[]'::jsonb
  ) as rows
  from (
    select event_type, count(*) as events, coalesce(sum(credits), 0) as credits, count(distinct user_id) as users
    from public.billing_credit_events, params
    where created_at >= now() - params.lookback::interval
    group by event_type
  ) t
)
select jsonb_build_object(
  'window_days', greatest(coalesce(p_days, 7), 1),
  'generated_at', now(),
  'analytics_events', (select rows from analytics_event_breakdown),
  'videos', (select rows from video_breakdown),
  'video_jobs', (select rows from job_breakdown),
  'api_keys', (select row from api_key_summary),
  'api_usage', (select rows from api_usage_breakdown),
  'billing_credit_events', (select rows from billing_breakdown)
);
$$;

revoke all on function public.reporting_metrics(integer) from public;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'fret_readonly') then
    execute 'grant execute on function public.reporting_metrics(integer) to fret_readonly';
  end if;
end
$$;

grant execute on function public.reporting_metrics(integer) to service_role;
