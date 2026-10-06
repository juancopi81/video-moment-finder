-- Schema only: this migration does not grant units to any account.
-- API_TRIAL_ENABLED defaults OFF; activation requires a separate operator step.
create table if not exists public.api_trial_grants (
  user_id text primary key,
  verified_email_id text not null,
  allowance_units integer not null check (allowance_units > 0),
  legacy_units_offset integer not null check (legacy_units_offset >= 0),
  granted_units integer not null check (granted_units >= 0),
  created_at timestamptz not null default now(),
  check (granted_units + legacy_units_offset = allowance_units)
);
alter table public.api_trial_grants enable row level security;
drop policy if exists api_trial_grants_service_role_all on public.api_trial_grants;
create policy api_trial_grants_service_role_all on public.api_trial_grants
  for all to service_role using (true) with check (true);
drop policy if exists api_trial_grants_owner_select on public.api_trial_grants;
create policy api_trial_grants_owner_select on public.api_trial_grants
  for select to authenticated using (user_id = (auth.jwt() ->> 'sub'));

create table if not exists public.web_processing_charges (
  video_id uuid primary key references public.videos(id),
  user_id text not null,
  source text not null check (source in ('api_units', 'web_credit')),
  units integer not null check (units > 0),
  created_at timestamptz not null default now()
);
alter table public.web_processing_charges enable row level security;
drop policy if exists web_processing_charges_service_role_all on public.web_processing_charges;
create policy web_processing_charges_service_role_all on public.web_processing_charges
  for all to service_role using (true) with check (true);
drop policy if exists web_processing_charges_owner_select on public.web_processing_charges;
create policy web_processing_charges_owner_select on public.web_processing_charges
  for select to authenticated using (user_id = (auth.jwt() ->> 'sub'));

create or replace function public.apply_api_trial_grant(
  p_user_id text, p_verified_email_id text, p_allowance_units integer,
  p_legacy_free_videos integer, p_index_cost_units integer
)
returns setof public.api_trial_grants
language plpgsql
set search_path = public
as $$
declare
  historical_videos integer;
  unattributed_api_uploads integer;
  legacy_offset integer;
begin
  if p_user_id is null or btrim(p_user_id) = ''
    or p_verified_email_id is null or btrim(p_verified_email_id) = ''
    or p_allowance_units is null or p_allowance_units <= 0
    or p_legacy_free_videos is null or p_legacy_free_videos < 0
    or p_index_cost_units is null or p_index_cost_units <= 0 then
    raise exception 'Invalid verified-account trial parameters';
  end if;
  perform pg_advisory_xact_lock(hashtextextended('api-billing:' || p_user_id, 0));
  if exists (select 1 from public.api_trial_grants where user_id = p_user_id) then
    return query select * from public.api_trial_grants where user_id = p_user_id;
    return;
  end if;

  -- Historical web free usage has no debit ledger. Conservatively infer only
  -- up to the legacy free quota, excluding failed videos, API-funded indexing,
  -- and explicit unlimited accounts. Never modify paid balances or history.
  select count(*) into unattributed_api_uploads from public.api_usage_events u
    where u.user_id = p_user_id and u.event_type = 'index_video'
      and u.video_id is null and u.units > 0
      and not exists (select 1 from public.api_usage_events r
                      where r.request_id = 'compensate:' || u.request_id);
  select count(*) filter (where source_type <> 'upload')
         + greatest(count(*) filter (where source_type = 'upload') - unattributed_api_uploads, 0)
    into historical_videos from (
    select v.id, v.source_type from public.videos v
    where v.user_id = p_user_id and v.status <> 'failed'
      and not exists (select 1 from public.video_access_overrides o
                      where o.user_id = p_user_id and o.unlimited_videos)
      and not exists (select 1 from public.api_usage_events u
                      where u.user_id = p_user_id and u.video_id = v.id
                        and u.event_type = 'index_video' and u.units > 0)
  ) legacy;
  legacy_offset := least(p_allowance_units::bigint,
                        least(historical_videos, p_legacy_free_videos)::bigint * p_index_cost_units)::integer;
  insert into public.api_trial_grants
    (user_id, verified_email_id, allowance_units, legacy_units_offset, granted_units)
  values (p_user_id, p_verified_email_id, p_allowance_units, legacy_offset,
          p_allowance_units - legacy_offset);
  if p_allowance_units > legacy_offset then
    perform public.apply_api_credit_grant(
      'verified_account_trial', p_user_id, 'trial_granted', p_user_id,
      p_allowance_units - legacy_offset,
      jsonb_build_object('allowance_units', p_allowance_units,
                         'legacy_units_offset', legacy_offset));
  end if;
  return query select * from public.api_trial_grants where user_id = p_user_id;
end;
$$;

-- Retry paths may enqueue only after admission is proven. A queued row by
-- itself can also mean the winning request has not completed its debit yet.
create or replace function public.has_video_processing_charge(p_user_id text, p_video_id uuid)
returns boolean
language sql
stable
set search_path = public
as $$
  select exists (select 1 from public.web_processing_charges
                 where user_id = p_user_id and video_id = p_video_id and source = 'web_credit')
    or exists (select 1 from public.api_usage_events u
               where u.user_id = p_user_id and u.video_id = p_video_id
                 and u.event_type = 'index_video' and u.units > 0
                 and not exists (select 1 from public.api_usage_events r
                                 where r.request_id = 'compensate:' || u.request_id));
$$;
revoke all on function public.has_video_processing_charge(text, uuid) from public, anon, authenticated;
grant execute on function public.has_video_processing_charge(text, uuid) to service_role;

create or replace function public.consume_trial_or_processing_credit(
  p_user_id text, p_video_id uuid, p_units integer
)
returns table (allowed boolean, remaining_balance integer)
language plpgsql
set search_path = public
as $$
declare
  api_result record;
  web_balance integer;
begin
  if p_units is null or p_units <= 0 then
    raise exception 'Positive units are required';
  end if;
  perform pg_advisory_xact_lock(hashtextextended('api-billing:' || p_user_id, 0));
  if not exists (select 1 from public.api_trial_grants where user_id = p_user_id)
    or not exists (select 1 from public.videos where id = p_video_id and user_id = p_user_id) then
    raise exception 'Trial enrollment and an owned video are required';
  end if;
  if exists (select 1 from public.web_processing_charges
             where video_id = p_video_id and user_id = p_user_id) then
    if not public.has_video_processing_charge(p_user_id, p_video_id) then
      raise exception 'Prior processing charge was refunded';
    end if;
    select balance into web_balance from public.api_credits where user_id = p_user_id;
    return query select true, coalesce(web_balance, 0);
    return;
  end if;
  select * into api_result from public.consume_api_units(
    p_user_id, null, 'index_video', p_units, p_video_id,
    'web-index:' || p_video_id::text, '{"origin":"website"}'::jsonb);
  if api_result.allowed then
    insert into public.web_processing_charges (video_id, user_id, source, units)
      values (p_video_id, p_user_id, 'api_units', p_units);
    return query select true, api_result.remaining_balance::integer;
    return;
  end if;
  update public.credits set balance = balance - 1
    where user_id = p_user_id and balance > 0 returning balance into web_balance;
  if found then
    insert into public.web_processing_charges (video_id, user_id, source, units)
      values (p_video_id, p_user_id, 'web_credit', 1);
    return query select true, web_balance;
    return;
  end if;
  return query select false, api_result.remaining_balance::integer;
end;
$$;

revoke all on function public.apply_api_trial_grant(text, text, integer, integer, integer)
  from public, anon, authenticated;
grant execute on function public.apply_api_trial_grant(text, text, integer, integer, integer)
  to service_role;
revoke all on function public.consume_trial_or_processing_credit(text, uuid, integer)
  from public, anon, authenticated;
grant execute on function public.consume_trial_or_processing_credit(text, uuid, integer)
  to service_role;
