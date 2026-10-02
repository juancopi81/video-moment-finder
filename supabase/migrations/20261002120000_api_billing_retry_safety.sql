-- Serialize account debits and make refunds refer to a real original debit.
-- Original compensation incorrectly found the charge's request_id and returned
-- before refunding it. Refunds now have a separate idempotency key.

create or replace function public.consume_api_units(
  p_user_id text, p_api_key_id uuid, p_event_type text, p_units integer,
  p_video_id uuid default null, p_request_id text default null,
  p_metadata jsonb default '{}'::jsonb
)
returns table (allowed boolean, remaining_balance integer)
language plpgsql
set search_path = public
as $$
declare
  current_balance integer;
  prior public.api_usage_events%rowtype;
begin
  if p_user_id is null or btrim(p_user_id) = '' or p_units is null or p_units <= 0 then
    raise exception 'A user and positive units are required';
  end if;
  if p_event_type = 'compensation' then
    raise exception 'Use compensate_api_units for refunds';
  end if;
  if p_request_id like 'compensate:%' then
    raise exception 'Reserved request id prefix';
  end if;
  perform pg_advisory_xact_lock(hashtextextended('api-billing:' || p_user_id, 0));
  if p_request_id is not null then
    select * into prior from public.api_usage_events where request_id = p_request_id;
    if found then
      if prior.user_id <> p_user_id or prior.units <> p_units
        or prior.event_type <> p_event_type
        or prior.video_id is distinct from p_video_id
        or prior.api_key_id is distinct from p_api_key_id then
        raise exception 'Request id does not match the original charge';
      end if;
      -- A compensated request cannot be replayed as a free successful charge.
      if exists (select 1 from public.api_usage_events
                 where request_id = 'compensate:' || p_request_id) then
        raise exception 'Use a new request id after compensation';
      end if;
      select balance into current_balance from public.api_credits where user_id = p_user_id;
      return query select true, coalesce(current_balance, 0);
      return;
    end if;
  end if;
  update public.api_credits set balance = balance - p_units
  where user_id = p_user_id and balance >= p_units
  returning balance into current_balance;
  if found then
    insert into public.api_usage_events
      (user_id, api_key_id, event_type, units, video_id, request_id, metadata)
    values (p_user_id, p_api_key_id, p_event_type, p_units, p_video_id,
            p_request_id, coalesce(p_metadata, '{}'::jsonb));
    return query select true, current_balance;
    return;
  end if;
  select balance into current_balance from public.api_credits where user_id = p_user_id;
  return query select false, coalesce(current_balance, 0);
end;
$$;

create or replace function public.compensate_api_units(
  p_user_id text, p_units integer, p_video_id uuid default null,
  p_request_id text default null, p_metadata jsonb default '{}'::jsonb
)
returns void
language plpgsql
set search_path = public
as $$
declare
  prior public.api_usage_events%rowtype;
begin
  if p_user_id is null or btrim(p_user_id) = '' or p_units is null or p_units <= 0
    or p_request_id is null or btrim(p_request_id) = '' then
    raise exception 'A user, positive units, and original request id are required';
  end if;
  perform pg_advisory_xact_lock(hashtextextended('api-billing:' || p_user_id, 0));
  select * into prior from public.api_usage_events where request_id = p_request_id;
  if not found or prior.user_id <> p_user_id or prior.units <> p_units
    or prior.event_type = 'compensation'
    or prior.video_id is distinct from p_video_id then
    raise exception 'Refund does not match an original debit';
  end if;
  insert into public.api_usage_events
    (user_id, api_key_id, event_type, units, video_id, request_id, metadata)
  values (p_user_id, prior.api_key_id, 'compensation', -p_units, p_video_id,
          'compensate:' || p_request_id,
          coalesce(p_metadata, '{}'::jsonb) || jsonb_build_object('original_request_id', p_request_id))
  on conflict (request_id) where request_id is not null do nothing;
  if not found then
    return;
  end if;
  insert into public.api_credits (user_id, balance) values (p_user_id, p_units)
  on conflict (user_id) do update set balance = public.api_credits.balance + excluded.balance;
end;
$$;

-- These operations are server-only. Client roles may read their RLS-scoped
-- balance/ledger but must never invoke arbitrary grants, debits, or refunds.
revoke all on function public.consume_api_units(text, uuid, text, integer, uuid, text, jsonb)
  from public, anon, authenticated;
grant execute on function public.consume_api_units(text, uuid, text, integer, uuid, text, jsonb)
  to service_role;
revoke all on function public.compensate_api_units(text, integer, uuid, text, jsonb)
  from public, anon, authenticated;
grant execute on function public.compensate_api_units(text, integer, uuid, text, jsonb)
  to service_role;
