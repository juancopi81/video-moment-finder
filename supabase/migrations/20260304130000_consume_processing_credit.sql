-- Atomic per-request credit consumption for paid processing admission.

create or replace function public.consume_processing_credit(
  p_user_id text
)
returns table (
  allowed boolean,
  charged boolean,
  remaining_balance integer
)
language plpgsql
set search_path = public
as $$
declare
  updated_balance integer;
  current_balance integer;
begin
  if p_user_id is null or btrim(p_user_id) = '' then
    raise exception 'p_user_id must be non-empty';
  end if;

  update public.credits
  set balance = balance - 1
  where user_id = p_user_id
    and balance > 0
  returning balance into updated_balance;

  if found then
    return query select true, true, updated_balance;
    return;
  end if;

  select balance
  into current_balance
  from public.credits
  where user_id = p_user_id;

  return query select false, false, coalesce(current_balance, 0);
end;
$$;
