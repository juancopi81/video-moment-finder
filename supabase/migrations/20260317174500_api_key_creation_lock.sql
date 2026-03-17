-- Serialize API key creation even when the user has zero active keys.
--
-- The original create_api_key_atomic() implementation row-locked the
-- caller's existing active keys, which works only when at least one row
-- already exists. On a first-key create, there are no rows to lock, so
-- concurrent requests can all observe active_count = 0 and exceed the
-- configured per-user cap.

create or replace function public.create_api_key_atomic(
  p_user_id text,
  p_name text,
  p_key_hash text,
  p_key_prefix text,
  p_max_keys integer default 10
)
returns setof public.api_keys
language plpgsql
set search_path = public
as $$
declare
  active_count integer;
begin
  -- Serialize create operations per user even when there are no rows yet.
  perform pg_advisory_xact_lock(hashtextextended('api_keys:' || p_user_id, 0));

  -- Lock active rows too so concurrent create/revoke activity observes a
  -- stable count within the transaction.
  perform 1
    from public.api_keys
   where user_id = p_user_id
     and revoked_at is null
     for update;

  select count(*) into active_count
    from public.api_keys
   where user_id = p_user_id
     and revoked_at is null;

  if active_count >= p_max_keys then
    raise exception 'Maximum of % active API keys per user', p_max_keys;
  end if;

  return query
    insert into public.api_keys (user_id, name, key_hash, key_prefix)
    values (p_user_id, p_name, p_key_hash, p_key_prefix)
    returning *;
end;
$$;
