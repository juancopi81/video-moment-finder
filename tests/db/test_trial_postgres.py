"""Run real migrations in a disposable PostgreSQL container, never a DB URL.

Opt in with VMF_RUN_POSTGRES_TESTS=1; requires a local Docker daemon and the
postgres:16-alpine image. No ports are published and the container has no network.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VMF_RUN_POSTGRES_TESTS") != "1",
    reason="Opt-in isolated PostgreSQL migration tests (VMF_RUN_POSTGRES_TESTS=1)",
)
ROOT = Path(__file__).resolve().parents[2]
DOCKER = ["docker", "--host=unix:///var/run/docker.sock"]


@pytest.fixture(scope="module")
def sql():
    env = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        env.pop(key, None)
    name = "vmf-trial-test-" + uuid4().hex[:12]
    def docker(*args, **kwargs):
        return subprocess.run(DOCKER + list(args), env=env, text=True, capture_output=True, **kwargs)
    started = docker("run", "--rm", "-d", "--network=none", "--name", name,
                     "-e", "POSTGRES_HOST_AUTH_METHOD=trust", "postgres:16-alpine")
    assert started.returncode == 0, started.stderr
    def execute(statement, *, check=True):
        result = docker("exec", "-i", name, "psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1",
                        "-U", "postgres", "-d", "postgres", input=statement)
        if check:
            assert result.returncode == 0, result.stderr
        return result.stdout.strip() if check else result
    try:
        for _ in range(100):
            # The image's temporary init server accepts Unix-socket traffic,
            # then shuts down. TCP readiness identifies the final server.
            if docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "postgres").returncode == 0:
                break
            time.sleep(0.1)
        else:
            pytest.fail("Isolated PostgreSQL did not start")
        execute("""
          create role anon;
          create role authenticated;
          create role service_role bypassrls;
          create schema auth;
          create function auth.jwt() returns jsonb language sql as
            $$ select coalesce(nullif(current_setting('request.jwt.claims', true), ''), '{}')::jsonb $$;
          grant usage on schema public, auth to anon, authenticated, service_role;
        """)
        for path in sorted((ROOT / "supabase/migrations").glob("*.sql")):
            execute(path.read_text())
        execute("grant all on all tables in schema public to service_role; grant select on all tables in schema public to authenticated;")
        yield execute
    finally:
        docker("rm", "-f", name)


@pytest.fixture
def account():
    return "trial_" + uuid4().hex


def grant_sql(user, *, units=600, quota=1, index_cost=500):
    return f"select granted_units from public.apply_api_trial_grant('{user}', 'email_verified', {units}, {quota}, {index_cost});"


def video_sql(user, *, status="ready"):
    vid = str(uuid4())
    return vid, f"insert into public.videos (id, user_id, youtube_url, status) values ('{vid}', '{user}', 'https://example.com/{vid}', '{status}');"


def balance_sql(user):
    return f"select balance from public.api_credits where user_id='{user}';"


def consume_sql(user, amount, request):
    return f"select allowed from public.consume_api_units('{user}', null, 'text_query', {amount}, null, '{request}');"


def test_all_migrations_apply_and_trial_is_empty_by_default(sql):
    assert sql("select count(*) from public.api_trial_grants;") == "0"
    # Both new migrations are safe to reapply before any grants are active.
    for filename in ("20261002120000_api_billing_retry_safety.sql", "20261002121000_verified_account_trial.sql"):
        sql((ROOT / "supabase/migrations" / filename).read_text())


def test_image_search_migration_debit_refund_and_existing_events(sql, account):
    migration = ROOT / "supabase/migrations/20261009120000_image_search_api_usage.sql"
    sql(migration.read_text())  # The production migration may be safely reapplied.
    sql(grant_sql(account))
    request = "image_" + uuid4().hex
    assert sql(f"select allowed from public.consume_api_units('{account}', null, 'image_query', 1, null, '{request}');") == "t"
    assert sql(balance_sql(account)) == "599"
    sql(f"select public.compensate_api_units('{account}', 1, null, '{request}');")
    assert sql(balance_sql(account)) == "600"
    assert sql(consume_sql(account, 1, "text_" + uuid4().hex)) == "t"


def test_new_verified_account_grant_is_stable_across_retries(sql, account):
    assert sql(grant_sql(account)) == "600"
    assert sql(grant_sql(account, units=900)) == "600"
    assert sql(balance_sql(account)) == "600"
    assert sql(f"select count(*) from public.api_billing_events where user_id='{account}';") == "1"


def test_concurrent_grants_add_exactly_once(sql, account):
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: sql(grant_sql(account)), range(24)))
    assert set(results) == {"600"}
    assert sql(balance_sql(account)) == "600"
    assert sql(f"select count(*) from public.api_trial_grants where user_id='{account}';") == "1"


def test_legacy_free_allowance_offsets_grant_without_touching_paid_balances(sql, account):
    _, insert = video_sql(account)
    sql(insert)
    sql(f"insert into public.credits (user_id,balance) values ('{account}',7); insert into public.api_credits (user_id,balance) values ('{account}',10000);")
    assert sql(grant_sql(account)) == "100"
    assert sql(balance_sql(account)) == "10100"
    assert sql(f"select balance from public.credits where user_id='{account}';") == "7"
    assert sql(f"select legacy_units_offset from public.api_trial_grants where user_id='{account}';") == "500"


def test_api_funded_and_failed_videos_do_not_count_as_legacy_free_usage(sql, account):
    video, insert = video_sql(account)
    sql(insert)
    _, failed = video_sql(account, status="failed")
    sql(failed)
    sql(f"insert into public.api_usage_events(user_id,event_type,units,video_id) values ('{account}','index_video',500,'{video}');")
    assert sql(grant_sql(account)) == "600"


def test_unattributed_historical_api_upload_does_not_lose_trial_units(sql, account):
    vid, insert = video_sql(account)
    sql(insert)
    sql(f"update public.videos set source_type='upload', youtube_url=null, source_r2_key='source/{vid}/test.mp4' where id='{vid}';")
    sql(f"insert into public.api_usage_events(user_id,event_type,units,video_id) values ('{account}','index_video',500,null);")
    assert sql(grant_sql(account)) == "600"


def test_unlimited_override_keeps_full_trial(sql, account):
    _, insert = video_sql(account)
    sql(insert)
    sql(f"insert into public.video_access_overrides(user_id,unlimited_videos) values ('{account}',true);")
    assert sql(grant_sql(account)) == "600"


def test_configured_cost_and_zero_remaining_still_record_one_enrollment(sql, account):
    _, insert = video_sql(account)
    sql(insert)
    assert sql(grant_sql(account, units=300, index_cost=400)) == "0"
    assert sql(grant_sql(account, units=1000)) == "0"
    assert sql(f"select count(*) from public.api_trial_grants where user_id='{account}';") == "1"


def test_concurrent_spending_cannot_overdraw_trial(sql, account):
    sql(grant_sql(account))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: sql(consume_sql(account, 100, account + str(i))), range(10)))
    assert results.count("t") == 6
    assert results.count("f") == 4
    assert sql(balance_sql(account)) == "0"
    assert sql(grant_sql(account)) == "600"  # reconnect cannot refill
    assert sql(balance_sql(account)) == "0"


def test_concurrent_debit_retry_is_idempotent(sql, account):
    sql(grant_sql(account))
    statement = consume_sql(account, 100, account + "charge")
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(lambda _: sql(statement), range(12))) == {"t"}
    assert sql(balance_sql(account)) == "500"


def test_failed_work_refunds_original_charge_exactly_once(sql, account):
    sql(grant_sql(account))
    request_id = account + "refund"
    sql(consume_sql(account, 5, request_id))
    refund = f"select public.compensate_api_units('{account}',5,null,'{request_id}');"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: sql(refund), range(12)))
    assert sql(balance_sql(account)) == "600"
    assert sql(f"select count(*) from public.api_usage_events where user_id='{account}' and event_type='compensation';") == "1"
    assert sql(consume_sql(account, 5, request_id), check=False).returncode != 0


def test_refunds_cannot_mint_units_or_cross_account_boundaries(sql, account):
    sql(grant_sql(account))
    request_id = account + "safe"
    sql(consume_sql(account, 5, request_id))
    for user, amount, req in [(account, 6, request_id), (account, 5, "missing"), (account + "other", 5, request_id)]:
        bad = sql(f"select public.compensate_api_units('{user}',{amount},null,'{req}');", check=False)
        assert bad.returncode != 0
    assert sql(balance_sql(account)) == "595"


def test_service_role_can_enroll_but_no_unverified_boolean_is_exposed(sql, account):
    assert sql("set role service_role; " + grant_sql(account)) == "600"
    assert sql(balance_sql(account)) == "600"


def test_grant_failure_rolls_back_enrollment_and_ledger(sql, account):
    sql(f"insert into public.api_credits(user_id,balance) values ('{account}',2147483647);")
    assert sql(grant_sql(account), check=False).returncode != 0
    assert sql(f"select count(*) from public.api_trial_grants where user_id='{account}';") == "0"
    assert sql(f"select count(*) from public.api_billing_events where user_id='{account}';") == "0"
    assert sql(balance_sql(account)) == "2147483647"


def test_web_index_charge_is_shared_once_then_paid_credit_fallback(sql, account):
    sql(grant_sql(account))
    sql(f"insert into public.credits(user_id,balance) values ('{account}',2);")
    first, insert = video_sql(account)
    sql(insert)
    assert sql(f"select public.has_video_processing_charge('{account}','{first}');") == "f"
    shared = f"select allowed from public.consume_trial_or_processing_credit('{account}','{first}',500);"
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(lambda _: sql(shared), range(12))) == {"t"}
    assert sql(balance_sql(account)) == "100"
    assert sql(f"select public.has_video_processing_charge('{account}','{first}');") == "t"
    assert sql(f"select balance from public.credits where user_id='{account}';") == "2"
    second, insert = video_sql(account)
    sql(insert)
    sql(f"select * from public.consume_trial_or_processing_credit('{account}','{second}',500);")
    assert sql(balance_sql(account)) == "100"
    assert sql(f"select balance from public.credits where user_id='{account}';") == "1"
    assert sql(f"select source from public.web_processing_charges where video_id='{second}';") == "web_credit"
    assert sql(f"select public.has_video_processing_charge('{account}','{second}');") == "t"


def test_refunded_indexing_no_longer_proves_processing_admission(sql, account):
    sql(grant_sql(account))
    video, insert = video_sql(account)
    sql(insert)
    sql(f"select * from public.consume_trial_or_processing_credit('{account}','{video}',500);")
    assert sql(f"select public.has_video_processing_charge('{account}','{video}');") == "t"
    sql(f"select public.compensate_api_units('{account}',500,'{video}','web-index:{video}');")
    assert sql(f"select public.has_video_processing_charge('{account}','{video}');") == "f"
    assert sql(f"select * from public.consume_trial_or_processing_credit('{account}','{video}',500);", check=False).returncode != 0


def test_client_roles_cannot_grant_or_refund(sql, account):
    for role in ("anon", "authenticated"):
        assert sql(f"set role {role}; " + grant_sql(account), check=False).returncode != 0
        assert sql(f"set role {role}; select public.compensate_api_units('{account}',1,null,'fake');", check=False).returncode != 0
    sql(grant_sql(account))
    assert sql(f"set role authenticated; set request.jwt.claims='{{\"sub\":\"{account}other\"}}'; select count(*) from public.api_trial_grants where user_id='{account}';") == "0"
    assert sql(f"set role authenticated; set request.jwt.claims='{{\"sub\":\"{account}\"}}'; select count(*) from public.api_trial_grants where user_id='{account}';") == "1"
