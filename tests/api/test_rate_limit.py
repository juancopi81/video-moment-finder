from __future__ import annotations

from src.api.rate_limit import SlidingWindowRateLimiter


def test_sliding_window_limiter_sweeps_expired_keys(monkeypatch) -> None:
    limiter = SlidingWindowRateLimiter()
    moments = iter([0.0, 0.1, 11.0])
    monkeypatch.setattr("src.api.rate_limit.monotonic", lambda: next(moments))

    assert limiter.check(key="key_a", limit=1, window_s=10).allowed is True
    assert limiter.check(key="key_b", limit=1, window_s=10).allowed is True
    assert "key_a" in limiter._hits
    assert "key_b" in limiter._hits

    assert limiter.check(key="key_c", limit=1, window_s=10).allowed is True
    assert "key_a" not in limiter._hits
    assert "key_b" not in limiter._hits
    assert "key_c" in limiter._hits
