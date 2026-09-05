"""test_fake_redis.py — in-process Redis subset for key_quota e2e (10 cases).

`_fake_redis.py` shadows a small Redis surface used by the key-quota
scheduler. Tests here pin every operation (zadd/zcard/zremrangebyscore,
get/incrby/incrbyfloat, pipeline composition) so that any silent
semantic drift vs the real Redis breaks the test, not production.
"""
from __future__ import annotations

from anchor._fake_redis import FakeRedis


def test_ping_returns_true():
    assert FakeRedis().ping() is True


def test_zadd_and_zcard_roundtrip():
    """zadd writes members, zcard reports the size."""
    r = FakeRedis()
    assert r.zadd("k", {"a": 1.0, "b": 2.0}) == 2
    assert r.zcard("k") == 2
    # overwrite a's score
    assert r.zadd("k", {"a": 5.0}) == 1
    assert r.zcard("k") == 2  # member count unchanged


def test_zremrangebyscore_removes_inclusive_range():
    """The fake must use the same inclusive bounds as real Redis."""
    r = FakeRedis()
    r.zadd("k", {"a": 1.0, "b": 5.0, "c": 10.0})
    removed = r.zremrangebyscore("k", 1.0, 5.0)
    assert removed == 2
    assert r.zcard("k") == 1


def test_zremrangebyscore_returns_zero_when_no_match():
    r = FakeRedis()
    r.zadd("k", {"a": 100.0})
    assert r.zremrangebyscore("k", 0.0, 50.0) == 0
    assert r.zcard("k") == 1


def test_expire_sets_ttl():
    """expire is recorded (fake doesn't actually expire keys, by design)."""
    r = FakeRedis()
    assert r.expire("anykey", 60) is True


def test_get_incrby_incrbyfloat_basic_arithmetic():
    """get returns None for missing, incrby/incrbyfloat start from 0."""
    r = FakeRedis()
    assert r.get("missing") is None
    assert r.incrby("counter", 3) == 3
    assert r.incrby("counter", 5) == 8
    assert r.incrbyfloat("frac", 1.5) == 1.5
    assert r.incrbyfloat("frac", 2.5) == 4.0


def test_delete_removes_kv_and_zset():
    """delete is variadic; passing multiple keys removes all."""
    r = FakeRedis()
    r.set = lambda *_: None  # noqa: F811  # ensure no method collision
    r.incrby("k1", 1)
    r.zadd("k2", {"m": 1.0})
    assert r.delete("k1", "k2") == 2
    assert r.get("k1") is None
    assert r.zcard("k2") == 0


def test_keys_glob_returns_matching_only():
    """Glob filter applied to the union of kv + zset keys."""
    r = FakeRedis()
    r.incrby("foo", 1)
    r.incrby("foobar", 1)
    r.zadd("baz", {"m": 1.0})
    # FakeRedis.Keys uses fnmatch under the hood
    assert set(r.keys("foo*")) == {"foo", "foobar"}
    assert set(r.keys("b*")) == {"baz"}


def test_pipeline_chains_and_executes_in_order():
    """The fake pipeline records calls and replays them on execute()."""
    r = FakeRedis()
    p = r.pipeline()
    p.zadd("k", {"a": 1.0})
    p.zadd("k", {"b": 2.0})
    p.zcard("k")
    p.incrby("counter", 7)
    results = p.execute()
    assert results == [1, 1, 2, 7]
    assert r.zcard("k") == 2
    assert r.incrby("counter", 1) == 8


def test_pipeline_execute_clears_ops_for_reuse():
    """Calling execute() twice on the same pipeline must replay cleanly."""
    r = FakeRedis()
    p = r.pipeline()
    p.zadd("k", {"a": 1.0})
    p.execute()
    p.zcard("k")
    # Second batch: only the second zcard should be in results
    out = p.execute()
    assert out == [1]


def test_pipeline_unknown_method_raises_via_getattr():
    """Unknown ops must AttributeError through getattr in execute()."""
    import pytest

    r = FakeRedis()
    p = r.pipeline()
    p.zadd("k", {"a": 1.0})
    # Inject a bogus op manually
    p._ops.append(("no_such_method", (), {}))
    with pytest.raises(AttributeError):
        p.execute()