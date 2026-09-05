"""Minimal in-process Redis subset for key_quota e2e tests (no redis package)."""
from __future__ import annotations

import time
from collections import defaultdict


class FakeRedis:
    def __init__(self) -> None:
        self._zsets: dict[str, dict[str, float]] = defaultdict(dict)
        self._kv: dict[str, str] = {}
        self._exp: dict[str, float] = {}

    def ping(self):
        return True

    def pipeline(self):
        return _Pipe(self)

    def zremrangebyscore(self, key, min_s, max_s):
        zs = self._zsets.get(key, {})
        drop = [m for m, sc in zs.items() if float(min_s) <= sc <= float(max_s)]
        for m in drop:
            del zs[m]
        return len(drop)

    def zcard(self, key):
        return len(self._zsets.get(key, {}))

    def zadd(self, key, mapping: dict):
        zs = self._zsets[key]
        for m, sc in mapping.items():
            zs[str(m)] = float(sc)
        return len(mapping)

    def expire(self, key, seconds):
        self._exp[key] = time.time() + float(seconds)
        return True

    def get(self, key):
        return self._kv.get(key)

    def incrby(self, key, amount):
        cur = int(self._kv.get(key) or 0)
        cur += int(amount)
        self._kv[key] = str(cur)
        return cur


    def incrbyfloat(self, key, amount):
        cur = float(self._kv.get(key) or 0.0)
        cur += float(amount)
        self._kv[key] = str(cur)
        return cur

    def delete(self, *keys):
        n = 0
        for key in keys:
            if key in self._kv:
                del self._kv[key]
                n += 1
            if key in self._zsets:
                del self._zsets[key]
                n += 1
            self._exp.pop(key, None)
        return n

    def keys(self, pattern="*"):
        import fnmatch
        all_keys = set(self._kv) | set(self._zsets)
        if pattern == "*":
            return list(all_keys)
        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]


class _Pipe:
    def __init__(self, r: FakeRedis):
        self._r = r
        self._ops = []

    def zremrangebyscore(self, *a, **k):
        self._ops.append(("zremrangebyscore", a, k)); return self

    def zcard(self, *a, **k):
        self._ops.append(("zcard", a, k)); return self

    def get(self, *a, **k):
        self._ops.append(("get", a, k)); return self

    def zadd(self, *a, **k):
        self._ops.append(("zadd", a, k)); return self

    def expire(self, *a, **k):
        self._ops.append(("expire", a, k)); return self

    def incrby(self, *a, **k):
        self._ops.append(("incrby", a, k)); return self

    def incrbyfloat(self, *a, **k):
        self._ops.append(("incrbyfloat", a, k)); return self

    def delete(self, *a, **k):
        self._ops.append(("delete", a, k)); return self

    def execute(self):
        out = []
        for name, a, k in self._ops:
            out.append(getattr(self._r, name)(*a, **k))
        self._ops.clear()
        return out
