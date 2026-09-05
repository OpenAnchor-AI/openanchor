"""End-to-end smoke: spin up server, hit pure-mode /v1 chat, assert 200."""
import os
import subprocess
import sys
import time

import httpx


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(0.5)
    return False


def main() -> int:
    print("=== Anchor pure-mode smoke ===\n")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "anchor.server:app",
         "--host", "127.0.0.1", "--port", "8088", "--log-level", "warning"],
        cwd="src", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        if not wait_for_server("http://127.0.0.1:8088/healthz"):
            print("FAIL: server did not start within 30s")
            return 1
        print("✓ server up")

        total_cost = 0.0
        api_key = os.environ.get("ANCHOR_API_KEYS", "").split(",")[0].strip() or None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        cases = [
            ("easy", "hi"),
            ("code", "how to read a file in python"),
            ("math", "explain the integral of x^2 from 0 to 1"),
        ]
        for label, query in cases:
            r = httpx.post(
                "http://127.0.0.1:8088/v1/chat/completions",
                json={
                    "model": "anchor",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 64,
                },
                headers=headers,
                timeout=60.0,
            )
            if r.status_code != 200:
                print(f"  ✗ {label} -> {r.status_code} {r.text[:120]}")
                return 1
            data = r.json()
            worker = (
                data.get("worker")
                or (data.get("metadata") or {}).get("worker")
                or data.get("model")
                or "?"
            )
            cost = float(data.get("cost_yuan") or 0.0)
            lat = data.get("latency_ms") or data.get("latency") or "?"
            total_cost += cost
            content = ""
            try:
                content = (data["choices"][0]["message"]["content"] or "")[:40]
            except Exception:
                content = str(data)[:40]
            print(f"  ✓ {label:6s} -> {str(worker):22s}  ¥{cost:.4f}  {lat}  {content!r}")

        print(f"\nTotal cost: ¥{total_cost:.4f}")
        print("\n✓ smoke PASS")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
