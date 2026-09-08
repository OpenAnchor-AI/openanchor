"""T-04: `anchor init` interactive setup wizard.

Goal: 5-minute onboarding for a fresh Anchor install.

Flow (when stdin is a TTY; non-TTY runs read scripted answers):
    1. Welcome banner + summary of what we'll do.
    2. Pick which providers (channels) the user wants to enable.
    3. For each chosen channel, collect API keys (getpass, no echo).
    4. For each chosen channel, pick which models under that channel to enable.
    5. Probe each selected worker's reachability (lightweight POST /chat/completions
       with max_tokens=1, no streaming).
    6. Write `~/.anchor/workers.yaml` (channel + worker subset; chmod 600).
    7. Write `~/.anchor/keys.yaml` (env-var -> key mapping; chmod 600).
    8. Print next-step hint: `anchor serve` (or `anchor doctor` first).

Design constraints (audit 2026-09-08):
    * No new third-party deps. questionary/click are tempting but the test
      surface grows; stdlib input() + getpass() + tabulate-style loops are
      more than enough for a 4-question wizard.
    * All IO is parameterized via ``IO`` so tests can mock stdin/stdout/
      getpass without monkey-patching sys.stdin. (Direct monkey-patching
      of sys.stdin in tests is a recipe for test-order coupling.)
    * Key material NEVER hits the YAML or the on-disk log. The keys.yaml
      file is 0600 to keep the leak radius small even on a multi-user box.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

# Pull the worker registry from anchor.config; this is the only place we
# couple to runtime config, so unit tests can keep the canonical anchor
# registry (real, frozen) but stub out the IO layer.
from anchor.config import WORKERS, _read_api_key, Worker  # noqa: F401


# Path constants are computed at import time so the module exposes a
# stable public surface (tests + other modules import these names). The
# helpers below recompute at call time so tests that mutate HOME after
# import get fresh paths — see _anchor_home().
ANCHOR_HOME = os.path.join(os.path.expanduser("~"), ".anchor")
WORKERS_YAML = os.path.join(ANCHOR_HOME, "workers.yaml")
KEYS_YAML = os.path.join(ANCHOR_HOME, "keys.yaml")


def _anchor_home() -> str:
    """Resolve ~/.anchor at call time so tests that mutate HOME work.

    Module-level expanduser() captures the value at import time; that
    breaks any test that flips HOME after import. Recomputing here is
    a 2-microsecond cost on every wizard call but lets tests run in
    any order without fixture setup.
    """
    return os.path.join(os.path.expanduser("~"), ".anchor")


def _workers_yaml_path() -> str:
    return os.path.join(_anchor_home(), "workers.yaml")


def _keys_yaml_path() -> str:
    return os.path.join(_anchor_home(), "keys.yaml")

# 7 days — head baseline older than this is "stale" and the gateway
# should re-run retrain before serving production traffic.
HEAD_BASELINE_MAX_AGE_DAYS = 7

# Probe timeout per worker. Keep short — we only want to confirm the API
# accepts a request, not measure latency.
PROBE_TIMEOUT_SEC = 5.0

# Display-name -> env-var that holds the key. Mirrors the channel layout
# in anchor.config.WORKERS; keep in sync when new providers are added.
#
# 2026-09-08: list curated from the live WORKERS registry. ``key_envs``
# per worker are collapsed to a primary env-var name; we write that
# env-var to keys.yaml (and tell the user to export it, or use the
# env-writer helper below).
_CHANNEL_PRIMARY_ENV: dict[str, str] = {
    "opencode-zen":     "OPENCODE_ZEN_API_KEY",
    "minimax-direct":   "MINIMAX_API_KEY_1",
    "baosiapi":         "BAOSIAPI_GPT_API_KEY",
    "baosiapi-gpt":     "BAOSIAPI_GPT_API_KEY",
    "baosiapi-grok":    "BAOSIAPI_GPT_API_KEY",
    "baosiapi-kimi":    "BAOSIAPI_GPT_API_KEY",
    "deepseek-official": "DEEPSEEK_API_KEY",
    "kilo":             "KILOCODE_API_KEY",
    # ollama-local has no key (local daemon).
}


# ---------------------------------------------------------------------------
# IO abstraction — every prompt/print/key entry is routed through these
# so tests can supply scripted answers and capture banners without touching
# the real stdio.
# ---------------------------------------------------------------------------


@dataclass
class IO:
    """Pluggable IO for the init wizard.

    Defaults target the real stdio (interactive TTY). Tests pass a stub
    that returns preset answers and records everything printed.
    """

    read: Callable[[str], str] = field(default=lambda prompt: input(prompt))
    read_password: Callable[[str], str] = field(
        default=lambda prompt: getpass.getpass(prompt)
    )
    write: Callable[[str], None] = field(default=lambda s: print(s, end=""))
    isatty: Callable[[], bool] = field(default=lambda: sys.stdin.isatty())

    def prompt(self, text: str, *, default: str = "") -> str:
        """Display ``text`` and return the user-typed answer.

        ``default`` is shown in brackets after the prompt; pressing Enter
        alone returns it. Empty default = answer is required.
        """
        suffix = f" [{default}]" if default else ""
        return self.read(f"{text}{suffix}: ").strip()

    def choose_one(self, header: str, options: Sequence[str]) -> str:
        """Render a numbered list of options and return the chosen label.

        Returns the original label (not the number) so callers can use
        the human-readable channel name directly. Re-prompts on invalid
        input.
        """
        self.write(header + "\n")
        for idx, label in enumerate(options, start=1):
            self.write(f"  {idx:>2}. {label}\n")
        while True:
            raw = self.read(f"Pick [1-{len(options)}]: ").strip()
            if not raw:
                # TTY default = first option; non-TTY empty input is an error
                if self.isatty():
                    return options[0]
                continue
            if raw.isdigit():
                n = int(raw)
                if 1 <= n <= len(options):
                    return options[n - 1]
            self.write(f"  ! choose a number 1..{len(options)}\n")

    def choose_many(self, header: str, options: Sequence[str]) -> list[str]:
        """Multi-select: comma-separated indices or labels.

        Empty input + TTY = all options selected. Empty + non-TTY = empty
        (the caller is expected to seed a sensible default in that case).
        """
        self.write(header + "\n")
        for idx, label in enumerate(options, start=1):
            self.write(f"  {idx:>2}. {label}\n")
        hint = "comma-separated numbers (e.g. 1,3) or empty for all"
        raw = self.read(f"{hint}: ").strip()
        if not raw:
            return list(options) if self.isatty() else []
        # accept either indices or labels
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        chosen: list[str] = []
        for p in parts:
            if p.isdigit():
                n = int(p)
                if 1 <= n <= len(options) and options[n - 1] not in chosen:
                    chosen.append(options[n - 1])
            elif p in options and p not in chosen:
                chosen.append(p)
        return chosen

    def yes_no(self, text: str, default_yes: bool = True) -> bool:
        suffix = "[Y/n]" if default_yes else "[y/N]"
        raw = self.read(f"{text} {suffix}: ").strip().lower()
        if not raw:
            return default_yes
        return raw in ("y", "yes")

    def say(self, msg: str) -> None:
        self.write(msg + "\n")


# ---------------------------------------------------------------------------
# Channel + worker model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelGroup:
    """A provider channel grouping one or more workers.

    A channel is a logical billing group (e.g. baosiapi-* share one key
    in production) and the wizard treats it as the unit of API-key
    collection. Workers inside a channel are then toggled individually.
    """

    name: str
    description: str
    workers: tuple[Worker, ...]
    needs_key: bool
    primary_env: str  # env-var the wizard writes into keys.yaml

    @property
    def worker_names(self) -> tuple[str, ...]:
        return tuple(w.name for w in self.workers)


def build_channel_groups() -> list[ChannelGroup]:
    """Group WORKERS by channel. One group per channel.

    Order: cheapest/free first, paid at the end. This is the order the
    wizard shows, so a single-key user who hits Enter ends up with the
    zero-marginal-cost providers enabled.
    """
    by_chan: dict[str, list[Worker]] = {}
    for w in WORKERS:
        by_chan.setdefault(w.channel, []).append(w)

    # Stable display order: keys in this list come first, then any new
    # channels in the order they appear in WORKERS.
    preferred_order = [
        "opencode-zen", "kilo", "minimax-direct", "ollama-local",
        "deepseek-official", "baosiapi", "baosiapi-gpt", "baosiapi-grok",
        "baosiapi-kimi",
    ]
    seen: set[str] = set()
    ordered: list[str] = [c for c in preferred_order if c in by_chan]
    for c in by_chan:
        if c not in ordered:
            ordered.append(c)
    groups: list[ChannelGroup] = []
    for ch in ordered:
        workers = sorted(by_chan[ch], key=lambda w: w.slot)
        needs_key = ch != "ollama-local"
        primary = _CHANNEL_PRIMARY_ENV.get(ch, "")
        # Description: pick the cheapest worker's model for context.
        if workers:
            descr = workers[0].description or ch
        else:
            descr = ch
        groups.append(ChannelGroup(
            name=ch,
            description=descr,
            workers=tuple(workers),
            needs_key=needs_key,
            primary_env=primary,
        ))
        seen.add(ch)
    return groups


def channel_label(g: ChannelGroup) -> str:
    """One-line human label for a channel menu item."""
    n = len(g.workers)
    suffix = "no key" if not g.needs_key else g.primary_env
    return f"{g.name}  ({n} model{'s' if n != 1 else ''}, {suffix})"


# ---------------------------------------------------------------------------
# YAML emitter (intentionally tiny — workers.yaml/keys.yaml are
# machine-generated, never hand-edited by users, and a 30-line emitter
# is less code than a pyyaml dependency here).
# ---------------------------------------------------------------------------


def _yaml_escape(s: str) -> str:
    """Quote a string for a single-line scalar; avoids pyyaml dep."""
    if s == "" or any(c in s for c in (":", "#", "\n", '"', "'")):
        return json.dumps(s)  # JSON happens to be a valid YAML scalar subset
    return s


def emit_workers_yaml(selected: dict[str, list[str]]) -> str:
    """Render the workers subset to YAML.

    Format:
        channels:
          - name: opencode-zen
            enabled: true
            models:
              - deepseek-v4-flash
          - name: baosiapi
            enabled: false   # omitted, kept here as a record of intent
            models: []
    """
    lines: list[str] = ["# Generated by `anchor init`.", "channels:"]
    for chan, names in selected.items():
        lines.append(f"  - name: {_yaml_escape(chan)}")
        enabled = "true" if names else "false"
        lines.append(f"    enabled: {enabled}")
        if names:
            lines.append("    models:")
            for n in names:
                lines.append(f"      - {_yaml_escape(n)}")
        else:
            lines[-1] = f"  - name: {_yaml_escape(chan)}  # skipped (no key)"
    return "\n".join(lines) + "\n"


def emit_keys_yaml(entries: dict[str, str]) -> str:
    """Render the env-var -> key YAML.

    Format:
        keys:
          OPENCODE_ZEN_API_KEY: "sk-..."
          MINIMAX_API_KEY_1: "ey..."
    """
    lines = ["# Generated by `anchor init`. chmod 600 (owner read/write only).", "keys:"]
    for env_name in sorted(entries):
        # We never want the secret visible in the rendered string during
        # emit — but on disk it must be. The caller hands us the plaintext
        # so the on-disk artifact is correct, and tests assert against
        # the file (not stdout).
        lines.append(f"  {env_name}: {_yaml_escape(entries[env_name])}")
    return "\n".join(lines) + "\n"


# Minimal YAML loader. Only supports the flat-key + 2-level list
# structure we emit. PyYAML would be a dependency just for this; a
# 40-line parser avoids it and keeps the test surface obvious.
def parse_workers_yaml(text: str) -> dict[str, list[str]]:
    """Inverse of emit_workers_yaml. Returns {channel: [model, ...]}."""
    result: dict[str, list[str]] = {}
    cur_chan: str | None = None
    in_models = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 2 and stripped.startswith("- name:"):
            cur_chan = stripped.split(":", 1)[1].strip()
            result.setdefault(cur_chan, [])
            in_models = False
        elif indent == 4 and stripped.startswith("models:"):
            in_models = True
        elif indent == 6 and in_models and stripped.startswith("-"):
            if cur_chan is not None:
                result[cur_chan].append(stripped[1:].strip())
        else:
            in_models = False
    return result


def parse_keys_yaml(text: str) -> dict[str, str]:
    """Inverse of emit_keys_yaml. Returns {env_var: key}.

    Handles both bare scalars and JSON-quoted scalars (json.dumps uses
    JSON which is a valid YAML scalar form). When the raw value is a
    JSON-quoted string, it is decoded so the caller sees the original
    value, not the escaped form.
    """
    import json as _json
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        if stripped.startswith("keys:"):
            continue
        if ":" in stripped and not stripped.startswith("-"):
            k, _, v = stripped.partition(":")
            v = v.strip()
            if len(v) >= 2 and v.startswith('"') and v.endswith('"'):
                try:
                    v = _json.loads(v)
                except Exception:
                    pass  # fall through; leave the raw value
            result[k.strip()] = v
    return result


# ---------------------------------------------------------------------------
# Permission helper
# ---------------------------------------------------------------------------


def _ensure_secure(path: str) -> None:
    """chmod 600 (owner read/write). Best-effort: on Windows os.chmod is a
    no-op for the high bits, which is the right behavior."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Connectivity probe — synchronous urllib; the wizard is sequential so
# there's no need for an async client.
# ---------------------------------------------------------------------------


def probe_worker(worker: Worker, api_key: str) -> tuple[bool, str]:
    """One-shot HTTP probe. Returns (ok, detail)."""
    if not api_key and worker.kind != "opencode-cli":
        # local ollama is reachable without a key
        if worker.channel != "ollama-local":
            return False, "no-key"
    payload = json.dumps({
        "model": worker.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }).encode()
    req = urllib.request.Request(
        f"{worker.base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_SEC) as resp:
            body = resp.read().decode(errors="ignore")
            if resp.status >= 400 or '"error"' in body:
                return False, body[:120] or f"http {resp.status}"
            return True, "ok"
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore") if e.fp else ""
        return False, f"http {e.code}: {body[:120]}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    except Exception as e:  # last-resort: keep wizard moving on any error
        return False, f"{type(e).__name__}: {str(e)[:120]}"


# ---------------------------------------------------------------------------
# Head baseline age check (used by both init's final summary and doctor)
# ---------------------------------------------------------------------------


def head_baseline_age_days() -> dict:
    """Snapshot the head baseline file. Returns dict with ts/age/exists."""
    import time as _time
    from pathlib import Path
    p = Path.home() / "anchor" / "data" / "head_baseline.json"
    if not p.exists():
        return {"exists": False, "age_days": None, "stale": False, "path": str(p)}
    try:
        import json as _json
        data = _json.loads(p.read_text())
        ts = float(data.get("ts") or 0.0)
    except Exception:
        return {"exists": True, "age_days": None, "stale": True, "path": str(p),
                "reason": "unreadable"}
    if ts <= 0:
        return {"exists": True, "age_days": None, "stale": True, "path": str(p),
                "reason": "no-ts"}
    age = (_time.time() - ts) / 86400.0
    return {
        "exists": True,
        "age_days": round(age, 1),
        "stale": age > HEAD_BASELINE_MAX_AGE_DAYS,
        "path": str(p),
        "ts": ts,
    }


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------


def run_wizard(groups: list[ChannelGroup], io: IO,
               *, skip_probes: bool = False) -> dict:
    """Drive the interactive wizard. Returns the collected config dict.

    The returned dict is shaped:
        {
            "selected": {channel: [worker_name, ...], ...},
            "keys":     {env_var: key, ...},
            "probes":   {worker_name: (ok, detail), ...},
        }
    The caller (or `init_command`) is responsible for emitting YAML files
    and chmod'ing them; this function stays pure of side-effects beyond
    stdin/stdout.
    """
    io.say("┌─────────────────────────────────────────────┐")
    io.say("│  Welcome to Anchor — 5-min setup wizard      │")
    io.say("└─────────────────────────────────────────────┘")
    io.say("")
    io.say("We'll:")
    io.say("  1. Pick providers you want to enable")
    io.say("  2. Collect API keys (hidden input)")
    io.say("  3. Pick which models under each provider")
    io.say("  4. Probe connectivity (fast)")
    io.say(f"  5. Write {WORKERS_YAML}")
    io.say(f"  6. Write {KEYS_YAML}  (chmod 600)")
    io.say("")

    # --- Q1: channels ---------------------------------------------------
    channel_labels = [channel_label(g) for g in groups]
    chosen_labels = io.choose_many(
        "Q1 — Which providers do you want to enable?",
        channel_labels,
    )
    chosen = [g for g in groups if channel_label(g) in chosen_labels]
    if not chosen:
        io.say("Nothing selected. Wrote an empty config; you can re-run.")
        return {"selected": {}, "keys": {}, "probes": {}}

    # --- Q2: keys -------------------------------------------------------
    keys: dict[str, str] = {}
    for g in chosen:
        if not g.needs_key:
            io.say(f"  • {g.name}: local channel, no key required")
            continue
        io.say("")
        io.say(f"  Provider: {g.name} ({g.description[:60]})")
        io.say(f"  Key env: {g.primary_env}  (we'll write it to keys.yaml)")
        if g.primary_env and _read_api_key(g.primary_env):
            if io.yes_no(
                f"  {g.primary_env} already set in env — reuse it?",
                default_yes=True,
            ):
                keys[g.primary_env] = _read_api_key(g.primary_env)
                continue
        while True:
            val = io.read_password(f"  API key for {g.name}: ")
            if val:
                keys[g.primary_env] = val
                break
            if io.yes_no("  Skip this provider?", default_yes=False):
                break
            io.say("  (input was empty; type/paste a key or skip)")

    # Drop channels for which we never collected a key.
    chosen = [g for g in chosen if (not g.needs_key) or (
        g.primary_env in keys and keys[g.primary_env]
    )]

    # --- Q3: models -----------------------------------------------------
    selected: dict[str, list[str]] = {}
    for g in chosen:
        options = [w.name for w in g.workers]
        if len(options) == 1:
            picked = options
            io.say(f"  {g.name}: enabling {options[0]} (only model in this channel)")
        else:
            picked = io.choose_many(
                f"Q3 — Which models under {g.name}?",
                options,
            )
        selected[g.name] = picked

    # --- Q4: probes -----------------------------------------------------
    probes: dict[str, tuple[bool, str]] = {}
    if not skip_probes:
        io.say("")
        io.say("Q4 — Probing selected workers (≤ 5s each)…")
        for g in chosen:
            for w in g.workers:
                if w.name not in selected.get(g.name, []):
                    continue
                k = keys.get(g.primary_env, "")
                if w.kind == "opencode-cli":
                    probes[w.name] = (True, "opencode-cli (skip)")
                    io.say(f"  ✓ {w.name}: opencode-cli (no probe)")
                    continue
                ok, detail = probe_worker(w, k)
                probes[w.name] = (ok, detail)
                mark = "✓" if ok else "✗"
                io.say(f"  {mark} {w.name}: {detail}")
                if not ok:
                    io.say(f"     ↳ {w.name} is unreachable; doctor will re-test later.")
    else:
        io.say("Q4 — Skipping probes (--no-probe)")

    return {"selected": selected, "keys": keys, "probes": probes}


def init_command(args: argparse.Namespace, io: IO | None = None) -> int:
    """Entry point for `anchor init`."""
    io = io or IO()
    groups = build_channel_groups()
    result = run_wizard(groups, io, skip_probes=getattr(args, "no_probe", False))

    selected = result["selected"]
    keys = result["keys"]
    anchor_home = _anchor_home()
    workers_yaml_path = _workers_yaml_path()
    keys_yaml_path = _keys_yaml_path()
    if not selected:
        # user picked nothing — still write a "no channels" config so
        # downstream `anchor doctor` has something to validate.
        os.makedirs(anchor_home, exist_ok=True)
        with open(workers_yaml_path, "w") as f:
            f.write(emit_workers_yaml({}))
        _ensure_secure(workers_yaml_path)
        io.say("Wrote an empty workers.yaml; nothing else to do.")
        return 0

    os.makedirs(anchor_home, exist_ok=True)
    with open(workers_yaml_path, "w") as f:
        f.write(emit_workers_yaml(selected))
    _ensure_secure(workers_yaml_path)

    if keys:
        with open(keys_yaml_path, "w") as f:
            f.write(emit_keys_yaml(keys))
        _ensure_secure(keys_yaml_path)
        io.say(f"  Wrote {keys_yaml_path} (chmod 600)")
    io.say(f"  Wrote {workers_yaml_path} (chmod 600)")

    # Print a single-line next-step hint, no fluff.
    io.say("")
    io.say(f"Next:  anchor doctor   (verify)  →  anchor serve   (start gateway)")
    return 0


# ---------------------------------------------------------------------------
# Parser glue
# ---------------------------------------------------------------------------


def add_init_subparser(sub) -> None:
    p = sub.add_parser(
        "init",
        help="interactive 5-min onboarding wizard",
        description=(
            "Walk through picking providers, entering API keys, and "
            "selecting models. Writes ~/.anchor/{workers,keys}.yaml."
        ),
    )
    p.add_argument(
        "--no-probe", action="store_true",
        help="skip connectivity probes (e.g. in CI / air-gapped install)",
    )
    p.set_defaults(_cli_init=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("import as anchor.cli_init; wire into anchor.cli")
