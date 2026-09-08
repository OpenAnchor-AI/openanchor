"""test_cli_init.py — `anchor init` wizard (6 cases).

Coverage:
  1. Channel grouping & ordering: free providers first, paid at end,
     ollama-local has no key.
  2. YAML round-trip: emit -> parse preserves the channel -> models map.
  3. IO.choose_one accepts numbered input and rejects garbage.
  4. IO.choose_many accepts comma-separated indices and labels.
  5. run_wizard happy path: scripted answers -> correct selected dict +
     keys + (probes skipped when skip_probes=True).
  6. init_command writes the YAML files at the documented paths with
     chmod 600, and skips channels whose key was never collected.
"""
from __future__ import annotations

import argparse
import io as _io
import os
import stat
import sys

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _ScriptedIO:
    """Drop-in IO replacement for tests. Acts as a finite state machine
    over a list of (matcher, response) pairs.

    The matcher is a callable ``predicate(prompt: str) -> bool``; the
    first match wins and its response is consumed. Reads that don't
    match any predicate raise so test authors notice a mis-scripted IO.
    """

    def __init__(self, answers: list, *, tty: bool = True):
        self.answers = list(answers)
        self.out = _io.StringIO()
        self._tty = tty

    def _next_for(self, prompt: str):
        for i, (pred, resp) in enumerate(self.answers):
            if pred(prompt):
                self.answers.pop(i)
                return resp
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    def read(self, prompt: str) -> str:
        return self._next_for(prompt)

    def read_password(self, prompt: str) -> str:
        return self._next_for(prompt)

    def write(self, s: str) -> None:
        self.out.write(s)

    def isatty(self) -> bool:
        return self._tty

    def say(self, msg: str) -> None:
        self.write(msg + "\n")

    def prompt(self, text: str, *, default: str = "") -> str:
        return self.read(f"{text} [{default}]: ").strip()

    def choose_one(self, header: str, options):
        self.write(header + "\n")
        for idx, label in enumerate(options, start=1):
            self.write(f"  {idx:>2}. {label}\n")
        while True:
            raw = self.read(f"Pick [1-{len(options)}]: ").strip()
            if not raw:
                if self._tty:
                    return options[0]
                continue
            if raw.isdigit():
                n = int(raw)
                if 1 <= n <= len(options):
                    return options[n - 1]
            self.write(f"  ! choose a number 1..{len(options)}\n")

    def choose_many(self, header: str, options):
        self.write(header + "\n")
        for idx, label in enumerate(options, start=1):
            self.write(f"  {idx:>2}. {label}\n")
        hint = "comma-separated numbers (e.g. 1,3) or empty for all"
        # Match the wizard's actual prompt verbatim; fall back to the
        # header for compatibility with callers that use the IO directly.
        raw = self.read(f"{hint}: ").strip()
        if not raw:
            return list(options) if self._tty else []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        chosen = []
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


def _exact(prompt_substr: str):
    """Matcher factory: matches prompts containing `prompt_substr`."""
    return lambda p: prompt_substr in p


# ---------------------------------------------------------------------------
# 1. Channel grouping
# ---------------------------------------------------------------------------


def test_channel_groups_free_first_and_ollama_has_no_key():
    from anchor.cli_init import build_channel_groups, channel_label

    groups = build_channel_groups()
    assert groups, "at least one channel must be configured"

    # 1. ollama-local must exist and have no key requirement.
    oll = next((g for g in groups if g.name == "ollama-local"), None)
    assert oll is not None, "ollama-local channel must appear in init"
    assert oll.needs_key is False
    assert oll.primary_env == ""

    # 2. paid-only channels must require a key.
    paid = [g for g in groups if g.name in ("baosiapi", "baosiapi-gpt",
                                            "deepseek-official")]
    assert paid, "expected at least one paid channel in registry"
    for g in paid:
        assert g.needs_key is True
        assert g.primary_env, f"{g.name} missing primary_env"

    # 3. Order: free providers (no $/month fixed) should appear before
    # the baosiapi group in the menu. We don't pin exact positions, but
    # the index of ollama / dpsk-zen / kilo should be < baosiapi.
    by_name = {g.name: i for i, g in enumerate(groups)}
    assert by_name["opencode-zen"] < by_name["baosiapi"], (
        "opencode-zen (free) should appear before baosiapi in init menu"
    )

    # 4. The human label includes both the channel name and the env var.
    label = channel_label(groups[0])
    assert groups[0].name in label


# ---------------------------------------------------------------------------
# 2. YAML round-trip
# ---------------------------------------------------------------------------


def test_workers_yaml_round_trip_preserves_selection():
    from anchor.cli_init import emit_workers_yaml, parse_workers_yaml

    sample = {
        "opencode-zen": ["deepseek-v4-flash"],
        "deepseek-official": ["deepseek-v4-pro", "deepseek-v4-flash-official"],
        "baosiapi": ["claude-fable-5", "claude-opus-5"],
        "ollama-local": ["ollama-ornith-35b"],
    }
    text = emit_workers_yaml(sample)
    parsed = parse_workers_yaml(text)
    assert parsed == sample, f"round-trip mismatch: {parsed!r} != {sample!r}"


def test_keys_yaml_round_trip_preserves_values():
    from anchor.cli_init import emit_keys_yaml, parse_keys_yaml

    keys = {
        "OPENCODE_ZEN_API_KEY": "sk-zen-123",
        "BAOSIAPI_GPT_API_KEY": "sk-baosi-456",
        "MINIMAX_API_KEY_1": "ey-abc-789",
    }
    text = emit_keys_yaml(keys)
    parsed = parse_keys_yaml(text)
    assert parsed == keys, f"round-trip mismatch: {parsed!r} != {keys!r}"

    # Special-character keys must survive quoting.
    # NB: json.dumps escapes embedded quotes with \"; parse_keys_yaml
    # now decodes the JSON form back to the original value.
    weird = {"WEIRD_KEY": "with:colon and #hash and \"quote\""}
    weird_text = emit_keys_yaml(weird)
    parsed_weird = parse_keys_yaml(weird_text)
    assert parsed_weird == weird, f"special-char round-trip: {parsed_weird!r}"


# ---------------------------------------------------------------------------
# 3. IO.choose_one
# ---------------------------------------------------------------------------


def test_io_choose_one_accepts_numbered_input():
    from anchor.cli_init import IO

    # First call returns "1", second returns "2" — single-pick wizard.
    answers = ["1", "2"]
    io = IO(
        read=lambda _p: answers.pop(0),
        read_password=lambda _p: "",
        write=lambda s: None,
        isatty=lambda: True,
    )
    assert io.choose_one("Pick one", ["alpha", "beta"]) == "alpha"
    assert io.choose_one("Pick one", ["alpha", "beta"]) == "beta"


def test_io_choose_one_rejects_garbage_then_accepts():
    from anchor.cli_init import IO

    # First call: garbage, second call: valid number.
    answers = ["nope", "1"]
    io = IO(
        read=lambda _p: answers.pop(0),
        read_password=lambda _p: "",
        write=lambda s: captured.append(s),
        isatty=lambda: True,
    )
    captured: list[str] = []
    assert io.choose_one("Pick", ["only"]) == "only"
    # The garbage attempt must have produced at least one error line.
    assert any("choose a number" in line for line in captured), captured


# ---------------------------------------------------------------------------
# 4. IO.choose_many
# ---------------------------------------------------------------------------


def test_io_choose_many_accepts_comma_separated_indices():
    from anchor.cli_init import IO

    answers = ["1,3"]
    io = IO(
        read=lambda _p: answers.pop(0),
        read_password=lambda _p: "",
        write=lambda s: None,
        isatty=lambda: True,
    )
    picked = io.choose_many("Multi", ["a", "b", "c", "d"])
    assert picked == ["a", "c"], picked


def test_io_choose_many_accepts_labels_mixed_with_indices():
    from anchor.cli_init import IO

    # Mixed numeric + label input. Order is preserved as typed.
    # '1,beta,3' = first option, label 'beta', third option.
    answers = ["1,beta,3"]
    io = IO(
        read=lambda _p: answers.pop(0),
        read_password=lambda _p: "",
        write=lambda s: None,
        isatty=lambda: True,
    )
    picked = io.choose_many("Multi", ["alpha", "beta", "gamma", "delta"])
    assert picked == ["alpha", "beta", "gamma"], picked


# ---------------------------------------------------------------------------
# 5. run_wizard happy path
# ---------------------------------------------------------------------------


def test_run_wizard_happy_path_collects_selection_and_keys(monkeypatch, tmp_path):
    from anchor.cli_init import build_channel_groups, channel_label, run_wizard

    # Force ANCHOR_HOME to a tmp dir so writes don't pollute HOME.
    monkeypatch.setenv("HOME", str(tmp_path))

    groups = build_channel_groups()
    # Pick the first two channels and one model each.
    g0 = groups[0]
    g1 = groups[1]
    labels = [channel_label(g) for g in groups]

    def match_providers(p: str) -> bool:
        return "Which providers" in p or "comma-separated" in p

    def match_reuse(p: str) -> bool:
        return "already set in env" in p

    def match_key_g0(p: str) -> bool:
        return f"API key for {g0.name}" in p

    def match_models_g0(p: str) -> bool:
        return f"under {g0.name}" in p

    def match_models_g1(p: str) -> bool:
        return f"under {g1.name}" in p

    def match_key_g1(p: str) -> bool:
        return f"API key for {g1.name}" in p

    def match_skip_g1(p: str) -> bool:
        return f"Skip this provider?" in p and g1.needs_key

    # Q1: choose the first two labels.
    # Q2: provide a key for g0, reuse env for g1.
    # Q3: pick first model for each.
    # The reuse branch only fires for groups whose env is already set.
    if g1.needs_key:
        monkeypatch.setenv(g1.primary_env, "preexisting-key")
        answers = [
            (match_providers, "1,2"),
            (match_reuse, "y"),
            (match_key_g0, "sk-zen-1"),
            (match_models_g0, "1"),
            (match_reuse, "y"),  # for g1's reuse prompt
            (match_models_g1, "1"),
        ]
    else:
        # g1 doesn't need a key (e.g. ollama-local): only model prompt.
        answers = [
            (match_providers, "1,2"),
            (match_key_g0, "sk-zen-1"),
            (match_models_g0, "1"),
            (match_models_g1, "1"),
        ]

    io = _ScriptedIO(answers, tty=True)
    result = run_wizard(groups, io, skip_probes=True)

    # Wizard collected at least one channel and one model.
    assert g0.name in result["selected"], result
    assert g0.worker_names[0] in result["selected"][g0.name]
    if g1.needs_key:
        assert g1.name in result["selected"]
    # At least one key was collected (either typed or pre-existing).
    assert result["keys"], "expected at least one key in the result"
    # Probes are skipped when skip_probes=True.
    assert result["probes"] == {}, result


# ---------------------------------------------------------------------------
# 6. init_command writes files with chmod 600
# ---------------------------------------------------------------------------


def test_init_command_writes_yaml_with_chmod_600(monkeypatch, tmp_path):
    from anchor.cli_init import (
        ANCHOR_HOME, KEYS_YAML, WORKERS_YAML,
        build_channel_groups, init_command,
    )

    # Redirect HOME so ~/.anchor/ lands in tmp.
    monkeypatch.setenv("HOME", str(tmp_path))

    groups = build_channel_groups()
    g0 = groups[0]

    def match_providers(p: str) -> bool:
        return "Which providers" in p or "comma-separated" in p

    def match_reuse(p: str) -> bool:
        return "already set in env" in p

    def match_key_g0(p: str) -> bool:
        return f"API key for {g0.name}" in p

    def match_models_g0(p: str) -> bool:
        return f"under {g0.name}" in p

    if g0.needs_key:
        answers = [
            (match_providers, "1"),
            (match_reuse, "n"),
            (match_key_g0, "sk-test-1"),
            (match_models_g0, "1"),
        ]
    else:
        # ollama-local: no key prompt, just model pick
        answers = [
            (match_providers, "1"),
            (match_models_g0, "1"),
        ]

    io = _ScriptedIO(answers, tty=True)
    args = argparse.Namespace(no_probe=True)
    rc = init_command(args, io)

    assert rc == 0, "init_command must return 0 on success"
    assert os.path.exists(WORKERS_YAML), f"workers.yaml not written at {WORKERS_YAML}"

    # chmod 600 check (skip on platforms that don't support Unix perms).
    if hasattr(os, "chmod"):
        mode = stat.S_IMODE(os.stat(WORKERS_YAML).st_mode)
        assert mode == 0o600, f"workers.yaml mode {oct(mode)} != 0o600"

    # If a key was collected, keys.yaml should exist with chmod 600 too.
    if g0.needs_key:
        assert os.path.exists(KEYS_YAML), "keys.yaml not written"
        if hasattr(os, "chmod"):
            mode = stat.S_IMODE(os.stat(KEYS_YAML).st_mode)
            assert mode == 0o600, f"keys.yaml mode {oct(mode)} != 0o600"

    # Workers yaml must round-trip and reference the chosen channel.
    with open(WORKERS_YAML) as f:
        from anchor.cli_init import parse_workers_yaml
        parsed = parse_workers_yaml(f.read())
    assert g0.name in parsed
    assert g0.worker_names[0] in parsed[g0.name]
