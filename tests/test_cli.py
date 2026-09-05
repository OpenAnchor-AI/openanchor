"""test_cli.py — argparse entry point for the `anchor` CLI (4 cases).

`anchor serve`, `anchor config`, `anchor smoke`, `anchor cron` are the
shipping surfaces for ops. The version flag is the cheapest happy path.
We invoke main() directly so we don't shell out.
"""
from __future__ import annotations

import pytest

from anchor.cli import _build_parser, main


def test_parser_top_level_help():
    """A bare `anchor` invocation prints help and returns 0."""
    rc = main([])
    assert rc == 0


def test_parser_version_flag():
    """--version prints the package version and exits 0."""
    rc = main(["--version"])
    assert rc == 0


def test_parser_unknown_subcommand_calls_sysexit():
    """argparse rejects unknown subcommands with SystemExit(2) — that's by design.

    Documenting the behaviour so a future refactor that suppresses it
    (e.g. to fall through to print_help) gets flagged.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["nope"])
    assert exc_info.value.code == 2


def test_parser_accepts_known_subcommands():
    """Each shipping subcommand must be in the parser, not raise."""
    p = _build_parser()
    for cmd in ("serve", "config", "smoke", "cron"):
        parsed = p.parse_args([cmd])
        assert parsed.command == cmd


def test_parser_serve_host_port_defaults():
    """`anchor serve` defaults to 127.0.0.1:8088, no reload."""
    p = _build_parser()
    parsed = p.parse_args(["serve"])
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 8088
    assert parsed.reload is False


def test_parser_serve_custom_port():
    p = _build_parser()
    parsed = p.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])
    assert parsed.host == "0.0.0.0"
    assert parsed.port == 9000
    assert parsed.reload is True


def test_parser_cron_collects_remainder():
    """`anchor cron list foo bar` captures the tail into args."""
    p = _build_parser()
    parsed = p.parse_args(["cron", "list", "foo", "bar"])
    assert parsed.command == "cron"
    assert parsed.args == ["list", "foo", "bar"]