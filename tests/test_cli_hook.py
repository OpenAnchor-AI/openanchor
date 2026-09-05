"""test_cli_hook.py — install/uninstall Codex/OpenCode/Claude/agy wrappers (8 cases).

The hook installer mutates the user's home directory (`~/anchor/hooks/`).
Tests must redirect ANCHOR_HOME to a tmp_path so they don't touch real files.
"""
from __future__ import annotations

import pytest

from anchor import cli_hook


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Pretend ANCHOR_HOME = tmp_path/anchor (no real ~/anchor writes)."""
    fake = tmp_path / "anchor"
    monkeypatch.setattr(cli_hook, "ANCHOR_HOME", fake)
    monkeypatch.setattr(cli_hook, "HOOKS_DIR", fake / "hooks")
    return fake


def test_detect_clis_reports_all_four_supported(monkeypatch):
    """Always returns one entry per supported CLI."""
    monkeypatch.setattr(cli_hook, "_check_cli", lambda name: True)
    detected = cli_hook.detect_clis()
    assert set(detected.keys()) == set(cli_hook.SUPPORTED_CLIS)
    assert all(detected.values())


def test_install_hooks_dry_run_creates_no_wrapper_files(fake_home):
    """dry_run=True must not create wrapper files but still reports the plan."""
    out = cli_hook.install_hooks(dry_run=True)
    # Hooks dir may not exist, OR exists but has no wrappers
    hooks = fake_home / "hooks"
    if hooks.exists():
        assert not any(hooks.iterdir()), (
            "dry_run must not create wrappers, found: " + str(list(hooks.iterdir()))
        )
    # Each supported CLI is in the result dict
    assert set(out.keys()) == set(cli_hook.SUPPORTED_CLIS)


def test_install_hooks_writes_wrappers_for_present_clis(fake_home, monkeypatch):
    """Present CLIs get wrapper files; missing CLIs are reported as 'not installed'."""
    monkeypatch.setattr(cli_hook, "_check_cli", lambda name: True)
    monkeypatch.setattr(cli_hook.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    out = cli_hook.install_hooks()
    assert all("not installed" not in v for v in out.values())
    for cli in cli_hook.SUPPORTED_CLIS:
        wrapper = fake_home / "hooks" / cli
        assert wrapper.exists()
        assert wrapper.stat().st_mode & 0o777 == 0o755
        # Wrapper content includes the wrapped binary path
        content = wrapper.read_text()
        assert f"/usr/local/bin/{cli}" in content
        assert "exec" in content


def test_install_hooks_skips_missing_clis(fake_home, monkeypatch):
    """CLIs not installed are reported as 'not installed' (no wrapper file)."""
    monkeypatch.setattr(cli_hook, "_check_cli", lambda name: False)
    out = cli_hook.install_hooks()
    for v in out.values():
        assert v == "not installed"
    hooks = fake_home / "hooks"
    # Either hooks dir wasn't created or it exists but is empty
    if hooks.exists():
        assert not any(hooks.iterdir())


def test_install_hooks_idempotent(fake_home, monkeypatch):
    """Re-running install on top of itself overwrites the wrappers cleanly."""
    monkeypatch.setattr(cli_hook, "_check_cli", lambda name: True)
    monkeypatch.setattr(cli_hook.shutil, "which", lambda name: "/bin/echo")
    cli_hook.install_hooks()
    cli_hook.install_hooks()  # second run must not raise
    for cli in cli_hook.SUPPORTED_CLIS:
        assert (fake_home / "hooks" / cli).exists()


def test_uninstall_hooks_no_directory(fake_home):
    """No hooks directory → returns 0, no error."""
    assert cli_hook.uninstall_hooks() == 0


def test_uninstall_hooks_removes_only_supported(fake_home):
    """Uninstall removes only the 4 supported CLI wrappers, not user files."""
    hooks = fake_home / "hooks"
    hooks.mkdir(parents=True)
    # Supported wrappers
    for cli in cli_hook.SUPPORTED_CLIS:
        (hooks / cli).write_text("# wrapper")
    # User file with the same name as a CLI but different content
    user_file = hooks / "my-notes.md"
    user_file.write_text("user content")

    removed = cli_hook.uninstall_hooks()
    assert removed == 4
    assert not any((hooks / cli).exists() for cli in cli_hook.SUPPORTED_CLIS)
    assert user_file.exists()  # user file untouched


def test_install_uninstall_roundtrip(fake_home, monkeypatch):
    """End-to-end: install then uninstall returns the count of removed wrappers."""
    monkeypatch.setattr(cli_hook, "_check_cli", lambda name: True)
    monkeypatch.setattr(cli_hook.shutil, "which", lambda name: "/bin/echo")
    cli_hook.install_hooks()
    n = cli_hook.uninstall_hooks()
    assert n == len(cli_hook.SUPPORTED_CLIS)
    assert not (fake_home / "hooks").exists() or not any(
        (fake_home / "hooks").iterdir()
    )