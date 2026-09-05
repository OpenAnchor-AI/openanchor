"""Server lifespan + background loops (extracted from server.py)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os

from fastapi import FastAPI

_logger = logging.getLogger("anchor.lifespan")



# v0.9.10: auto-recovery background task (Sonnet 5 #2 priority)
_RECOVERY_INTERVAL_SEC = 60

async def _auto_recovery_loop():
    """Background task: every 60s, ping cooling workers with a tiny probe
    to detect actual recovery, and prune expired cooldowns.

    Real recovery test = send a 5-token chat to the worker; if success,
    unblock it. Otherwise keep cooling.
    """
    from anchor.cooldown import all_cooling, _state
    from anchor.config import WORKERS as _W
    from anchor.clients.factory import build_client as _bc
    while True:
        try:
            await asyncio.sleep(_RECOVERY_INTERVAL_SEC)
            cooling = all_cooling()
            if not cooling:
                continue
            for worker_name, remaining in list(cooling.items()):
                # If still has > 30s remaining, skip
                if remaining > 30:
                    continue
                w = next((x for x in _W if x.name == worker_name), None)
                if not w:
                    continue
                try:
                    client = _bc(w)
                    # Probe with minimal query
                    r = await client.chat(
                        [{"role": "user", "content": "ping"}],
                        max_tokens=5, temperature=0.0,
                    )
                    content = (r.get("content") or "").strip()
                    # If non-empty and not error marker, recovered
                    if content and not content.startswith("[error"):
                        _state.pop(worker_name, None)
                        _logger.info(f"[recovery] {worker_name} recovered, cooldown cleared")
                except Exception:
                    # Still failing; let TTL expire naturally
                    pass
        except Exception as e:
            # Never let this loop die
            _logger.exception("recovery_loop_err: %s", type(e).__name__)


# v0.9.14 (Sonnet 5 acceptance #4): weekly calibration refresh cron.
# 7 days = 604800s. Default min_n=3 (sparser data still useful).
import asyncio as asyncio_cron
_CALIB_INTERVAL_SEC = 7 * 86400  # weekly

async def _auto_calibration_loop():
    while True:
        try:
            await asyncio_cron.sleep(_CALIB_INTERVAL_SEC)
            from anchor.calibration import refresh as _calib_refresh_cron
            stats = _calib_refresh_cron(min_n=3, days=7)
            _logger.info(f"[calib_cron] refresh: {stats}")
        except ImportError as _cie:
            # L2 (maintainability MEDIUM): missing optional dep for calibration
            # (e.g. scipy). Surface at ERROR — otherwise calibration silently
            # breaks every 7 days.
            _logger.error(
                "calib_cron IMPORT_ERROR: missing dependency (%s). "
                "Reinstall anchor or check deps. Loop will retry in 7 days.",
                _cie,
            )
        except Exception as _ce:
            _logger.exception("calib_cron_err: %s", type(_ce).__name__)


# v0.9.55: quarantine recheck loop. Probes currently quarantined
# workers with a tiny PONG-style request; if they answer cleanly, the
# quarantine is cleared. This is the safety net for any false-positive
# from the sweep.
_QUARANTINE_RECHECK_INTERVAL_SEC = int(
    os.environ.get("ANCHOR_QUARANTINE_RECHECK_INTERVAL_SEC", "180")
)


async def _auto_quarantine_recheck_loop():
    """Every N seconds, probe quarantined workers; clear if healthy."""
    from anchor.release.circuit_breaker import (
        list_quarantined as _lq, clear_quarantine as _cq,
        update_quarantine_probe as _uqp,
    )
    from anchor.clients.factory import build_client as _bc_q
    from anchor.config import WORKERS as _W_Q
    while True:
        try:
            await asyncio.sleep(_QUARANTINE_RECHECK_INTERVAL_SEC)
            quarantined = _lq() or {}
            for worker_name, info in list(quarantined.items()):
                w = next((x for x in _W_Q if x.name == worker_name), None)
                if not w:
                    continue
                ok = False
                try:
                    client = _bc_q(w)
                    r = await client.chat(
                        [{"role": "user", "content": "ping"}],
                        max_tokens=5, temperature=0.0,
                    )
                    content = (r.get("content") or "").strip()
                    finish_reason = r.get("finish_reason")
                    ok = (
                        bool(content)
                        and not content.startswith(("[error", "[stub"))
                        and finish_reason in (None, "stop")
                    )
                except Exception:
                    ok = False
                _uqp(worker_name, ok)
                if ok:
                    _cq(worker_name)
                    _logger.info(
                        "[quarantine_recheck] %s PONG ok, cleared", worker_name,
                    )
                else:
                    _logger.info(
                        "[quarantine_recheck] %s still unhealthy, keeping", worker_name,
                    )
        except ImportError as _qie:
            # L2 (maintainability MEDIUM): missing optional dep. Operator alert
            # at ERROR (not WARNING) so the loop's silent-failure mode is
            # visible in JSON logs. Loop still retries next interval.
            _logger.error(
                "quarantine_recheck IMPORT_ERROR: missing dependency (%s). "
                "Reinstall anchor or check deps. Loop will retry next interval.",
                _qie,
            )
        except Exception as _qre:
            _logger.exception("quarantine_recheck_err: %s", type(_qre).__name__)



# v0.9.56 (PR-A3): archive old session logs so load_sessions() doesn't
# re-read weeks of JSONL every 5 minutes. Keeps the hot directory lean.
_SESSION_ARCHIVE_INTERVAL_SEC = int(
    os.environ.get("ANCHOR_SESSION_ARCHIVE_INTERVAL_SEC", str(6 * 3600))
)
_SESSION_ARCHIVE_RETENTION_DAYS = int(
    os.environ.get("ANCHOR_SESSION_ARCHIVE_RETENTION_DAYS", "7")
)


async def _auto_archive_sessions_loop():
    """Periodically gzip JSONL files older than retention into _archive/."""
    import gzip
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    while True:
        try:
            await asyncio.sleep(_SESSION_ARCHIVE_INTERVAL_SEC)
            from anchor.session_log import SESSIONS_DIR as _sd
            if not _sd.exists():
                continue
            archive_dir = _sd / "_archive"
            archive_dir.mkdir(exist_ok=True)
            cutoff = (_dt.now(_tz.utc) - _td(days=_SESSION_ARCHIVE_RETENTION_DAYS)).date()
            archived = 0
            for fp in sorted(_sd.glob("*.jsonl")):
                try:
                    fdate = _dt.strptime(fp.stem, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if fdate >= cutoff:
                    continue
                gz_path = archive_dir / (fp.name + ".gz")
                if gz_path.exists():
                    continue
                with fp.open("rb") as _in, gzip.open(gz_path, "wb") as _out:
                    while True:
                        chunk = _in.read(64 * 1024)
                        if not chunk:
                            break
                        _out.write(chunk)
                fp.unlink()
                archived += 1
            if archived:
                _logger.info("[session_archive] archived %d files to %s",
                              archived, archive_dir)
        except Exception as _arch_e:
            _logger.exception("session_archive_err: %s", type(_arch_e).__name__)


# T-AUDIT-11 (S17): periodic head snapshot writer. startup _lifespan() reads
# the snapshot, but `head.update()` mutates W + prior_table in-process only.
# Single-worker deploys lose all learned state on restart unless we write
# back. Every 60s (same cadence as recovery probe — cost ~negligible: one
# np.savez of ~10 KB). Multi-worker deploys still diverge (Redis is the
# fix there, deferred to P2); this at least gives single-worker ops a
# sensible warm restart.
_HEAD_SAVE_INTERVAL_SEC = 60
async def _auto_head_save_loop():
    while True:
        try:
            await asyncio.sleep(_HEAD_SAVE_INTERVAL_SEC)
            from pathlib import Path as _P_hs
            # T-AUDIT-11: write JSON format (NOT np.savez) so the file is
            # readable by load_from_snapshot() at next startup. head.save()
            # writes .npz which load_from_snapshot (JSON reader) cannot parse;
            # drift.snapshot() writes the canonical JSON layout.
            from anchor.drift import snapshot as _drift_snap_save
            from anchor.head import (
                W as _hs_W, prior_table as _hs_prior,
            )
            _snap = _P_hs.home() / "anchor" / "data" / "head_baseline.json"
            _drift_snap_save(_hs_prior, _hs_W, _snap)
            _logger.info("head snapshot saved: %s", _snap)
        except Exception as _hse:
            _logger.exception("head_save_err: %s", type(_hse).__name__)


# v0.9.47: quarantine sweep cron — replaces old inline update_err_rate_quarantine
# inside compute_sla() that /admin/sla was mutating state on read.
_QUARANTINE_SWEEP_INTERVAL_SEC = 300  # 5 minutes
async def _auto_quarantine_sweep_loop():
    while True:
        try:
            await asyncio.sleep(_QUARANTINE_SWEEP_INTERVAL_SEC)
            from anchor.admin_ops import _cron_quarantine_sweep as _qs
            stats = _qs()
            if stats:
                _logger.info("[quarantine_sweep] %s", stats)
        except Exception as _qe:
            _logger.exception("quarantine_sweep_err: %s", type(_qe).__name__)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Server lifespan: startup dependency checks + background loops + drain.

    Startup (C-04): load config, verify at least 1 enabled worker has a key,
    init the DB, load the head snapshot from disk. Any failure here aborts
    startup with a clear error message (process exits 1).

    Body: auto-recovery (60s) + auto-calibration (7d) loops run in background.

    Shutdown: drain in-flight by setting a sentinel + cancelling background
    tasks. New requests after shutdown starts receive 503 (handled below).
    """
    _ls_log = _logger

    # --- Startup dependency checks (C-04) ---
    try:
        from anchor import config as _cfg
        _enabled = _cfg.enabled_workers()
        if not _enabled:
            raise RuntimeError("no enabled workers in config — refusing to start")
        # Best-effort: check keys exist for at least one enabled worker.
        # Don't fail startup on missing keys (graceful: worker will just error
        # at request time), but do log it so operators notice.
        _missing = []
        for _w in _enabled:
            _envs = list(getattr(_w, "api_key_envs", ())) + [getattr(_w, "api_key_env", "")]
            _envs = [e for e in _envs if e]
            if not any(os.environ.get(e, "").strip() for e in _envs):
                _missing.append(_w.name)  # noqa: F841
        if _missing:
            _ls_log.warning("workers missing API keys: %s", _missing)
    except Exception as _start_e:
        _ls_log.exception("startup config check failed")
        raise

    try:
        from anchor.db import initdb as _initdb
        _initdb()
        _ls_log.info("db initialized")
    except Exception as _db_e:
        _ls_log.exception("startup db init failed")
        raise

    # P5.5 doctrine: probe each enabled worker via /chat/completions max_tokens=1.
    # P5.6: ANCHOR_VERIFY_RUNTIME=0 disables the probe at startup (best-effort:
    # log unhealthy workers but NEVER block boot).
    import os as _os_vr
    if _os_vr.environ.get("ANCHOR_VERIFY_RUNTIME", "1") == "0":
        _ls_log.info("worker probe skipped via ANCHOR_VERIFY_RUNTIME=0")
    else:
        try:
            from anchor.config import verify_runtime as _verify_rt
            # FastAPI best-practice: verify_runtime uses sync urllib (blocking).
            # Offload to threadpool so startup doesn't block the event loop.
            from starlette.concurrency import run_in_threadpool as _run_tp
            results = await _run_tp(_verify_rt)
            ok_n = sum(1 for v in results.values() if v == "ok")
            _ls_log.info("worker probe: %d/%d ok", ok_n, len(results))
            bad = {k: v for k, v in results.items() if v != "ok"}
            if bad:
                _ls_log.warning("worker probe: %d unhealthy at boot: %s",
                                len(bad), bad)
        except Exception as _vr_e:
            _ls_log.warning("verify_runtime skipped at startup: %s", _vr_e)

    # v0.9.46d: reset prior_table to baseline so server starts from cold-start
    # state every time (not mutated state from previous process).
    from anchor.head import reset_to_baseline as _rtb
    _rtb()

    # T-AUDIT-04 (SRE C-01): startup-only state bootstrap, NOT runtime
    # coordination. With uvicorn --workers N, each worker loads the same
    # snapshot at startup, then diverges after its first head.update()
    # call. Single-worker deploys get full benefit (cold-start picks up
    # learned priors from the last run); multi-worker deploys do NOT get
    # runtime consistency. Full coordination requires Redis (P2 backlog).
    try:
        from anchor.head import load_from_snapshot as _lfs
        from pathlib import Path as _P
        _snap = _P.home() / "anchor" / "data" / "head_baseline.json"
        _lfs_out = _lfs(_snap)
        _ls_log.info(
            "head snapshot: %s (loaded=%s, n_priors=%s). "
            "NOTE: startup-only state, NOT runtime coordination across "
            "uvicorn workers.",
            _snap, _lfs_out.get("loaded"), _lfs_out.get("n_priors"),
        )
    except Exception as _snap_e:
        _ls_log.warning("head snapshot load skipped: %s", _snap_e)

    tasks = [
        asyncio.create_task(_auto_recovery_loop()),
        asyncio.create_task(_auto_calibration_loop()),
        asyncio.create_task(_auto_head_save_loop()),
        asyncio.create_task(_auto_quarantine_sweep_loop()),
        asyncio.create_task(_auto_quarantine_recheck_loop()),
        asyncio.create_task(_auto_archive_sessions_loop()),
    ]
    _ls_log.info(
        "auto-recovery (60s) + auto-calibration (7d) + head-save (60s) + quarantine-sweep (300s) + quarantine-recheck (180s) + session-archive (6h) loops started; %d enabled workers",
        len(_enabled),
    )
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        from anchor.clients.pool import shutdown as _pool_shutdown
        await _pool_shutdown()




def attach_lifespan(app: FastAPI) -> None:
    """Bind the lifespan context manager onto a FastAPI app."""
    app.router.lifespan_context = _lifespan
