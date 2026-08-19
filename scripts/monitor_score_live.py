"""Daily score_live.py monitor (B4): run the real live-scoring entry point
once a day, log its full output, and diff against the prior day's log so
a change that isn't obviously explained gets surfaced rather than missed
in a scroll of daily output.

Deliberately thin: this does NOT reimplement any of score_live.py's logic
-- it shells out to the exact same `python -m scripts.score_live` a human
would run, captures stdout+stderr+exit code verbatim, and treats that
transcript as the unit to log and diff. Whatever score_live.py does or
doesn't do (including failing its own _preflight check) is exactly what
gets recorded -- this script adds logging, diffing, a non-mutation guard,
and gap detection. Nothing else.

Known, expected near-term behavior (not a bug in this script): with
config/stock_list.py still empty and config/hold_period.py::
WINDOW_START_DATE still unset (baseline_system_architecture.md §5,
"Blocked on External Input"), every run until those are populated will
fail the same _preflight check the same way -- the diff will correctly
report "no change" every day until that's true, and the day it stops
failing IS the notable change this script exists to catch. `--tickers`
overrides the STOCK_LIST-empty half of that (for exactly this reason
score_live.py itself accepts it, see its own docstring) but NOT
WINDOW_START_DATE -- that gate is unconditional in score_live.py's own
_preflight and this script does not, and should not, work around it.

NON-MUTATION GUARD: score_live.py only ever READS
data/processed/portfolio_state.json (confirmed by inspection --
`load_portfolio_state` is called, `save_portfolio_state` never is) and
never writes anything under config/. This script hashes both before and
after every real invocation and raises LiveStateMutationError (loud,
distinct exit code, logged before crashing) if either changed --
guarding the one failure mode worse than any missed day: an unattended
monitoring run silently corrupting the live state it's only supposed to
observe.

Trading-day detection: weekends plus a manually-maintained NYSE holiday
list (NYSE_HOLIDAYS below) -- a small, bounded, project-owned calendar,
not a market-calendar dependency, consistent with this project's existing
weekends-only-by-default convention (config/hold_period.py::
period_number_for) but more precise where it matters here: a
misidentified holiday would otherwise show up as a false "gap".

GAP CHECK: every run scans the full log history and flags any expected
trading day with no log file at all -- distinct from a day that ran and
correctly logged SKIPPED, which means the monitor DID execute and
recognized a non-trading day. A gap means the monitor didn't run at all
(machine off, cron didn't fire) -- the failure mode nobody watching only
the latest log would ever notice on their own.

Usage:
    python -m scripts.monitor_score_live
    python -m scripts.monitor_score_live --force              # run even on a non-trading day (testing)
    python -m scripts.monitor_score_live --tickers AAPL MSFT  # placeholder universe, passed through to score_live.py
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "data" / "processed" / "live_monitor_logs"
CONFIG_DIR = REPO_ROOT / "config"

from src.selection.portfolio_state import DEFAULT_PATH as PORTFOLIO_STATE_PATH  # noqa: E402

# Manually maintained, project-owned NYSE holiday calendar -- NOT derived
# from a market-calendar library (ten fixed dates a year didn't justify a
# new dependency). Each date checked by hand against its claimed weekday
# before being trusted (e.g. 2026-01-01 = Thursday), not copied from
# memory alone. Extend for 2027+ if this is still running past year-end --
# a missing future year fails SAFE: an unlisted real holiday just reads as
# an ordinary expected trading day, producing one extra (correctly
# explained by score_live.py's own output, e.g. stale/unchanged data)
# log entry, not a crash and not a silently-wrong gap-check flag.
NYSE_HOLIDAYS: set[str] = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Washington's Birthday (Presidents Day)
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth National Independence Day
    "2026-07-03",  # Independence Day (observed -- July 4, 2026 falls on a Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving Day
    "2026-12-25",  # Christmas Day
}


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS


class LiveStateMutationError(RuntimeError):
    """Raised when a monitor run detects that a protected live-state path
    (data/processed/portfolio_state.json, or anything under config/)
    changed during a score_live.py invocation. Deliberately loud --
    propagates out of main() with a distinct exit code -- rather than
    logged-and-continued: a monitoring run mutating the state it's only
    supposed to observe is worse than any number of missed days."""


def _snapshot_protected() -> dict[str, str | None]:
    """sha256 of every currently-existing protected file, keyed by a
    path relative to REPO_ROOT. ``None`` means the file doesn't exist --
    also worth comparing, since a monitor run creating a protected file
    that didn't exist before is exactly as much a violation as changing
    one that did."""
    paths = [PORTFOLIO_STATE_PATH]
    if CONFIG_DIR.exists():
        paths += sorted(p for p in CONFIG_DIR.rglob("*") if p.is_file())
    snapshot: dict[str, str | None] = {}
    for p in paths:
        key = str(p)
        snapshot[key] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return snapshot


def _run_score_live(tickers: list[str] | None = None) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "scripts.score_live"]
    if tickers:
        cmd += ["--tickers", *tickers]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    combined = proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, combined


def _prior_log(before: str) -> Path | None:
    """The most recent existing log strictly before ``before`` (an
    ISO date string), by filename -- filenames sort lexicographically in
    date order since they're YYYY-MM-DD.log."""
    candidates = sorted(p for p in LOG_DIR.glob("*.log") if p.stem < before)
    return candidates[-1] if candidates else None


def _find_gaps(through: str) -> list[str]:
    """Every expected trading day strictly from the earliest existing log
    through ``through`` (inclusive) that has NO log file at all -- as
    opposed to a present SKIPPED entry, which means the monitor DID run
    and correctly recognized a non-trading day."""
    existing = sorted(p.stem for p in LOG_DIR.glob("*.log"))
    if not existing:
        return []
    start, end = date.fromisoformat(existing[0]), date.fromisoformat(through)
    have = set(existing)
    gaps, d = [], start
    while d <= end:
        if _is_trading_day(d) and d.isoformat() not in have:
            gaps.append(d.isoformat())
        d += timedelta(days=1)
    return gaps


_SEPARATOR = "=" * 70


def _diffable_body(text: str) -> list[str]:
    """The part of a log file worth diffing -- everything AFTER the
    separator line, i.e. exit_code + score_live.py's actual output, never
    the timestamp/heartbeat header above the separator. The header
    changes every single run by construction (it's a wall-clock
    timestamp), so including it would make every day read as 'changed'
    even when score_live.py's real output is byte-identical -- exactly
    the noise this script exists to avoid (a deliberate failing-case test
    in tests/test_monitor_score_live.py catches this directly)."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.rstrip("\n") == _SEPARATOR:
            return lines[i + 1:]
    return lines  # separator missing (unexpected format) -- diff the whole thing rather than crash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="run even on a non-trading day")
    parser.add_argument("--tickers", nargs="+", default=None,
                         help="override config/stock_list.py, passed through to score_live.py")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{today}.log"

    if not _is_trading_day(now.date()) and not args.force:
        reason = "weekend" if now.weekday() >= 5 else "NYSE holiday"
        print(f"{today}: {reason}, skipping. Use --force to override.")
        log_path.write_text(
            f"[{now.isoformat()}] HEARTBEAT exit_code=skipped reason={reason}\n"
            f"{_SEPARATOR}\nSKIPPED ({reason})\n"
        )
        _report_gaps(today)
        return 0

    print(f"{today}: running `python -m scripts.score_live`"
          f"{' --tickers ' + ' '.join(args.tickers) if args.tickers else ''} ...")

    before = _snapshot_protected()
    exit_code, output = _run_score_live(args.tickers)
    after = _snapshot_protected()

    if before != after:
        changed = {k: (before.get(k), after.get(k)) for k in set(before) | set(after) if before.get(k) != after.get(k)}
        log_path.write_text(
            f"[{now.isoformat()}] HEARTBEAT exit_code={exit_code}\n"
            f"{_SEPARATOR}\nLIVE STATE MUTATION DETECTED -- changed paths: {changed}\n{output}\n"
        )
        print(f"  CRITICAL: protected live state changed during this run: {list(changed)}")
        raise LiveStateMutationError(f"protected paths changed: {changed}")

    log_path.write_text(
        f"[{now.isoformat()}] HEARTBEAT exit_code={exit_code}\n"
        f"{_SEPARATOR}\nexit_code={exit_code}\n{output}\n"
    )
    print(f"  exit_code={exit_code}  logged to {log_path}")

    prior = _prior_log(today)
    if prior is None:
        print("  no prior log to diff against (first run).")
    else:
        prior_lines = _diffable_body(prior.read_text())
        today_lines = _diffable_body(log_path.read_text())
        diff = list(difflib.unified_diff(prior_lines, today_lines, fromfile=str(prior.name), tofile=str(log_path.name)))
        if not diff:
            print(f"  no change vs {prior.name}.")
        else:
            print(f"  CHANGED vs {prior.name} ({len(diff)} diff lines) -- review before treating as routine:")
            print("".join(diff))

    _report_gaps(today)
    return 0


def _report_gaps(today: str) -> None:
    gaps = _find_gaps(today)
    if gaps:
        print(f"  GAP CHECK: {len(gaps)} expected trading day(s) with no log at all -- monitor did not run: {gaps}")
    else:
        print("  gap check: none.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LiveStateMutationError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)
