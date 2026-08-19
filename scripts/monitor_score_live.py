"""Daily score_live.py monitor (B4): run the real live-scoring entry point
once a day, log its full output, and diff against the prior day's log so
a change that isn't obviously explained gets surfaced rather than missed
in a scroll of daily output.

Deliberately thin: this does NOT reimplement any of score_live.py's logic
-- it shells out to the exact same `python -m scripts.score_live` a human
would run, captures stdout+stderr+exit code verbatim, and treats that
transcript as the unit to log and diff. Whatever score_live.py does or
doesn't do (including failing its own _preflight check) is exactly what
gets recorded -- this script adds logging and diffing, nothing else.

Known, expected near-term behavior (not a bug in this script): with
config/stock_list.py still empty and config/hold_period.py::
WINDOW_START_DATE still unset (baseline_system_architecture.md §5,
"Blocked on External Input"), every run until those are populated will
fail the same _preflight check the same way -- the diff will correctly
report "no change" every day until that's true, and the day it stops
failing IS the notable change this script exists to catch.

Trading-day approximation: skips Saturday/Sunday only, the same
weekends-only, no-NYSE-holidays approximation already used and named by
config/hold_period.py::period_number_for -- a small, bounded, honestly-
stated imprecision, not a real trading calendar. A holiday run just
produces a log identical to a normal non-trading day's failure/output,
which the diff step will correctly call "no change".

Usage:
    python -m scripts.monitor_score_live
    python -m scripts.monitor_score_live --force   # run even on a weekend (testing)
"""
from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "data" / "processed" / "live_monitor_logs"


def _run_score_live() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.score_live"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    combined = proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, combined


def _prior_log(before: str) -> Path | None:
    """The most recent existing log strictly before ``before`` (an
    ISO date string), by filename -- filenames sort lexicographically in
    date order since they're YYYY-MM-DD.log."""
    candidates = sorted(p for p in LOG_DIR.glob("*.log") if p.stem < before)
    return candidates[-1] if candidates else None


_SEPARATOR = "=" * 70


def _diffable_body(text: str) -> list[str]:
    """The part of a log file worth diffing -- everything AFTER the
    separator line, i.e. exit_code + score_live.py's actual output, never
    the timestamp header above the separator. The header changes every
    single run by construction (it's a wall-clock timestamp), so
    including it would make every day read as 'changed' even when
    score_live.py's real output is byte-identical -- exactly the noise
    this script exists to avoid (a deliberate failing-case test in
    tests/test_monitor_score_live.py catches this directly)."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.rstrip("\n") == _SEPARATOR:
            return lines[i + 1:]
    return lines  # separator missing (unexpected format) -- diff the whole thing rather than crash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="run even on a weekend")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{today}.log"

    if now.weekday() >= 5 and not args.force:  # 5=Saturday, 6=Sunday
        print(f"{today}: weekend, skipping (weekends-only approximation, see docstring). Use --force to override.")
        log_path.write_text(f"[{now.isoformat()}] SKIPPED (weekend approximation)\n")
        return 0

    print(f"{today}: running `python -m scripts.score_live` ...")
    exit_code, output = _run_score_live()
    log_path.write_text(
        f"[{now.isoformat()}] exit_code={exit_code}\n"
        f"{_SEPARATOR}\n"
        f"exit_code={exit_code}\n{output}\n"
    )
    print(f"  exit_code={exit_code}  logged to {log_path}")

    prior = _prior_log(today)
    if prior is None:
        print("  no prior log to diff against (first run).")
        return 0

    prior_lines = _diffable_body(prior.read_text())
    today_lines = _diffable_body(log_path.read_text())
    diff = list(difflib.unified_diff(prior_lines, today_lines, fromfile=str(prior.name), tofile=str(log_path.name)))

    if not diff:
        print(f"  no change vs {prior.name}.")
    else:
        print(f"  CHANGED vs {prior.name} ({len(diff)} diff lines) -- review before treating as routine:")
        print("".join(diff))

    return 0


if __name__ == "__main__":
    sys.exit(main())
