#!/usr/bin/env bash
# Confirms the installed crontab's B4 line (if any) matches
# ops/monitor_score_live.crontab exactly -- so the two never silently
# drift apart (the repo copy edited but the machine never re-installed,
# or vice versa). Prints a diff and exits non-zero if they differ or if
# nothing is installed yet.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_LINE=$(grep -E '^[0-9]' ops/monitor_score_live.crontab || true)
INSTALLED_LINE=$(crontab -l 2>/dev/null | grep monitor_score_live || true)

if [[ -z "$INSTALLED_LINE" ]]; then
    echo "NOT INSTALLED: no monitor_score_live line in this user's crontab yet."
    echo "  proposed line (ops/monitor_score_live.crontab):"
    echo "    $REPO_LINE"
    exit 1
fi

if [[ "$REPO_LINE" == "$INSTALLED_LINE" ]]; then
    echo "OK: installed crontab line matches ops/monitor_score_live.crontab exactly."
    exit 0
else
    echo "MISMATCH between the installed crontab and ops/monitor_score_live.crontab:"
    diff <(echo "$INSTALLED_LINE") <(echo "$REPO_LINE") || true
    exit 1
fi
