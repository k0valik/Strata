#!/usr/bin/env bash
# .fork/scripts/replay-candidates.sh -- run fork-replay.sh over a candidates
# file: one replay branch and one report per candidate, failures do not stop
# the rest.
#
#   replay-candidates.sh <candidates-file> <reports-dir> [fork-replay arguments...]
#
# Extra arguments are passed through to fork-replay.sh (for example --push,
# --pr, or --repo DIR for a different checkout).  With no --report argument of
# its own, fork-replay.sh writes each candidate's report into the reports dir.
set -uo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAND=${1:?usage: replay-candidates.sh <candidates-file> <reports-dir> [fork-replay arguments...]}
REPORTS=${2:?usage: replay-candidates.sh <candidates-file> <reports-dir> [fork-replay arguments...]}
shift 2 || true
mkdir -p "$REPORTS"

rc_all=0
while read -r slug url branch; do
  case $slug in ''|\#*) continue ;; esac
  echo "==================================================================="
  echo "== $slug ($url @ $branch)"
  echo "==================================================================="
  if bash "$SELF_DIR/fork-replay.sh" --fork "$url" --branch "$branch" \
       --branch-out "replay/$slug" --report "$REPORTS/$slug.md" "$@"; then
    echo "== $slug: OK"
  else
    echo "== $slug: FAILED (see $REPORTS/$slug.md)"
    rc_all=1
  fi
done < "$CAND"
exit "$rc_all"
