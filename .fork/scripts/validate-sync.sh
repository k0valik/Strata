#!/usr/bin/env bash
# .fork/scripts/validate-sync.sh -- static checks for the fork's synchronization
# setup.  No network, no builds, nothing from upstream is executed.
#
#   validate-sync.sh [--repo DIR] [--base-ref REF]
#
# Checks the metadata and path policy, the repository's fork-control invariants
# (vendored submodule, fork-owned files), and the workflow's supply-chain shape.
# Used as a pre-flight by .github/workflows/sync-upstream.yml and by the fixture
# tests in .fork/tests/run.sh.
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=.fork/scripts/lib.sh
. "$SELF_DIR/lib.sh"

REPO_DIR=$(cd "$SELF_DIR/../.." && pwd)
BASE_REF=HEAD

while [ $# -gt 0 ]; do
  case $1 in
    --repo)     REPO_DIR=$2; shift 2 ;;
    --base-ref) BASE_REF=$2; shift 2 ;;
    -h|--help)  printf 'usage: validate-sync.sh [--repo DIR] [--base-ref REF]\n'; exit 0 ;;
    *)          printf 'validate-sync: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

CHECKS=0
PROBLEMS=0
pass()  { CHECKS=$((CHECKS + 1)); printf '  ok   %s\n' "$1"; }
fail()  { CHECKS=$((CHECKS + 1)); PROBLEMS=$((PROBLEMS + 1)); printf '  FAIL %s\n' "$1"; }
warn()  { printf '  warn %s\n' "$1"; }
check() { # check <what> <expected> <actual>
  if [ "$2" = "$3" ]; then pass "$1"; else fail "$1 (expected [$3], got [$2])"; fi
}
expect_file() { # expect_file <path>   (from the base ref's tree)
  if git cat-file -e "$BASE_REF:$1" 2>/dev/null; then pass "tracked: $1"; else fail "missing from $BASE_REF: $1"; fi
}

fs_require_git
cd "$REPO_DIR"
git rev-parse --git-dir >/dev/null 2>&1 || fs_die "$REPO_DIR is not a git repository"
git rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null || fs_die "no such ref: $BASE_REF"

work=$(fs_mktempdir validate)
trap fs_cleanup EXIT

printf 'validate-sync: %s at %s\n' "$REPO_DIR" "$BASE_REF"

# --------------------------------------------------------------- files exist --

printf '\n== fork files ==\n'
for f in .fork/upstream.lock .fork/path-policy .fork/README.md \
         .fork/scripts/sync-upstream.sh .fork/scripts/validate-sync.sh \
         .fork/scripts/fork-replay.sh .fork/scripts/replay-candidates.sh \
         .fork/tests/replay-fixture.sh .fork/fork-candidates \
         .github/workflows/sync-upstream.yml FORK_NOTES.md .gitmodules; do
  expect_file "$f"
done

for f in .fork/scripts/sync-upstream.sh .fork/scripts/validate-sync.sh \
         .fork/scripts/fork-replay.sh .fork/scripts/replay-candidates.sh \
         .fork/tests/replay-fixture.sh; do
  if [ -x "$f" ]; then pass "executable: $f"; else fail "not executable: $f"; fi
done

# ----------------------------------------------------------------- metadata --

printf '\n== .fork/fork-candidates ==\n'
git show "$BASE_REF:.fork/fork-candidates" > "$work/candidates"
cand_bad=0
cand_n=0
while read -r slug url branch; do
  case $slug in ''|\#*) continue ;; esac
  cand_n=$((cand_n + 1))
  [ -n "$branch" ] || { fail "candidate $slug: missing branch"; cand_bad=1; }
  case $url in https://*) ;; *) fail "candidate $slug: URL is not https: $url"; cand_bad=1 ;; esac
done < "$work/candidates"
if [ "$cand_bad" -eq 0 ]; then
  pass "candidates parse ($cand_n entries, https URLs, branches set)"
fi

printf '\n== .fork/upstream.lock ==\n'
git show "$BASE_REF:.fork/upstream.lock" > "$work/upstream.lock"
if fs_meta_parse "$work/upstream.lock"; then
  pass "parses"
else
  fail "does not parse"
fi
case ${FS_META_URL:-} in
  https://*) pass "upstream URL is https ($FS_META_URL)" ;;
  '')        fail "no upstream repository" ;;
  *)         fail "upstream URL is not https: $FS_META_URL" ;;
esac
[ -n "${FS_META_BRANCH:-}" ] && pass "upstream branch: $FS_META_BRANCH" || fail "no upstream branch"
if [[ ${FS_META_COMMIT:-} =~ ^[0-9a-f]{40}$ ]]; then
  pass "baseline is a full commit id"
else
  fail "baseline is not a full commit id: ${FS_META_COMMIT:-<missing>}"
fi
if git cat-file -e "${FS_META_COMMIT}^{commit}" 2>/dev/null; then
  pass "baseline is present in this repository"
  if git merge-base --is-ancestor "$FS_META_COMMIT" HEAD 2>/dev/null \
     || git diff --quiet "$FS_META_COMMIT" HEAD -- .fork/upstream.lock 2>/dev/null; then
    pass "baseline is reachable (or is the recorded one)"
  else
    warn "baseline is not an ancestor of HEAD (expected after a squash-merged synchronization)"
  fi
else
  warn "baseline is not in this repository yet (a sync fetches it)"
fi

# -------------------------------------------------------------- path policy --

printf '\n== .fork/path-policy ==\n'
git show "$BASE_REF:.fork/path-policy" > "$work/path-policy"
if fs_policy_load "$work/path-policy"; then
  pass "parses ($(grep -cvE '^[[:space:]]*(#|$)' "$work/path-policy") rules)"
else
  fail "does not parse"
fi

pruned_count=0
for m in "${FS_POLICY_MODE[@]}"; do
  [ "$m" = pruned ] && pruned_count=$((pruned_count + 1))
done
check "exactly one 'pruned' rule" 1 "$pruned_count"

# Behaviour on representative paths.  The baseline tree is used for both sides,
# so the `pruned` rule cannot fire here: these assertions cover the rules only.
fs_path_sets_init "$BASE_REF^{tree}" "$BASE_REF^{tree}"
expect_mode() {
  local got
  got=$(fs_classify "$1")
  check "policy: $1 -> $2" "$2" "$got"
}
expect_mode .fork/upstream.lock        protected
expect_mode .gitmodules                protected
expect_mode third_party/llama.cpp      protected
expect_mode README.md                  ours
expect_mode AGENTS.md                  ours
expect_mode FORK_NOTES.md              ours
expect_mode .github/workflows/x.yml    ours
expect_mode CMakeLists.txt             mirror
expect_mode src/engine.cpp             mirror
expect_mode include/strata/x.hpp       mirror
expect_mode cmake/hip_backend.cmake    mirror
expect_mode tests/core/x_test.cpp      mirror
expect_mode ref/load.py                mirror
expect_mode tools/iq_pack.py           mirror
expect_mode bench/micro/x.cu           mirror
expect_mode data/draft_vocab.bin       mirror
expect_mode docs/whatever.md           dropped
expect_mode serve/server.py            dropped
expect_mode sycl/x.cpp                 dropped
expect_mode metal/x.mm                 dropped
expect_mode bench/results/r.json       dropped
expect_mode install_now.py             unknown

# --------------------------------------------------------------- fork shape --

printf '\n== vendored submodule ==\n'
if git show "$BASE_REF:.gitmodules" | grep -q 'path = third_party/llama.cpp'; then
  pass ".gitmodules declares third_party/llama.cpp"
else
  fail ".gitmodules does not declare third_party/llama.cpp"
fi
if git show "$BASE_REF:.gitmodules" | grep -qE 'url = https://'; then
  pass "submodule URL is https"
else
  fail "submodule URL is not https"
fi
gitlink=$(git ls-tree "$BASE_REF" third_party/llama.cpp | awk '{print $2, $3}')
check "third_party/llama.cpp is a gitlink" "commit" "$(git ls-tree "$BASE_REF" third_party/llama.cpp | awk '{print $2}')"
[ -n "$gitlink" ] && pass "submodule pin: $(git ls-tree "$BASE_REF" third_party/llama.cpp | awk '{print $3}')"

# ----------------------------------------------------------------- workflow --

printf '\n== .github/workflows/sync-upstream.yml ==\n'
wf=$work/workflow.yml
git show "$BASE_REF:.github/workflows/sync-upstream.yml" > "$wf"

if python3 -c 'import yaml, sys; yaml.safe_load(open(sys.argv[1]))' "$wf" 2>/dev/null; then
  pass "YAML parses"
elif command -v python3 >/dev/null 2>&1; then
  warn "PyYAML is not available; the YAML was not parsed"
else
  warn "python3 is not available; the YAML was not parsed"
fi

if grep -qE '^permissions:' "$wf"; then pass "declares top-level permissions"; else fail "no top-level permissions block"; fi
if grep -qE '^\s*contents: write' "$wf"; then pass "contents: write"; else fail "missing contents: write"; fi
if grep -qE '^\s*pull-requests: write' "$wf"; then pass "pull-requests: write"; else fail "missing pull-requests: write"; fi
if grep -qE '^\s*(write-all|actions: write|id-token: write)' "$wf"; then
  fail "grants permissions it should not need"
else
  pass "no unnecessary permissions"
fi

if grep -qE '^\s*uses:' "$wf"; then
  unpinned=$(grep -E '^\s*uses:' "$wf" | grep -vE '@[0-9a-f]{40}(\s|$)' || true)
  if [ -n "$unpinned" ]; then
    fail "actions not pinned to a full commit SHA: $unpinned"
  else
    pass "every action is pinned to an immutable SHA"
  fi
else
  pass "no actions are used"
fi

if grep -qE 'workflow_dispatch:' "$wf"; then pass "workflow_dispatch"; else fail "no workflow_dispatch trigger"; fi
if grep -qE '^[[:space:]]*schedule:' "$wf"; then pass "schedule trigger"; else fail "no schedule trigger"; fi
if grep -qE "cron: *'0 " "$wf" || grep -qE 'cron: *"0 ' "$wf"; then
  fail "the schedule fires at the top of the hour"
else
  pass "the schedule avoids the top of the hour"
fi
if grep -qE '^concurrency:' "$wf"; then pass "concurrency guard"; else fail "no concurrency guard"; fi
if grep -qE 'curl|wget' "$wf"; then
  fail "the workflow downloads things with curl/wget"
else
  pass "no curl/wget"
fi
if grep -qE 'secrets\.' "$wf"; then
  fail "the workflow references secrets instead of github.token"
else
  pass "no repository secrets are used"
fi
if grep -q 'sync-upstream.sh' "$wf"; then pass "runs the fork's own script"; else fail "does not run .fork/scripts/sync-upstream.sh"; fi
if grep -qE 'uses: actions/checkout' "$wf" && ! grep -qE 'submodules: true|submodules: recursive' "$wf"; then
  pass "does not check out submodules"
else
  fail "the workflow may check out submodules"
fi

# ----------------------------------------------------------------- secrets --

printf '\n== credentials ==\n'
if grep -rInE '(BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-)' .fork/ 2>/dev/null | grep -v '^Binary'; then
  fail "credential-looking material under .fork/"
else
  pass "no credential-looking material under .fork/"
fi

printf '\n%d checks, %d problem(s)\n' "$CHECKS" "$PROBLEMS"
[ "$PROBLEMS" -eq 0 ]
