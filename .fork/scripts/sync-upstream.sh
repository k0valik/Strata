#!/usr/bin/env bash
# .fork/scripts/sync-upstream.sh -- prepare (and optionally publish) the rolling
# upstream-synchronization candidate for this fork.
#
#   sync-upstream.sh                       # dry run against the checked-out branch
#   sync-upstream.sh --push                # also update the rolling branch
#   sync-upstream.sh --pr                  # ... and create/update the rolling PR
#   sync-upstream.sh --classify PATH...    # print the path policy's verdict
#
# What it does, in one line: apply `<lock>.commit..<upstream head>` to the
# checked-out branch with the locked commit as the merge base, enforce
# .fork/path-policy, and hand the result to a rolling branch + pull request for
# review.
#
# It is a Git/filesystem synchronization, not CI: it never builds, tests or runs
# anything that came from upstream (see .fork/README.md).
#
# Nothing here touches the working tree or the real index: the merge happens in
# memory (git merge-tree) and the candidate tree is assembled in a throwaway
# index file.
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=.fork/scripts/lib.sh
. "$SELF_DIR/lib.sh"

REPO_DIR=$(cd "$SELF_DIR/../.." && pwd)
BASE_REF=HEAD
SYNC_BRANCH=sync/upstream
REMOTE=origin
PR_BASE=''
PR_TITLE='sync: upstream Strata'
UPSTREAM_URL_OVERRIDE=''
UPSTREAM_BRANCH_OVERRIDE=''
ALLOW_LOCAL_UPSTREAM=0
DO_PUSH=0
DO_PR=0
FS_REPORT_PATH=''
CLASSIFY_ARGS=()

# Set as the run progresses; the failure report uses whatever is known.  These must
# all exist from the start: the EXIT trap reports even failures that happen before
# the metadata was read.
BASE=''
UPHEAD=''
remote_head=''
UPSTREAM_URL=''
UPSTREAM_BRANCH=''
PUSHED=0
PUSHED_SHA=''

usage() {
  cat <<'EOF'
usage: sync-upstream.sh [options] [--classify PATH...]

  --repo DIR              repository to synchronize (default: the fork itself)
  --base-ref REF          branch/commit to build the candidate on (default: HEAD)
  --sync-branch NAME      rolling branch name (default: sync/upstream)
  --remote NAME|URL       remote to push to (default: origin)
  --upstream-url URL      override .fork/upstream.lock's repository
  --upstream-branch NAME  override .fork/upstream.lock's branch
  --allow-local-upstream  permit a non-https upstream URL (tests only)
  --dry-run               analyse and report only; this is the default
  --push                  update the rolling branch on --remote
  --pr                    create or refresh the rolling PR (implies --push)
  --pr-base NAME          base branch for the pull request (default: gh's default)
  --report FILE           write the markdown report to FILE (and it stays on disk)
  --classify PATH...      print 'PATH MODE' for each path and exit (no network)
  -h, --help              this text

The report file doubles as the pull-request body; with --pr and no --report a
temporary file is used.
EOF
}

while [ $# -gt 0 ]; do
  case $1 in
    --repo)                 REPO_DIR=$2; shift 2 ;;
    --base-ref)             BASE_REF=$2; shift 2 ;;
    --sync-branch)          SYNC_BRANCH=$2; shift 2 ;;
    --remote)               REMOTE=$2; shift 2 ;;
    --upstream-url)         UPSTREAM_URL_OVERRIDE=$2; shift 2 ;;
    --upstream-branch)      UPSTREAM_BRANCH_OVERRIDE=$2; shift 2 ;;
    --allow-local-upstream) ALLOW_LOCAL_UPSTREAM=1; shift ;;
    --dry-run)              DO_PUSH=0; DO_PR=0; shift ;;
    --push)                 DO_PUSH=1; shift ;;
    --pr)                   DO_PUSH=1; DO_PR=1; shift ;;
    --pr-base)              PR_BASE=$2; shift 2 ;;
    --report)               FS_REPORT_PATH=$2; shift 2 ;;
    --classify)
      shift
      [ $# -gt 0 ] || { usage >&2; exit 2; }
      CLASSIFY_ARGS=("$@")
      break ;;
    -h|--help)              usage; exit 0 ;;
    *) printf 'sync-upstream: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- reporting --

report_reset() {
  [ -n "$FS_REPORT_PATH" ] || return 0
  : > "$FS_REPORT_PATH"
}

report_start() {                        # report_start <status>
  fs_report "## Upstream sync — $1"
  fs_report ''
  fs_report '| | |'
  fs_report '|---|---|'
  fs_report "| upstream | \`${UPSTREAM_URL:-unknown}\` (\`${UPSTREAM_BRANCH:-unknown}\`) |"
}

report_footer() {
  fs_report ''
  fs_report '---'
  fs_report ''
  fs_report 'No upstream code was executed while preparing this synchronization: it reads git'
  fs_report 'objects and applies `.fork/path-policy`.'
  fs_report ''
  fs_report 'Building and running the engine against a candidate is a manual step (see'
  fs_report 'FORK_NOTES.md); this workflow runs no engine CI.'
}

# The PR must never keep showing a healthy candidate after a run that was blocked,
# so the failure path rewrites the report from scratch and pushes it as the body.
report_blocked() {
  local reason tip represents row number
  reason=$(cat "$FS_ERROR_FILE" 2>/dev/null || true)
  [ -n "$reason" ] || reason=${FS_ERROR:-the run failed; see the workflow log}

  report_reset
  report_start "🔴 BLOCKED"
  [ -n "$BASE" ] || BASE=$(printf '%s' "${FS_META_COMMIT:-}")
  [ -n "$BASE" ] && fs_report "| accepted upstream | \`$(fs_short "$BASE")\` |"
  if [ -n "$UPHEAD" ]; then
    fs_report "| latest upstream seen | \`$(fs_short "$UPHEAD")\` |"
  elif [ -n "$remote_head" ]; then
    fs_report "| latest upstream seen | \`$(fs_short "$remote_head")\` |"
  fi
  if [ "$PUSHED" -eq 1 ]; then
    fs_report "| rolling branch | **updated** to \`$(fs_short "$PUSHED_SHA")\` |"
  else
    fs_report '| rolling branch | **not modified** |'
  fi

  # What the branch still stands for, so the report is not vague about it.  Only
  # worth looking up when this run did not itself put a candidate there.
  if [ "$PUSHED" -eq 0 ] && [ "$DO_PR" -eq 1 ] && command -v git >/dev/null 2>&1; then
    tip=$(git ls-remote --heads "$REMOTE" "refs/heads/$SYNC_BRANCH" 2>/dev/null | awk 'NR==1 {print $1}' || true)
    if [ -n "$tip" ]; then
      git fetch --quiet --no-tags --no-recurse-submodules "$REMOTE" \
          "+refs/heads/$SYNC_BRANCH:refs/fork-sync/rolling" 2>/dev/null || true
      represents=$(git show "refs/fork-sync/rolling:.fork/upstream.lock" 2>/dev/null \
                   | sed -n 's/^commit[[:space:]]*=[[:space:]]*//p' | head -1 || true)
      row="| candidate still on the branch | \`$(fs_short "$tip")\`"
      [ -n "$represents" ] && row="$row (represents upstream \`$(fs_short "$represents")\`)"
      fs_report "$row |"
    fi
  fi

  fs_report ''
  fs_report '### Reason'
  fs_report ''
  fs_report "$reason"
  fs_report ''
  if [ "$PUSHED" -eq 1 ]; then
    fs_report "The candidate \`$PUSHED_SHA\` did reach the rolling branch, but this run did not"
    fs_report 'finish (see above), so the pull request may not describe it.  `main` is'
    fs_report 'unchanged, and the branch is self-consistent: it carries the advanced'
    fs_report '`.fork/upstream.lock`.'
  else
    fs_report 'Nothing was pushed and `main` was not changed.  Any candidate already on the'
    fs_report 'rolling branch still stands — it simply does not include this newer upstream'
    fs_report 'range yet, and the ancestry graft was deliberately not attached to it.'
  fi
  report_footer

  if [ "$DO_PR" -eq 1 ] && command -v gh >/dev/null 2>&1; then
    number=$(gh pr list --head "$SYNC_BRANCH" --state open --json number \
             --jq '.[0].number // empty' 2>/dev/null || true)
    if [ -n "$number" ]; then
      gh pr edit "$number" --body-file "$FS_REPORT_PATH" --title "$PR_TITLE" >/dev/null 2>&1 || true
      printf 'fork-sync: pull request #%s now reports the blocked attempt\n' "$number" >&2
    fi
  fi
  return 0
}

report_up_to_date() {                   # report_up_to_date <tip>
  report_reset
  report_start "🟢 UP TO DATE"
  fs_report "| accepted upstream | \`$(fs_short "$BASE")\` |"
  fs_report "| latest upstream | \`$(fs_short "$1")\` |"
  fs_report ''
  fs_report 'Nothing to synchronize: upstream is already at the accepted revision.'
  report_footer
}

# Is a mirrored path part of what actually runs?  tools/, tests/, ref/ and
# bench/micro/ are retained but are not the runtime.
runtime_path() {
  case $1 in
    src/*|include/*|cmake/*|CMakeLists.txt) return 0 ;;
    *) return 1 ;;
  esac
}

# Areas worth a second look even when the synchronization is mechanically clean:
# the build integration the fork edited for vendoring, upstream's own dependency
# integration, and the runtime data files.
sensitive_path() {
  case $1 in
    CMakeLists.txt|cmake/*|third_party/*|data/*) return 0 ;;
    *) return 1 ;;
  esac
}

work=$(fs_mktempdir work)
if [ "$DO_PR" -eq 1 ] && [ -z "$FS_REPORT_PATH" ]; then
  FS_REPORT_PATH=$work/sync-report.md
fi
if [ -n "$FS_REPORT_PATH" ]; then
  : > "$FS_REPORT_PATH" || fs_die "cannot write the report to $FS_REPORT_PATH"
fi

# Set before anything can fail, so the report can always name the reason (a
# failure inside a subshell cannot set FS_ERROR in this shell).
FS_ERROR_FILE=$work/error.txt
: > "$FS_ERROR_FILE"
export FS_ERROR_FILE

fs_on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ -n "$FS_REPORT_PATH" ]; then
    report_blocked || true
  fi
  fs_cleanup
  exit "$rc"
}
trap fs_on_exit EXIT

# ------------------------------------------------------------------ context --

fs_require_git
[ -d "$REPO_DIR" ] || fs_die "no such repository: $REPO_DIR"
cd "$REPO_DIR"
git rev-parse --git-dir >/dev/null 2>&1 || fs_die "$REPO_DIR is not a git repository"

FS_BASE_REF=$BASE_REF
git rev-parse --verify --quiet "$FS_BASE_REF^{commit}" >/dev/null \
  || fs_die "base ref '$FS_BASE_REF' does not resolve to a commit"

# Metadata and policy come from the base ref's tree, not the working tree: the
# branch being synchronized is what defines the rules.
git show "$FS_BASE_REF:.fork/upstream.lock" > "$work/upstream.lock" 2>/dev/null \
  || fs_die "no .fork/upstream.lock in $FS_BASE_REF"
git show "$FS_BASE_REF:.fork/path-policy"  > "$work/path-policy"  2>/dev/null \
  || fs_die "no .fork/path-policy in $FS_BASE_REF"

if ! fs_meta_parse "$work/upstream.lock"; then
  fs_die ".fork/upstream.lock on $FS_BASE_REF is missing or malformed (see the log above)"
fi
if ! fs_policy_load "$work/path-policy"; then
  fs_die ".fork/path-policy on $FS_BASE_REF is missing or malformed (see the log above)"
fi

UPSTREAM_URL=${UPSTREAM_URL_OVERRIDE:-$FS_META_URL}
UPSTREAM_BRANCH=${UPSTREAM_BRANCH_OVERRIDE:-$FS_META_BRANCH}
BASE=$FS_META_COMMIT

case $UPSTREAM_URL in
  https://*) ;;
  *) [ "$ALLOW_LOCAL_UPSTREAM" = 1 ] \
       || fs_die "upstream URL '$UPSTREAM_URL' is not https (--allow-local-upstream is for tests only)" ;;
esac
[[ $BASE =~ ^[0-9a-f]{40}$ ]] || fs_die "recorded baseline '$BASE' is not a full commit id"

# The `pruned` rule needs the baseline tree.  For --classify (no fetch) it must
# already be here: without it, pruned paths would fall through to a broader rule
# and be treated as ordinary upstream files, so say so instead of guessing.
if [ "${#CLASSIFY_ARGS[@]}" -gt 0 ]; then
  git cat-file -e "$BASE^{tree}" 2>/dev/null \
    || fs_die "baseline $(fs_short "$BASE") is not in this repository, so the 'pruned' rule cannot be evaluated"
  fs_path_sets_init "$FS_BASE_REF^{tree}" "$BASE^{tree}"
  for p in "${CLASSIFY_ARGS[@]}"; do
    printf '%s %s\n' "$p" "$(fs_classify "$p")"
  done
  exit 0
fi

echo "fork-sync: repository  $REPO_DIR"
echo "fork-sync: base ref    $FS_BASE_REF -> $(fs_short "$(git rev-parse "$FS_BASE_REF")")"
echo "fork-sync: upstream    $UPSTREAM_URL ($UPSTREAM_BRANCH)"
echo "fork-sync: accepted    $(fs_short "$BASE")"

# ------------------------------------------------------- is there anything? --

# Advertisement only: the common case (upstream unchanged) costs one round trip
# and downloads no objects at all.
set +e
remote_head=$(git ls-remote --heads "$UPSTREAM_URL" "refs/heads/$UPSTREAM_BRANCH" 2>/dev/null | awk 'NR==1 {print $1}')
ls_rc=$?
set -e
[ "$ls_rc" -eq 0 ] || fs_die "cannot reach upstream '$UPSTREAM_URL' (git ls-remote failed)"
[ -n "$remote_head" ] || fs_die "upstream '$UPSTREAM_URL' has no branch '$UPSTREAM_BRANCH'"
[[ $remote_head =~ ^[0-9a-f]{40}$ ]] || fs_die "upstream returned a strange ref: '$remote_head'"

if [ "$remote_head" = "$BASE" ]; then
  echo "fork-sync: up to date (upstream is at the accepted revision)"
  report_up_to_date "$remote_head"
  exit 0
fi

# --------------------------------------------------------------- the range --

UP_REF="refs/remotes/fork-sync/$UPSTREAM_BRANCH"
echo "fork-sync: fetching $UPSTREAM_BRANCH (git objects only; nothing is executed)"
git fetch --quiet --no-tags --no-recurse-submodules "$UPSTREAM_URL" "+refs/heads/$UPSTREAM_BRANCH:$UP_REF"

UPHEAD=$(git rev-parse --verify --quiet "$UP_REF^{commit}") \
  || fs_die "fetched ref $UP_REF does not resolve to a commit"
[ "$UPHEAD" = "$remote_head" ] \
  || fs_warn "upstream moved while synchronizing ($(fs_short "$remote_head") -> $(fs_short "$UPHEAD"))"

git cat-file -e "$BASE^{commit}" 2>/dev/null \
  || fs_die "the accepted baseline $(fs_short "$BASE") is not in the fetched upstream history; upstream probably rewrote history and a human must decide"

if [ "$BASE" = "$UPHEAD" ]; then
  echo "fork-sync: up to date"
  report_up_to_date "$UPHEAD"
  exit 0
fi

if ! git merge-base --is-ancestor "$BASE" "$UPHEAD"; then
  fs_die "accepted baseline $(fs_short "$BASE") is not an ancestor of upstream $(fs_short "$UPHEAD"); upstream history was rewritten and a human must decide (review it, then update .fork/upstream.lock by hand)"
fi

# Only now, with the baseline confirmed present and reachable, can the `pruned`
# rule be evaluated.
fs_path_sets_init "$FS_BASE_REF^{tree}" "$BASE^{tree}"
# ... and the same comparison tells which paths the fork changed on its own.
fs_divergence_init "$BASE" "$FS_BASE_REF"

mapfile -t RANGE < <(git rev-list --reverse "$BASE..$UPHEAD")
RANGE_COUNT=${#RANGE[@]}
echo "fork-sync: range        $RANGE_COUNT commit(s) $(fs_short "$BASE")..$(fs_short "$UPHEAD")"

git diff --no-renames --name-status "$BASE" "$UPHEAD" > "$work/changed.txt"
changed_paths=$(wc -l < "$work/changed.txt" | tr -d ' ')

MODE_LIMIT=200
RUNTIME_COUNT=0
SENSITIVE_COUNT=0
SENSITIVE_LIST=''
DIVERGED_COUNT=0
DIVERGED_LIST=''
DIVERGED_LIMIT=20
declare -A MODE_COUNT=() MODE_LIST=() AREA_MIRROR=() AREA_SUPPRESS=()

while IFS=$'\t' read -r status path; do
  [ -n "$path" ] || continue
  mode=$(fs_classify "$path")
  MODE_COUNT[$mode]=$(( ${MODE_COUNT[$mode]:-0} + 1 ))
  if [ "${MODE_COUNT[$mode]}" -le "$MODE_LIMIT" ]; then
    MODE_LIST[$mode]="${MODE_LIST[$mode]:-}  ${path} (${status})"$'\n'
  fi
  area=${path%%/*}
  [ "$area" = "$path" ] && area='(repository root)'
  if [ "$mode" = mirror ]; then
    AREA_MIRROR[$area]=$(( ${AREA_MIRROR[$area]:-0} + 1 ))
    if runtime_path "$path"; then
      RUNTIME_COUNT=$((RUNTIME_COUNT + 1))
    fi
    # A file upstream maintains that the fork has patched: the merge result keeps
    # the patch and applies upstream's edit, but "clean" says nothing about
    # meaning -- upstream may have fixed the same thing differently.  Only files
    # that exist in the accepted revision count; a fork-only path is not a patch
    # to something upstream maintains.
    if fs_path_in_baseline "$path" && fs_path_diverged "$path"; then
      DIVERGED_COUNT=$((DIVERGED_COUNT + 1))
      if [ "$DIVERGED_COUNT" -le "$DIVERGED_LIMIT" ]; then
        DIVERGED_LIST="${DIVERGED_LIST}  - \`${path}\` (upstream ${status}; this fork also changed it)"$'\n'
      fi
    fi
  else
    AREA_SUPPRESS[$area]=$(( ${AREA_SUPPRESS[$area]:-0} + 1 ))
  fi
  if sensitive_path "$path"; then
    SENSITIVE_COUNT=$((SENSITIVE_COUNT + 1))
    if [ "$SENSITIVE_COUNT" -le 20 ]; then
      SENSITIVE_LIST="${SENSITIVE_LIST}  - \`${path}\` (${status}, ${mode})"$'\n'
    fi
  fi
done < "$work/changed.txt"
echo "fork-sync: paths        $changed_paths (mirror ${MODE_COUNT[mirror]:-0}, ours ${MODE_COUNT[ours]:-0}, dropped ${MODE_COUNT[dropped]:-0}, protected ${MODE_COUNT[protected]:-0}, unknown ${MODE_COUNT[unknown]:-0})"
if [ "$DIVERGED_COUNT" -gt 0 ]; then
  echo "fork-sync: divergences  $DIVERGED_COUNT mirrored path(s) that carry a local fork patch were changed upstream too"
fi

# ---------------------------------------------------------------- the merge --

fs_merge_tree "$BASE" "$FS_BASE_REF" "$UPHEAD"
conflict_count=$(printf '%s\n' "$FS_MERGE_CONFLICTS" | grep -c . || true)
echo "fork-sync: merge        $([ "$FS_MERGE_RC" -eq 0 ] && echo clean || echo "conflicted, $conflict_count path(s)")"

# Applies the policy; any protected/unknown/conflicting-mirror path stops the run
# (before any commit object exists, so no ancestry is claimed for unreviewed work).
CANDIDATE_TREE=$(fs_build_candidate_tree "$FS_BASE_REF^{tree}" "$FS_MERGE_TREE" "$FS_MERGE_CONFLICTS")

MAIN_TREE=$(git rev-parse "$FS_BASE_REF^{tree}")
source_changed=1
[ "$CANDIDATE_TREE" = "$MAIN_TREE" ] && source_changed=0

# The candidate always advances the lock, so merging the PR advances the baseline.
lock_dir=$(fs_mktempdir lock-index)
export GIT_INDEX_FILE=$lock_dir/index
git read-tree "$CANDIDATE_TREE"
fs_write_lock "$FS_BASE_REF" "$UPHEAD"
CANDIDATE_TREE=$(git write-tree)
unset GIT_INDEX_FILE

# The second parent of the candidate is an assertion that the range was accounted
# for.  Make sure the content that assertion points at agrees, before any commit
# object exists.
lock_written=$(git show "$CANDIDATE_TREE:.fork/upstream.lock" | sed -n 's/^commit[[:space:]]*=[[:space:]]*//p' | head -1)
[ "$lock_written" = "$UPHEAD" ] \
  || fs_die "internal: the candidate's .fork/upstream.lock records '$lock_written', not $UPHEAD"

BASE_SHA=$(git rev-parse "$FS_BASE_REF")
fs_identity_setup
msg=$work/commit-msg
{
  printf 'sync: upstream Strata\n\n'
  printf 'Upstream range: %s..%s (%d commit%s)\n' \
    "$(fs_short "$BASE")" "$(fs_short "$UPHEAD")" "$RANGE_COUNT" "$([ "$RANGE_COUNT" -eq 1 ] && echo '' || echo 's')"
  printf 'Content: three-way merge with the accepted upstream revision %s as the merge\n' "$(fs_short "$BASE")"
  printf 'base (.fork/upstream.lock), with .fork/path-policy enforced -- dropped paths stay\n'
  printf 'dropped, fork-owned files stay ours.\n'
  printf '\nThe second parent asserts that every change in that range was accounted for: the\n'
  printf 'tree above is the policy-filtered result and the lock file inside this commit\n'
  printf 'records the same upstream revision.  Merge this pull request with a merge commit to\n'
  printf 'keep the fork from reading as "N commits behind" upstream; squashing also works --\n'
  printf 'the lock file carries the baseline -- but the fork then shows as behind until the\n'
  printf 'next synchronization is merged into it.\n'
  if [ "$source_changed" -eq 0 ]; then
    printf '\nNo source changes: fork policy suppressed every upstream change in this range.\n'
  fi
  if [ "$DIVERGED_COUNT" -gt 0 ]; then
    printf '\nFork divergences: %d mirrored path(s) that already carry a local fork patch were\n' "$DIVERGED_COUNT"
    printf 'changed upstream as well and merged textually; see the pull-request body.\n'
  fi
  printf '\nNo upstream code was executed to produce this commit.\n'
  printf '\nFork-Sync-Bot: sync-upstream.sh\n'
  printf 'Fork-Sync-Automated: true\n'
  printf 'Fork-Sync-Base: %s\n' "$BASE_SHA"
  printf 'Fork-Sync-Upstream: %s\n' "$UPHEAD"
} > "$msg"

CANDIDATE=$(fs_commit_tree "$CANDIDATE_TREE" "$BASE_SHA" "$UPHEAD" "$msg")
echo "fork-sync: candidate    $(fs_short "$CANDIDATE")$([ "$source_changed" -eq 0 ] && echo ' (lock update only)')"

# ---------------------------------------------------------------- publish ---
# Done before the report is written, so the report (and therefore the pull-request
# body) can state what actually happened.
if [ "$DO_PUSH" -eq 1 ]; then
  existing=$(git ls-remote --heads "$REMOTE" "refs/heads/$SYNC_BRANCH" 2>/dev/null | awk 'NR==1 {print $1}')
  if [ -n "$existing" ]; then
    git fetch --quiet --no-tags --no-recurse-submodules "$REMOTE" "+refs/heads/$SYNC_BRANCH:refs/fork-sync/rolling" \
      || fs_die "cannot read the rolling branch $SYNC_BRANCH from $REMOTE"
    fs_automation_owned "$existing" "$BASE_SHA" \
      || fs_die "refusing to overwrite $SYNC_BRANCH: its tip ($(fs_short "$existing")) was not written by this tooling, so it was left alone; inspect it with 'git log --oneline -3 $existing'"
    lease="--force-with-lease=refs/heads/$SYNC_BRANCH:$existing"
  else
    lease="--force-with-lease=refs/heads/$SYNC_BRANCH:"
  fi
  git push --quiet $lease "$REMOTE" "$CANDIDATE:refs/heads/$SYNC_BRANCH" \
    || fs_die "push of $SYNC_BRANCH to $REMOTE failed (it may have moved; re-run)"
  PUSHED=1
  PUSHED_SHA=$CANDIDATE
  echo "fork-sync: pushed       $SYNC_BRANCH -> $(fs_short "$CANDIDATE")"
fi

# --------------------------------------------------------------- the report --

report_reset
if [ "$SENSITIVE_COUNT" -gt 0 ] || [ "$DIVERGED_COUNT" -gt 0 ]; then
  report_start "🟡 REVIEW"
else
  report_start "🟢 READY"
fi
fs_report "| accepted upstream | \`$(fs_short "$BASE")\` |"
fs_report "| candidate upstream | \`$(fs_short "$UPHEAD")\` |"
fs_report "| upstream commits | $RANGE_COUNT (\`$(fs_short "$BASE")..$(fs_short "$UPHEAD")\`) |"
fs_report "| changed paths | $changed_paths (mirror ${MODE_COUNT[mirror]:-0}, ours ${MODE_COUNT[ours]:-0}, dropped ${MODE_COUNT[dropped]:-0}, protected ${MODE_COUNT[protected]:-0}, unknown ${MODE_COUNT[unknown]:-0}) |"
if [ "$RUNTIME_COUNT" -gt 0 ]; then
  fs_report "| retained runtime affected | **yes** — $RUNTIME_COUNT mirrored path(s) under \`src/\`, \`include/\`, \`cmake/\` or \`CMakeLists.txt\` |"
else
  fs_report '| retained runtime affected | no |'
fi
if [ "$DIVERGED_COUNT" -gt 0 ]; then
  fs_report "| fork divergences touched | **yes** — $DIVERGED_COUNT mirrored path(s) carry a local fork patch; upstream changed them too |"
else
  fs_report '| fork divergences touched | no |'
fi
fs_report "| candidate | \`$CANDIDATE\`$([ "$source_changed" -eq 0 ] && echo ' — lock update only: no fork tree changes after policy') |"
fs_report "| ancestry | second parent \`$(fs_short "$UPHEAD")\`; merge with a **merge commit** to keep this fork from reading as behind upstream |"
if [ "$DO_PUSH" -eq 0 ]; then
  fs_report '| published | no — dry run |'
else
  fs_report "| published | \`$SYNC_BRANCH\` on \`$REMOTE\` |"
fi
fs_report ''

if [ "$SENSITIVE_COUNT" -gt 0 ]; then
  fs_report '### Attention'
  fs_report ''
  fs_report '🟡 Upstream touched fork-sensitive areas (build integration, dependency'
  fs_report 'integration, or runtime data) — worth a look before merging:'
  fs_report ''
  printf '%s' "$SENSITIVE_LIST" | fs_report_stdin
  if [ "$SENSITIVE_COUNT" -gt 20 ]; then
    fs_report "  - … and $((SENSITIVE_COUNT - 20)) more"
  fi
  fs_report ''
fi

if [ "$DIVERGED_COUNT" -gt 0 ]; then
  fs_report '### Fork divergences touched by this range'
  fs_report ''
  fs_report 'These mirrored paths already differ from the accepted upstream revision -- they'
  fs_report 'carry a fork patch -- and upstream changed them again.  The three-way merge kept'
  fs_report "your patch and applied upstream's change, and it did so without a conflict: were"
  fs_report 'the edits on the same lines, the run would have stopped instead.  A *clean*'
  fs_report 'merge is a statement about the text, not about the meaning: if upstream fixed'
  fs_report 'the same problem its own way, the merged file can hold a duplicated or'
  fs_report 'contradictory fix.  Read the merged file in the candidate, not just its status.'
  fs_report ''
  printf '%s' "$DIVERGED_LIST" | fs_report_stdin
  if [ "$DIVERGED_COUNT" -gt "$DIVERGED_LIMIT" ]; then
    fs_report "  - … and $((DIVERGED_COUNT - DIVERGED_LIMIT)) more"
  fi
  fs_report ''
  fs_report 'Why each divergence exists is recorded under "Active source divergences" in'
  fs_report '`FORK_NOTES.md`; this list comes from git and does not guess intent.'
  fs_report ''
fi

fs_report '### What changed'
fs_report ''
{
  for a in "${!AREA_MIRROR[@]}"; do printf '%d\t%s\n' "${AREA_MIRROR[$a]}" "$a"; done
  for a in "${!AREA_SUPPRESS[@]}"; do printf '%d\t%s\n' "${AREA_SUPPRESS[$a]}" "$a"; done
} | awk -F'\t' '{t[$2] += $1} END {for (a in t) print t[a] "\t" a}' | sort -rn | head -12 | \
while IFS=$'\t' read -r _ area; do
  took=${AREA_MIRROR[$area]:-0}
  skipped=${AREA_SUPPRESS[$area]:-0}
  parts=''
  [ "$took" -gt 0 ] && parts="$took integrated"
  if [ "$skipped" -gt 0 ]; then
    [ -n "$parts" ] && parts="$parts, "
    parts="${parts}${skipped} suppressed by fork policy"
  fi
  fs_report "- \`$area\` — $parts"
done
fs_report ''

# Commits: complete while it stays readable, bounded when upstream is far ahead.
if [ "$RANGE_COUNT" -le 40 ]; then
  fs_report "<details><summary>Upstream commits ($RANGE_COUNT)</summary>"
  fs_report ''
  for c in "${RANGE[@]}"; do
    fs_report "- \`$(fs_short "$c")\` $(git log -1 --format='%s' "$c")"
  done
  fs_report ''
  fs_report '</details>'
else
  fs_report "<details><summary>Upstream commits ($RANGE_COUNT, showing the oldest 10 and newest 25)</summary>"
  fs_report ''
  for c in "${RANGE[@]:0:10}"; do
    fs_report "- \`$(fs_short "$c")\` $(git log -1 --format='%s' "$c")"
  done
  fs_report ''
  fs_report "_$((RANGE_COUNT - 35)) commits omitted; the whole range is \`$(fs_short "$BASE")..$(fs_short "$UPHEAD")\`._"
  fs_report ''
  for c in "${RANGE[@]: -25}"; do
    fs_report "- \`$(fs_short "$c")\` $(git log -1 --format='%s' "$c")"
  done
  fs_report ''
  fs_report '</details>'
fi
fs_report ''

for mode in ours dropped protected; do
  [ -n "${MODE_LIST[$mode]:-}" ] || continue
  count=${MODE_COUNT[$mode]}
  shown=$count; [ "$shown" -gt "$MODE_LIMIT" ] && shown=$MODE_LIMIT
  omitted=$((count - shown))
  title="$mode — $count path(s) upstream changed that the fork does not take"
  [ "$omitted" -gt 0 ] && title="$title (showing the first $shown)"
  fs_report "<details><summary><b>$title</b></summary>"
  fs_report ''
  fs_report '```'
  printf '%s' "${MODE_LIST[$mode]}" | fs_report_stdin
  fs_report '```'
  if [ "$omitted" -gt 0 ]; then
    fs_report ''
    fs_report "_$omitted more path(s) omitted; list them with_ \`git diff --name-status $(fs_short "$BASE")..$(fs_short "$UPHEAD")\`."
  fi
  fs_report ''
  fs_report '</details>'
  fs_report ''
done

if [ -n "${MODE_LIST[mirror]:-}" ]; then
  count=${MODE_COUNT[mirror]}
  shown=$count; [ "$shown" -gt "$MODE_LIMIT" ] && shown=$MODE_LIMIT
  omitted=$((count - shown))
  title="mirror — $count path(s) taken from upstream"
  [ "$omitted" -gt 0 ] && title="$title (showing the first $shown)"
  fs_report "<details><summary><b>$title</b></summary>"
  fs_report ''
  fs_report '```'
  printf '%s' "${MODE_LIST[mirror]}" | fs_report_stdin
  fs_report '```'
  if [ "$omitted" -gt 0 ]; then
    fs_report ''
    fs_report "_$omitted more path(s) omitted; list them with_ \`git diff --name-status $(fs_short "$BASE")..$(fs_short "$UPHEAD")\`."
  fi
  fs_report ''
  fs_report '</details>'
  fs_report ''
fi

fs_report '### Inspect locally'
fs_report ''
fs_report '```'
fs_report "git log --oneline $(fs_short "$BASE")..$(fs_short "$UPHEAD")"
fs_report "git show $CANDIDATE"
fs_report '```'
if [ "$DO_PUSH" -eq 0 ]; then
  fs_report ''
  fs_report '> **Dry run** — the candidate commit exists locally but was not pushed and no'
  fs_report '> pull request was touched.'
fi
report_footer

if [ "$DO_PR" -eq 1 ]; then
  command -v gh >/dev/null 2>&1 || fs_die "--pr needs the GitHub CLI (gh)"
  [ -n "$PR_BASE" ] || PR_BASE=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')
  number=$(gh pr list --head "$SYNC_BRANCH" --state open --json number --jq '.[0].number // empty')
  if [ -n "$number" ]; then
    gh pr edit "$number" --title "$PR_TITLE" --body-file "$FS_REPORT_PATH" >/dev/null \
      || fs_die "gh pr edit $number failed"
    echo "fork-sync: updated pull request #$number"
  else
    gh pr create --base "$PR_BASE" --head "$SYNC_BRANCH" --title "$PR_TITLE" --body-file "$FS_REPORT_PATH" \
      || fs_die "gh pr create failed (does the repository allow Actions to create pull requests?)"
    echo "fork-sync: created the rolling pull request"
  fi
fi

if [ "$DO_PUSH" -eq 0 ]; then
  echo "fork-sync: dry run; inspect with: git show $CANDIDATE"
else
  echo "fork-sync: done"
fi
