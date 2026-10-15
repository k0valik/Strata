#!/usr/bin/env bash
# .fork/scripts/fork-replay.sh -- replay another fork's work onto this fork as
# a few policy-filtered, per-area squashed commits.
#
#   fork-replay.sh --fork URL --branch NAME              # local branch + report
#   fork-replay.sh --fork URL --branch NAME --pr         # ... and a pull request
#   fork-replay.sh --fork URL --branch NAME --dry-run    # classify and measure only
#   replay-candidates.sh ../../fork-candidates           # run over the candidate list
#
# What it does, in one line: take diff(merge-base(upstream, fork-head),
# fork-head), throw away every path .fork/path-policy does not mirror, and
# replay the rest onto the checked-out branch as one squashed commit per area
# -- so a foreign fork's 500+ commits become a handful of reviewable commits
# against the surface this fork actually kept.  Mixed commits lose nothing:
# grouping is per path, so a commit's engine files land in their area group
# while its Python/docs files vanish under the policy.
#
# Files that refuse to apply come in as three-way merges with conflict markers
# -- a pull request with conflicts, not a rejection.  Only mirrored paths ever
# enter the replay; dropped/ours/protected paths never do.  A fork whose
# history shares no ancestry with upstream (rewritten or grafted commits) is
# diffed against the accepted baseline instead, is flagged HISTORY REPLACED in
# the report, and its whole-file deletions are never replayed automatically.
#
# It is a Git/filesystem tool, not CI: it never builds, tests or runs anything
# that came from the foreign fork (see .fork/README.md; same security model as
# the upstream sync).  The replay happens on a throwaway worktree and branch;
# nothing is pushed and no pull request is opened unless --push/--pr is given.
# The classifier is sourced from the replayed-onto repository's own
# .fork/scripts/lib.sh, so policy semantics cannot drift from the upstream sync.
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

REPO_DIR=$(cd "$SELF_DIR/../.." && pwd)
BASE_REF=HEAD
FORK_URL=''
FORK_BRANCH=''
BRANCH_OUT=''
REMOTE=origin
PR_BASE=''
UPSTREAM_URL_OVERRIDE=''
UPSTREAM_BRANCH_OVERRIDE=''
BASELINE_OVERRIDE=''
ALLOW_LOCAL_UPSTREAM=0
DRY_RUN=0
KEEP_WORKTREE=0
DO_PUSH=0
DO_PR=0
FS_REPORT_PATH=''

# Set as the run progresses; the failure report uses whatever is known.  These must
# all exist from the start: the EXIT trap reports even failures that happen before
# the metadata was read.
BASE=''
UPHEAD=''
FORK_HEAD=''
THEIR_BASE=''
UPSTREAM_URL=''
UPSTREAM_BRANCH=''
FORK_SLUG=''
WT=''
REPLAY_TIP=''
HISTORY_REPLACED=0
PUSHED=0
PR_NUMBER=''

usage() {
  cat <<'EOF'
usage: fork-replay.sh --fork URL --branch NAME [options]

  --repo DIR              repository to replay onto (default: the fork itself)
  --base-ref REF          branch/commit to replay onto (default: HEAD)
  --fork URL              the other fork's repository
  --branch NAME           the other fork's branch to take work from
  --branch-out NAME       replay branch name (default: replay/<slug>)
  --upstream-url URL      override .fork/upstream.lock's repository
  --upstream-branch NAME  override .fork/upstream.lock's branch
  --baseline-ref SHA      override the accepted-baseline commit (default: lock)
  --allow-local-upstream  permit non-https URLs (tests only)
  --dry-run               classify and measure only; replay nothing
  --keep                  keep the replay worktree for inspection
  --push                  push the replay branch to --remote
  --pr                    push and open/update a pull request from the report
  --pr-base NAME          base branch for the pull request (default: gh's default)
  --remote NAME|URL       remote to push to (default: origin)
  --report FILE           write the markdown report to FILE (and it stays on disk;
                          with --pr and no --report a temporary file is used)
  -h, --help              this text
EOF
}

while [ $# -gt 0 ]; do
  case $1 in
    --repo)                 REPO_DIR=$2; shift 2 ;;
    --base-ref)             BASE_REF=$2; shift 2 ;;
    --fork)                 FORK_URL=$2; shift 2 ;;
    --branch)               FORK_BRANCH=$2; shift 2 ;;
    --branch-out)           BRANCH_OUT=$2; shift 2 ;;
    --upstream-url)         UPSTREAM_URL_OVERRIDE=$2; shift 2 ;;
    --upstream-branch)      UPSTREAM_BRANCH_OVERRIDE=$2; shift 2 ;;
    --baseline-ref)         BASELINE_OVERRIDE=$2; shift 2 ;;
    --allow-local-upstream) ALLOW_LOCAL_UPSTREAM=1; shift ;;
    --dry-run)              DRY_RUN=1; shift ;;
    --keep)                 KEEP_WORKTREE=1; shift ;;
    --push)                 DO_PUSH=1; shift ;;
    --pr)                   DO_PUSH=1; DO_PR=1; shift ;;
    --pr-base)              PR_BASE=$2; shift 2 ;;
    --remote)               REMOTE=$2; shift 2 ;;
    --report)               FS_REPORT_PATH=$2; shift 2 ;;
    -h|--help)              usage; exit 0 ;;
    *) printf 'fork-replay: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$FORK_URL" ]    || { usage >&2; exit 2; }
[ -n "$FORK_BRANCH" ] || { usage >&2; exit 2; }

# ---------------------------------------------------------------- reporting --

report_reset() {
  [ -n "$FS_REPORT_PATH" ] || return 0
  : > "$FS_REPORT_PATH"
}

report_start() {                        # report_start <status>
  fs_report "## Fork replay — $1"
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

# The report must never keep showing a healthy replay after a run that failed,
# so the failure path rewrites the report from scratch with whatever is known.
report_blocked() {
  local reason
  reason=$(cat "$FS_ERROR_FILE" 2>/dev/null || true)
  [ -n "$reason" ] || reason=${FS_ERROR:-the run failed; see the log}

  report_reset
  report_start "🔴 BLOCKED"
  [ -n "$BASE" ] && fs_report "| accepted baseline | \`$(fs_short "$BASE")\` |"
  [ -n "$FORK_HEAD" ] && fs_report "| foreign fork head | \`$(fs_short "$FORK_HEAD")\` |"
  [ -n "$THEIR_BASE" ] && fs_report "| their fork point | \`$(fs_short "$THEIR_BASE")\` |"
  if [ -n "$REPLAY_TIP" ]; then
    fs_report "| local branch | \`$BRANCH_OUT\` -> \`$(fs_short "$REPLAY_TIP")\` (partial; the branch remains locally) |"
  fi
  if [ "$PUSHED" -eq 1 ]; then
    fs_report "| replay branch | **pushed** to \`$REMOTE\` before the failure |"
  elif [ -z "$REPLAY_TIP" ]; then
    fs_report '| replay | **not produced** -- nothing was committed or pushed |'
  fi
  fs_report ''
  fs_report '### Reason'
  fs_report ''
  fs_report "$reason"
  report_footer
  return 0
}

report_nothing() {                      # report_nothing <fork-head>
  report_reset
  report_start "🟢 NOTHING TO REPLAY"
  fs_report "| foreign fork head | `$(fs_short "$1")` |"
  fs_report "| their fork point | `${THEIR_BASE:-unknown}` |"
  fs_report ''
  fs_report 'Nothing to replay: the foreign branch sits at its fork point from upstream.'
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

# fr_automation_owned <commit>
# True when <commit> is a replay branch tip this tooling wrote and may replace:
# it carries the replay trailers, names this fork's source, and sits on this
# fork's line of history.  A human commit -- or one that merely copies the
# trailers -- does not pass, so the push refuses instead of overwriting it.
fr_automation_owned() {
  local commit=$1 body
  body=$(git log -1 --format=%B "$commit" 2>/dev/null || true)
  case $body in *'Fork-Replay-Automated: true'*) ;; *) return 1 ;; esac
  case $body in *"Fork-Replay-Source: $FORK_URL"*) ;; *) return 1 ;; esac
  # The branch must grow from this fork's line of history.
  git merge-base --is-ancestor "$(git rev-parse "$BASE_REF")" "$commit" 2>/dev/null
}


# One squashed commit per entry; engine directories get a second level so the
# kernel work does not drown the rest of src/.  The natural next step is a
# small config file next to path-policy if real use wants different chunks.
fr_group_of() {
  local p=$1 d rest
  case $p in */*) d=${p%%/*}; rest=${p#*/} ;; *) printf 'root'; return ;; esac
  case $d in
    src|include|tools)
      case $rest in */*) printf '%s/%s' "$d" "${rest%%/*}" ;; *) printf '%s' "$d" ;; esac ;;
    *) printf '%s' "$d" ;;
  esac
}

# ------------------------------------------------------------------ context --

[ -d "$REPO_DIR" ] || { printf 'fork-replay: error: no such repository: %s\n' "$REPO_DIR" >&2; exit 1; }
cd "$REPO_DIR"
git rev-parse --git-dir >/dev/null 2>&1 || { printf 'fork-replay: error: %s is not a git repository\n' "$REPO_DIR" >&2; exit 1; }
# The classifier comes from the repository being replayed onto, so this tool
# cannot drift from the upstream sync's policy semantics.
[ -f ".fork/scripts/lib.sh" ] || { printf 'fork-replay: error: %s has no .fork/scripts/lib.sh\n' "$REPO_DIR" >&2; exit 1; }
# shellcheck source=.fork/scripts/lib.sh
. .fork/scripts/lib.sh
fs_require_git

work=$(fs_mktempdir work)
if [ -z "$FS_REPORT_PATH" ]; then
  FS_REPORT_PATH=$work/replay-report.md
fi
: > "$FS_REPORT_PATH" || fs_die "cannot write the report to $FS_REPORT_PATH"

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
  if [ "$KEEP_WORKTREE" -eq 1 ] && [ -n "$WT" ]; then
    fs_log "fork-replay: kept the replay worktree at $WT (branch $BRANCH_OUT)"
  else
    if [ -n "$WT" ]; then
      git worktree remove --force "$WT" >/dev/null 2>&1 || true
    fi
    fs_cleanup
  fi
  exit "$rc"
}
trap fs_on_exit EXIT

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
BASE=${BASELINE_OVERRIDE:-$FS_META_COMMIT}

case $UPSTREAM_URL in
  https://*) ;;
  *) [ "$ALLOW_LOCAL_UPSTREAM" = 1 ] \
       || fs_die "upstream URL '$UPSTREAM_URL' is not https (--allow-local-upstream is for tests only)" ;;
esac
case $FORK_URL in
  https://*) ;;
  *) [ "$ALLOW_LOCAL_UPSTREAM" = 1 ] \
       || fs_die "foreign fork URL '$FORK_URL' is not https (--allow-local-upstream is for tests only)" ;;
esac
[[ $BASE =~ ^[0-9a-f]{40}$ ]] || fs_die "recorded baseline '$BASE' is not a full commit id"

FORK_SLUG=$(printf '%s' "$FORK_URL-$FORK_BRANCH" | sed -E 's#https?://##; s#[^A-Za-z0-9._-]#-#g; s#-+$##' | awk '{print substr($0,1,40)}')
BRANCH_OUT=${BRANCH_OUT:-replay/$FORK_SLUG}



echo "fork-replay: repository  $REPO_DIR"
echo "fork-replay: base ref    $FS_BASE_REF -> $(fs_short "$(git rev-parse "$FS_BASE_REF")")"
echo "fork-replay: foreign     $FORK_URL ($FORK_BRANCH)"
echo "fork-replay: upstream    $UPSTREAM_URL ($UPSTREAM_BRANCH)"
echo "fork-replay: accepted    $(fs_short "$BASE")"

# ------------------------------------------------------------ foreign fork --

# Advertisement only: a wrong URL or branch name should cost one round trip,
# not a fetch.
set +e
remote_head=$(git ls-remote --heads "$FORK_URL" "refs/heads/$FORK_BRANCH" 2>/dev/null | awk 'NR==1 {print $1}')
ls_rc=$?
set -e
[ "$ls_rc" -eq 0 ] || fs_die "cannot reach the foreign fork '$FORK_URL' (git ls-remote failed)"
[ -n "$remote_head" ] || fs_die "foreign fork '$FORK_URL' has no branch '$FORK_BRANCH'"
[[ $remote_head =~ ^[0-9a-f]{40}$ ]] || fs_die "the foreign fork returned a strange ref: '$remote_head'"

UP_REF="refs/fork-review/upstream"
FORK_REF="refs/fork-review/$FORK_SLUG"
echo "fork-replay: fetching the foreign fork and upstream (git objects only; nothing is executed)"
git fetch --quiet --no-tags --no-recurse-submodules "$UPSTREAM_URL" "+refs/heads/$UPSTREAM_BRANCH:$UP_REF" \
  || fs_die "cannot fetch upstream $UPSTREAM_URL"
git fetch --quiet --no-tags --no-recurse-submodules "$FORK_URL" "+refs/heads/$FORK_BRANCH:$FORK_REF" \
  || fs_die "cannot fetch $FORK_URL"

FORK_HEAD=$(git rev-parse --verify --quiet "$FORK_REF^{commit}") \
  || fs_die "the fetched foreign ref $FORK_REF does not resolve to a commit"
UPHEAD=$(git rev-parse --verify --quiet "$UP_REF^{commit}") \
  || fs_die "the fetched upstream ref $UP_REF does not resolve to a commit"
[ "$FORK_HEAD" = "$remote_head" ] \
  || fs_warn "the foreign fork moved while replaying ($(fs_short "$remote_head") -> $(fs_short "$FORK_HEAD"))"

git cat-file -e "$BASE^{commit}" 2>/dev/null \
  || fs_die "the accepted baseline $(fs_short "$BASE") is not in the fetched upstream history"

# Their work starts at their fork point from upstream -- not at git's common
# ancestor with our line, which may sit older or newer than their fork point.
# A fork whose history was rewritten (rebased/grafted commits) shares nothing
# with upstream; for those, diff against the accepted baseline instead and say
# so loudly -- the diff then also contains upstream moves this fork has.
HISTORY_REPLACED=0
if ! THEIR_BASE=$(git merge-base "$UPHEAD" "$FORK_HEAD" 2>/dev/null); then
  fs_warn "the foreign fork shares no ancestry with upstream (its history was replaced?)"
  fs_warn "diffing against the accepted baseline $(fs_short "$BASE") instead -- expect noise"
  THEIR_BASE=$BASE
  HISTORY_REPLACED=1
fi
if [ "$HISTORY_REPLACED" -eq 0 ]; then
  git merge-base --is-ancestor "$THEIR_BASE" "$FORK_HEAD" \
    || fs_die "the foreign fork's history was rewritten relative to upstream; a human must decide"
fi

# Only now, with the baseline confirmed present and reachable, can the `pruned`
# rule be evaluated.
fs_path_sets_init "$FS_BASE_REF^{tree}" "$BASE^{tree}"
# ... and the same comparison against *their* fork point tells which retained
# paths already differ from the state their patches were written against.
fs_divergence_init "$THEIR_BASE" "$FS_BASE_REF"

mapfile -t RANGE < <(git rev-list --reverse "$THEIR_BASE..$FORK_HEAD")
RANGE_COUNT=${#RANGE[@]}
BEHIND_UP=$(git rev-list --count "$THEIR_BASE..$UPHEAD")
if [ "$HISTORY_REPLACED" -eq 1 ]; then
  BASE_VS_BASELINE="n/a -- their history shares no ancestry with upstream"
elif git merge-base --is-ancestor "$BASE" "$THEIR_BASE" 2>/dev/null; then
  BASE_VS_BASELINE="+$(git rev-list --count "$BASE..$THEIR_BASE") (their fork point is newer than our accepted baseline)"
elif git merge-base --is-ancestor "$THEIR_BASE" "$BASE" 2>/dev/null; then
  BASE_VS_BASELINE="-$(git rev-list --count "$THEIR_BASE..$BASE") (their fork point is older than our accepted baseline)"
else
  BASE_VS_BASELINE="diverged from our accepted baseline"
fi

if [ "$RANGE_COUNT" -eq 0 ]; then
  echo "fork-replay: nothing to replay (the foreign branch sits at its fork point)"
  report_nothing "$FORK_HEAD"
  exit 0
fi
echo "fork-replay: their base   $(fs_short "$THEIR_BASE") ($BEHIND_UP commit(s) behind upstream; $BASE_VS_BASELINE)"
echo "fork-replay: their range  $RANGE_COUNT commit(s) $(fs_short "$THEIR_BASE")..$(fs_short "$FORK_HEAD")"

git diff --no-renames --name-status "$THEIR_BASE" "$FORK_HEAD" > "$work/changed.txt"
changed_paths=$(wc -l < "$work/changed.txt" | tr -d ' ')

MODE_LIMIT=200
RUNTIME_COUNT=0
SENSITIVE_COUNT=0
SENSITIVE_LIST=''
DIVERGED_COUNT=0
DIVERGED_LIST=''
DIVERGED_LIMIT=20
DELETIONS_SKIPPED=0
DELETIONS_SKIPPED_LIST=''
declare -A MODE_COUNT=() MODE_LIST=() AREA_MIRROR=() AREA_SUPPRESS=() GROUP_FILES=()
mirror_paths=()

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
    # Without shared ancestry we cannot tell "they deleted this" from "upstream
    # added this after their fork point" -- a history-replaced fork's tree
    # predates upstream's newer work, so most deletions here are that second
    # kind.  Never replay them silently; list them for a human instead.
    if [ "$HISTORY_REPLACED" -eq 1 ] && [ "$status" = D ]; then
      AREA_SUPPRESS[$area]=$(( ${AREA_SUPPRESS[$area]:-0} + 1 ))
      DELETIONS_SKIPPED=$((DELETIONS_SKIPPED + 1))
      if [ "$DELETIONS_SKIPPED" -le 20 ]; then
        DELETIONS_SKIPPED_LIST="${DELETIONS_SKIPPED_LIST}  - \`${path}\`"$'\n'
      fi
      continue
    fi
    AREA_MIRROR[$area]=$(( ${AREA_MIRROR[$area]:-0} + 1 ))
    mirror_paths+=("$path")
    g=$(fr_group_of "$path")
    GROUP_FILES[$g]="${GROUP_FILES[$g]:-}$path"$'\n'
    if runtime_path "$path"; then
      RUNTIME_COUNT=$((RUNTIME_COUNT + 1))
    fi
    # A retained path whose content already differs from their fork point:
    # upstream moved since they forked, or this fork patched the file.  The
    # three-way apply layers their change on top of our copy, but "clean" says
    # nothing about meaning -- these are the files to actually read.
    if fs_path_diverged "$path"; then
      DIVERGED_COUNT=$((DIVERGED_COUNT + 1))
      if [ "$DIVERGED_COUNT" -le "$DIVERGED_LIMIT" ]; then
        DIVERGED_LIST="${DIVERGED_LIST}  - \`${path}\` (foreign fork ${status}; this fork's copy differs from their fork point)"$'\n'
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

RAW_STAT=$(git diff --no-renames --stat "$THEIR_BASE" "$FORK_HEAD" | tail -1)
MIRROR_STAT=''
if [ "${#mirror_paths[@]}" -gt 0 ]; then
  MIRROR_STAT=$(git diff --no-renames --stat "$THEIR_BASE" "$FORK_HEAD" -- "${mirror_paths[@]}" | tail -1)
fi
echo "fork-replay: paths       $changed_paths (mirror ${MODE_COUNT[mirror]:-0}, ours ${MODE_COUNT[ours]:-0}, dropped ${MODE_COUNT[dropped]:-0}, protected ${MODE_COUNT[protected]:-0}, unknown ${MODE_COUNT[unknown]:-0})"
echo "fork-replay: diffstat    raw: $RAW_STAT"
if [ -n "$MIRROR_STAT" ]; then
  echo "fork-replay: diffstat    mirror: $MIRROR_STAT"
fi
if [ "$DIVERGED_COUNT" -gt 0 ]; then
  echo "fork-replay: both-sides  $DIVERGED_COUNT mirrored path(s) whose content already differs from their fork point"
fi
if [ "${MODE_COUNT[unknown]:-0}" -gt 0 ]; then
  echo "fork-replay: excluded    ${MODE_COUNT[unknown]} path(s) match no policy rule -- listed in the report"
fi
if [ "$DELETIONS_SKIPPED" -gt 0 ]; then
  echo "fork-replay: deletions   $DELETIONS_SKIPPED skipped (no ancestry: their deletion vs upstream's later addition is indistinguishable)"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  for g in $(printf '%s\n' "${!GROUP_FILES[@]}" | sort); do
    n=$(printf '%s' "${GROUP_FILES[$g]}" | grep -c . || true)
    echo "fork-replay: group       $g: $n file(s)"
  done
  echo "fork-replay: dry run; nothing replayed"
  exit 0
fi

# --------------------------------------------------------------- replay ----

# The replay happens on a throwaway worktree of this fork: one squashed commit
# per area, each produced by applying that area's filtered diff with a
# three-way merge (their fork point is the merge base, our tree is ours, their
# tip is theirs).  Nothing is pushed; the branch exists for reading.

WT=$(fs_mktempdir worktree)
if git show-ref --verify --quiet "refs/heads/$BRANCH_OUT"; then
  # A rerun: rebuild the branch only if this tooling wrote it, moving the ref
  # with a reflog entry rather than deleting anything.
  fr_automation_owned "$(git rev-parse "refs/heads/$BRANCH_OUT")" \
    || fs_die "refusing to rebuild $BRANCH_OUT: the existing local branch was not written by this tooling; inspect it with 'git log --oneline -3 $BRANCH_OUT'"
  git update-ref -m "fork-replay: rebuild $BRANCH_OUT from $(git rev-parse --short=12 "$FS_BASE_REF")" \
    "refs/heads/$BRANCH_OUT" "$(git rev-parse "$FS_BASE_REF")"
  git worktree add -q "$WT" "$BRANCH_OUT" \
    || fs_die "cannot create the replay worktree at $BRANCH_OUT"
else
  git worktree add -q -b "$BRANCH_OUT" "$WT" "$(git rev-parse "$FS_BASE_REF")" \
    || fs_die "cannot create the replay worktree and branch $BRANCH_OUT"
fi
WT_GIT() { git -C "$WT" "$@"; }

fs_identity_setup

fr_gen_patch() {                        # fr_gen_patch <outfile> <path>...
  local out=$1; shift
  git diff --binary --no-renames "$THEIR_BASE" "$FORK_HEAD" -- "$@" > "$out"
}

fr_is_binary() {                        # fr_is_binary <file>: nonempty and not text
  [ -s "$1" ] && ! grep -qI . "$1"
}

# fr_merge_conflict <path>
# The apply refused this file -- its hunks landed on lines this fork has since
# changed.  That is a pull request with conflicts, not a rejection: build the
# file as a real three-way merge (their fork point as base, this fork as ours,
# their tip as theirs) and leave conflict markers in it for the reviewer to
# resolve.  Only mirrored paths ever get here.  Returns 0 when the file now
# stands in the worktree (merged, with or without markers) and sets
# FR_MERGE_CONFLICTS to the merge-file status; returns 1 when the file cannot
# be represented as a merge (binary).
fr_merge_conflict() {
  local p=$1 rc
  local base_f=$work/m-base ours_f=$work/m-ours theirs_f=$work/m-theirs
  if git cat-file -e "$THEIR_BASE:$p" 2>/dev/null; then
    git cat-file blob "$THEIR_BASE:$p" > "$base_f"
  else
    : > "$base_f"
  fi
  if git cat-file -e "$FORK_HEAD:$p" 2>/dev/null; then
    git cat-file blob "$FORK_HEAD:$p" > "$theirs_f"
  else
    : > "$theirs_f"                     # they deleted it: an empty "theirs" side
  fi
  if [ -f "$WT/$p" ]; then
    cp "$WT/$p" "$ours_f"
  else
    : > "$ours_f"                       # new on their side
  fi
  if fr_is_binary "$base_f" || fr_is_binary "$ours_f" || fr_is_binary "$theirs_f"; then
    return 1
  fi
  git merge-file -L "this fork" -L "their fork point $(fs_short "$THEIR_BASE")" \
      -L "their fork ($FORK_SLUG)" "$ours_f" "$base_f" "$theirs_f" >/dev/null
  rc=$?
  if [ "$rc" -eq 255 ]; then
    return 1                            # merge-file itself failed
  fi
  cp "$ours_f" "$WT/$p"
  WT_GIT add -- "$p"
  FR_MERGE_CONFLICTS=$rc                # 0 = merged clean, >0 = N conflict hunks
  return 0
}

fr_commit_group() {                     # fr_commit_group <group> <manual-file>
  local g=$1 manual=$2 msg=$work/msg.txt
  local p nnoop=0 nsubjects noopnote=''
  FR_GROUP_EMPTY=0
  # Stage each applied path according to the state apply left it in: a clean
  # `git apply --3way` already stages its result (including deletions); the
  # `add` below covers the worktree state; merge-file results are plain adds.
  for p in "${APPLIED_PATHS[@]}" "${CONFLICTED_PATHS[@]}"; do
    if [ -e "$WT/$p" ] || [ -L "$WT/$p" ]; then
      WT_GIT add -- "$p"
    elif WT_GIT ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
      WT_GIT rm -q --cached -- "$p"
    fi
  done
  # A merge can legitimately end where this fork already is: their change is
  # already present in our tree (upstream may have taken it since).  Count
  # those, and never let "nothing to commit" kill the run.
  for p in "${APPLIED_PATHS[@]}"; do
    if WT_GIT diff --cached --quiet HEAD -- "$p"; then
      nnoop=$((nnoop + 1))
    fi
  done
  FR_GROUP_NOOP=$nnoop
  if WT_GIT diff --cached --quiet HEAD; then
    FR_GROUP_EMPTY=1
    return 0
  fi
  if [ "$nnoop" -gt 0 ]; then
    noopnote=", $nnoop of which already matched this fork's content"
  fi
  nsubjects=$(printf '%s' "${GROUP_COMMITS:-}" | grep -c . || true)
  {
    printf "replay(%s): %s from %s\n" "$FORK_SLUG" "$g" "$FORK_BRANCH"
    printf '\nSquashed from %s commit(s) %s..%s on %s, restricted to the\n' \
      "$RANGE_COUNT" "$(fs_short "$THEIR_BASE")" "$(fs_short "$FORK_HEAD")" "$FORK_URL"
    printf 'paths this fork still mirrors: %s file(s) under %s -- %s clean,\n' \
      "$((${#APPLIED_PATHS[@]} + ${#CONFLICTED_PATHS[@]}))" "$g" "${#APPLIED_PATHS[@]}"
    printf '%s with conflict markers%s.\n' "${#CONFLICTED_PATHS[@]}" "$noopnote"
    if [ "${#CONFLICTED_PATHS[@]}" -gt 0 ]; then
      printf '\nConflict-marked files -- a pull request with conflicts: resolve them,\n'
      printf 'do not discard the work:\n'
      for p in "${CONFLICTED_PATHS[@]}"; do printf '  %s\n' "$p"; done
    fi
    if [ -s "$manual" ]; then
      printf '%s file(s) could not be represented as a merge (binary conflict?) and\n' \
        "$(grep -c . "$manual" || true)"
      printf 'were left out:\n'
      sed 's/^/  /' "$manual"
    fi
    if [ -n "$GROUP_AUTHORS" ]; then
      printf '\nOriginal authors: %s\n' "$GROUP_AUTHORS"
    fi
    if [ "$nsubjects" -gt 0 ]; then
      printf '\nOriginal commits touching these paths (%s subject(s)):\n' "$nsubjects"
      printf '%s\n' "$GROUP_COMMITS" | head -8 | sed 's/^/  - /'
    fi
    printf '\nReview artifact produced by fork-replay.sh (a spike adapted from\n'
    printf 'sync-upstream.sh): a filtered, squashed replay for reading -- not built,\n'
    printf 'not tested, and not evidence that any of it works.\n'
    printf '\nFork-Replay-Source: %s\n' "$FORK_URL"
    printf 'Fork-Replay-Branch: %s\n' "$FORK_BRANCH"
    printf 'Fork-Replay-Base: %s\n' "$THEIR_BASE"
    printf 'Fork-Replay-Tip: %s\n' "$FORK_HEAD"
    printf 'Fork-Replay-Group: %s\n' "$g"
    printf 'Fork-Replay-Automated: true\n'
  } > "$msg"
  WT_GIT commit -q -F "$msg"
}

REPLAYED_FILES=0
MANUAL_FILE=$work/manual-all.txt
: > "$MANUAL_FILE"
declare -A GROUP_REPORT=()
APPLIED_PATHS=()
CONFLICTED_PATHS=()
ALL_APPLIED=()
ALL_CONFLICTED=()

for g in $(printf '%s\n' "${!GROUP_FILES[@]}" | sort); do
  mapfile -t paths < <(printf '%s' "${GROUP_FILES[$g]}" | sed '/^$/d')
  [ "${#paths[@]}" -gt 0 ] || continue
  fr_gen_patch "$work/group.patch" "${paths[@]}"
  APPLIED_PATHS=()
  CONFLICTED_PATHS=()
  manual=$work/manual-$(printf '%s' "$g" | tr '/' '-').txt
  : > "$manual"
  if WT_GIT apply --binary --3way --whitespace=nowarn "$work/group.patch" 2>/dev/null; then
    APPLIED_PATHS=("${paths[@]}")
  else
    # The group failed as a whole: undo whatever partial application it left,
    # then give each file its own chance.  When even one file refuses, the work
    # is not discarded: the file comes in as a three-way merge with conflict
    # markers (a pull request with conflicts, not a rejection).  Only files
    # that cannot be represented at all (binary conflicts) are left out.
    WT_GIT reset --hard -q HEAD
    for p in "${paths[@]}"; do
      fr_gen_patch "$work/one.patch" "$p"
      if WT_GIT apply --binary --3way --whitespace=nowarn "$work/one.patch" 2>/dev/null; then
        APPLIED_PATHS+=("$p")
      else
        WT_GIT checkout -q HEAD -- "$p" 2>/dev/null \
          || WT_GIT rm -q --cached -- "$p" 2>/dev/null || true
        if fr_merge_conflict "$p"; then
          if [ "$FR_MERGE_CONFLICTS" -eq 0 ]; then
            APPLIED_PATHS+=("$p")     # merged clean after all
          else
            CONFLICTED_PATHS+=("$p")
          fi
        else
          printf '%s\n' "$p" >> "$manual"
        fi
      fi
    done
  fi
  GROUP_AUTHORS=$(git log --format='%an' "$THEIR_BASE..$FORK_HEAD" -- "${paths[@]}" 2>/dev/null | sort -u | head -4 | paste -sd, -)
  GROUP_COMMITS=$(git log --format='%s' "$THEIR_BASE..$FORK_HEAD" -- "${paths[@]}" 2>/dev/null | sort -u)
  nclean=${#APPLIED_PATHS[@]}
nconf=${#CONFLICTED_PATHS[@]}
  if [ "$((nclean + nconf))" -gt 0 ]; then
    fr_commit_group "$g" "$manual"
    ALL_APPLIED+=("${APPLIED_PATHS[@]}")
    ALL_CONFLICTED+=("${CONFLICTED_PATHS[@]}")
    if [ "$FR_GROUP_EMPTY" -eq 1 ]; then
      GROUP_REPORT[$g]="no content change -- their result already matches this fork's tree"
      echo "fork-replay: group       $g: no content change (their work is already present here)"
    else
      REPLAYED_FILES=$((REPLAYED_FILES + nclean + nconf))
      gr="$((nclean + nconf)) applied ($nclean clean, $nconf conflict-marked)"
      if [ "$FR_GROUP_NOOP" -gt 0 ]; then
        gr="$gr, $FR_GROUP_NOOP already matched"
      fi
      if [ -s "$manual" ]; then
        gr="$gr, $(grep -c . "$manual" || true) not representable"
      fi
      GROUP_REPORT[$g]="$gr"
      echo "fork-replay: group       $g: $gr"
    fi
  elif [ -s "$manual" ]; then
    GROUP_REPORT[$g]="0 applied, $(grep -c . "$manual" || true) not representable"
    echo "fork-replay: group       $g: nothing applied"
  fi
  if [ -s "$manual" ]; then
    cat "$manual" >> "$MANUAL_FILE"
  fi
done

REPLAY_TIP=$(git rev-parse "$BRANCH_OUT")
echo "fork-replay: branch      $BRANCH_OUT -> $(fs_short "$REPLAY_TIP") ($RANGE_COUNT foreign commit(s) -> $(git rev-list --count "$FS_BASE_REF..$REPLAY_TIP") squashed commit(s), $REPLAYED_FILES file(s))"

# ---------------------------------------------------------- verification --

# 1. structural: the replay may touch only paths classified mirror -- anything
#    else on the branch means the policy filter leaked
leaked=0
declare -A MIRROR_SET=()
for p in "${mirror_paths[@]}"; do MIRROR_SET[$p]=1; done
while IFS= read -r p; do
  [ -n "$p" ] || continue
  if [ -z "${MIRROR_SET[$p]+x}" ]; then
    printf 'fork-replay: VERIFY LEAK: %s\n' "$p" >&2
    leaked=$((leaked + 1))
  fi
done < <(git diff --no-renames --name-only "$FS_BASE_REF" "$REPLAY_TIP")

# 2. files whose copy matches their fork point exactly must come out
#    byte-identical to the foreign fork's version; files we changed ourselves
#    are three-way merge results and get read, not diffed
pristine_ok=0
pristine_bad=0
for p in "${ALL_APPLIED[@]}"; do
  [ -n "$p" ] || continue
  if fs_path_diverged "$p"; then
    continue
  fi
  a=$(git rev-parse -q --verify "$REPLAY_TIP:$p" 2>/dev/null || echo none)
  b=$(git rev-parse -q --verify "$FORK_HEAD:$p" 2>/dev/null || echo none)
  if [ "$a" = "$b" ]; then
    pristine_ok=$((pristine_ok + 1))
  else
    printf 'fork-replay: VERIFY MISMATCH: %s\n' "$p" >&2
    pristine_bad=$((pristine_bad + 1))
  fi
done
echo "fork-replay: verify      structural leaks: $leaked | pristine files exact: $pristine_ok, mismatched: $pristine_bad | conflict-marked: ${#ALL_CONFLICTED[@]}"

# --------------------------------------------------------------- publish ---

# Done before the report is written, so the report can state what actually
# happened.  Nothing is pushed unless --push/--pr asked for it.
if [ "$DO_PUSH" -eq 1 ]; then
  existing=$(git ls-remote --heads "$REMOTE" "refs/heads/$BRANCH_OUT" 2>/dev/null | awk 'NR==1 {print $1}')
  if [ -n "$existing" ]; then
    git fetch --quiet --no-tags --no-recurse-submodules "$REMOTE" "+refs/heads/$BRANCH_OUT:refs/fork-review/rolling" \
      || fs_die "cannot read the existing replay branch $BRANCH_OUT from $REMOTE"
    fr_automation_owned "$existing" \
      || fs_die "refusing to overwrite $BRANCH_OUT on $REMOTE: its tip ($(fs_short "$existing")) was not written by this tooling, so it was left alone; inspect it with 'git log --oneline -3 $existing'"
    lease="--force-with-lease=refs/heads/$BRANCH_OUT:$existing"
  else
    lease="--force-with-lease=refs/heads/$BRANCH_OUT:"
  fi
  git push --quiet $lease "$REMOTE" "$REPLAY_TIP:refs/heads/$BRANCH_OUT" \
    || fs_die "push of $BRANCH_OUT to $REMOTE failed (it may have moved; re-run)"
  PUSHED=1
  echo "fork-replay: pushed      $BRANCH_OUT -> $REMOTE"
fi

# --------------------------------------------------------------- the report --

report_reset
if [ "$HISTORY_REPLACED" -eq 1 ]; then
  report_start "🟡 REVIEW — HISTORY REPLACED"
elif [ "$SENSITIVE_COUNT" -gt 0 ] || [ "$DIVERGED_COUNT" -gt 0 ]; then
  report_start "🟡 REVIEW"
else
  report_start "🟢 READY"
fi
fs_report "| foreign fork | \`${FORK_URL}\` (\`${FORK_BRANCH}\`) |"
fs_report "| foreign commits | $RANGE_COUNT (\`$(fs_short "$THEIR_BASE")..$(fs_short "$FORK_HEAD")\`) |"
fs_report "| their fork point | $BEHIND_UP commit(s) behind upstream; $BASE_VS_BASELINE |"
if [ "$HISTORY_REPLACED" -eq 1 ]; then
  fs_report "| history | **replaced** — no shared ancestry with upstream; diffed against the accepted baseline, so expect upstream moves inside the diff |"
fi
fs_report "| replayed onto | \`$(fs_short "$(git rev-parse "$FS_BASE_REF")")\` (\`$FS_BASE_REF\`) |"
fs_report "| raw change | $RAW_STAT |"
if [ "$HISTORY_REPLACED" -eq 1 ]; then
  fs_report "| **replayed change** | ${MIRROR_STAT:-none} — includes upstream moves shown in reverse; see the history-replaced section |"
else
  fs_report "| **replayed change** | **${MIRROR_STAT:-none}** |"
fi
fs_report "| filtered by policy | ${MODE_COUNT[dropped]:-0} dropped, ${MODE_COUNT[ours]:-0} ours, ${MODE_COUNT[protected]:-0} protected, ${MODE_COUNT[unknown]:-0} unclassified |"
if [ "$DELETIONS_SKIPPED" -gt 0 ]; then
  fs_report "| deletions skipped | $DELETIONS_SKIPPED mirror deletion(s) not replayed (history replaced: their deletion vs upstream's later addition is indistinguishable) |"
fi
fs_report "| replay branch | \`$BRANCH_OUT\` — $(git rev-list --count "$FS_BASE_REF..$REPLAY_TIP") squashed commit(s), $REPLAYED_FILES file(s) |"
fs_report "| verification | $leaked structural leak(s); pristine files: $pristine_ok exact / $pristine_bad mismatched; both-sides-changed: $DIVERGED_COUNT; conflict-marked: ${#ALL_CONFLICTED[@]} |"
if [ "$PUSHED" -eq 1 ]; then
  publishednote=' — a pull request opens from this report'
  if [ "$DO_PR" -eq 0 ]; then publishednote=''; fi
  fs_report "| published | \`$BRANCH_OUT\` pushed to \`$REMOTE\`$publishednote |"
else
  fs_report '| published | no — the branch and the worktree are local only |'
fi
fs_report ''

if [ "$HISTORY_REPLACED" -eq 1 ]; then
  fs_report '### ⚠️ History replaced — read this before any diff hunk'
  fs_report ''
  fs_report 'This fork shares **no commit ancestry** with upstream (its history was rewritten'
  fs_report "or grafted), so there is no fork point to diff against.  The diff here is their"
  fs_report "tree against this fork's accepted upstream baseline, which means:"
  fs_report ''
  fs_report "- it may contain upstream's own post-fork changes **in reverse** — replaying a"
  fs_report '  file can regress upstream progress this fork already has;'
  fs_report '- the diffstat numbers are not "their work" — treat them as an upper bound;'
  fs_report '- whole-file deletions are therefore never replayed automatically (see the'
  fs_report '  deletions section below when present);'
  fs_report '- every applied file deserves a per-file read — the byte-exact verification'
  fs_report '  here means "matches their tree", **not** "safe to take".'
  fs_report ''
fi

if [ "$SENSITIVE_COUNT" -gt 0 ]; then
  fs_report '### Attention'
  fs_report ''
  fs_report '🟡 The foreign fork touched fork-sensitive areas (build integration,'
  fs_report 'dependency integration, or runtime data) — worth a look before adopting anything:'
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
  fs_report 'These mirrored paths already differ from their fork point — upstream moved'
  fs_report 'since they forked, or this fork patched them — and the foreign fork changed'
  fs_report "them too. The three-way apply kept this fork's version and layered their"
  fs_report 'change on top, and it did so without a conflict: were the edits on the same'
  fs_report 'lines, the file would have landed in the manual list instead. A *clean* apply'
  fs_report 'is a statement about the text, not about the meaning: if the foreign fork'
  fs_report 'fixed the same thing its own way, the replayed file can hold a duplicated or'
  fs_report 'contradictory fix. Read the replayed file, not just its status.'
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
  fs_report "<details><summary>Foreign fork commits ($RANGE_COUNT)</summary>"
  fs_report ''
  for c in "${RANGE[@]}"; do
    fs_report "- \`$(fs_short "$c")\` $(git log -1 --format='%s' "$c")"
  done
  fs_report ''
  fs_report '</details>'
else
  fs_report "<details><summary>Foreign fork commits ($RANGE_COUNT, showing the oldest 10 and newest 25)</summary>"
  fs_report ''
  for c in "${RANGE[@]:0:10}"; do
    fs_report "- \`$(fs_short "$c")\` $(git log -1 --format='%s' "$c")"
  done
  fs_report ''
  fs_report "_$((RANGE_COUNT - 35)) commits omitted; the whole range is \`$(fs_short "$THEIR_BASE")..$(fs_short "$FORK_HEAD")\`._"
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
  title="$mode — $count path(s) the foreign fork changed that this fork does not take"
  [ "$omitted" -gt 0 ] && title="$title (showing the first $shown)"
  fs_report "<details><summary><b>$title</b></summary>"
  fs_report ''
  fs_report '```'
  printf '%s' "${MODE_LIST[$mode]}" | fs_report_stdin
  fs_report '```'
  if [ "$omitted" -gt 0 ]; then
    fs_report ''
    fs_report "_$omitted more path(s) omitted; list them with_ \`git diff --name-status $(fs_short "$THEIR_BASE")..$(fs_short "$FORK_HEAD")\`."
  fi
  fs_report ''
  fs_report '</details>'
  fs_report ''
done

if [ -n "${MODE_LIST[mirror]:-}" ]; then
  count=${MODE_COUNT[mirror]}
  shown=$count; [ "$shown" -gt "$MODE_LIMIT" ] && shown=$MODE_LIMIT
  omitted=$((count - shown))
  title="mirror — $count path(s) taken from the foreign fork"
  [ "$omitted" -gt 0 ] && title="$title (showing the first $shown)"
  fs_report "<details><summary><b>$title</b></summary>"
  fs_report ''
  fs_report '```'
  printf '%s' "${MODE_LIST[mirror]}" | fs_report_stdin
  fs_report '```'
  if [ "$omitted" -gt 0 ]; then
    fs_report ''
    fs_report "_$omitted more path(s) omitted; list them with_ \`git diff --name-status $(fs_short "$THEIR_BASE")..$(fs_short "$FORK_HEAD")\`."
  fi
  fs_report ''
  fs_report '</details>'
  fs_report ''
fi

fs_report '### Groups (one squashed commit each)'
fs_report ''
for g in $(printf '%s\n' "${!GROUP_FILES[@]}" | sort); do
  if [ -n "${GROUP_REPORT[$g]:-}" ]; then
    fs_report "- \`$g\` — ${GROUP_REPORT[$g]} file(s)"
  fi
done
fs_report ''

if [ "$DELETIONS_SKIPPED" -gt 0 ]; then
  fs_report '### Deletions not replayed (history replaced)'
  fs_report ''
  fs_report 'This fork shares no ancestry with the foreign fork, so a path missing from'
  fs_report 'their tree may be either their deletion or upstream work from after their'
  fs_report 'fork point.  Deletions are therefore never replayed automatically; this is'
  fs_report 'the list to look at:'
  fs_report ''
  printf '%s' "$DELETIONS_SKIPPED_LIST" | fs_report_stdin
  if [ "$DELETIONS_SKIPPED" -gt 20 ]; then
    fs_report "  - … and $((DELETIONS_SKIPPED - 20)) more"
  fi
  fs_report ''
fi

if [ "${#ALL_CONFLICTED[@]}" -gt 0 ]; then
  fs_report '### Applied with conflict markers — resolve, do not discard'
  fs_report ''
  fs_report "These files carry this fork's version and the foreign fork's version in one file,"
  fs_report 'separated by conflict markers — exactly like a pull request with conflicts. The'
  fs_report 'work in them is not rejected: read each file, decide what to keep, and resolve'
  fs_report 'the markers consciously. Only mirrored paths ever reach this state; dropped,'
  fs_report 'ours and protected paths never enter the replay at all.'
  fs_report ''
  fs_report '```'
  printf '%s\n' "${ALL_CONFLICTED[@]}" | sort -u | fs_report_stdin
  fs_report '```'
  fs_report ''
fi

if [ -s "$MANUAL_FILE" ]; then
  fs_report '### Not representable as a merge (a human decides)'
  fs_report ''
  fs_report '```'
  sort -u "$MANUAL_FILE" | fs_report_stdin
  fs_report '```'
  fs_report ''
fi

if [ "${MODE_COUNT[unknown]:-0}" -gt 0 ]; then
  fs_report '### Unclassified paths (excluded; they need a policy decision first)'
  fs_report ''
  fs_report '```'
  awk -F'\t' 'NF >= 2 { print $2 }' "$work/changed.txt" | while IFS= read -r p; do
    if [ "$(fs_classify "$p")" = unknown ]; then
      printf '%s\n' "$p"
    fi
  done | fs_report_stdin
  fs_report '```'
  fs_report ''
fi

fs_report '### Inspect locally'
fs_report ''
fs_report '```'
fs_report "git log --oneline $FS_BASE_REF..$BRANCH_OUT"
fs_report "git diff $FS_BASE_REF...$BRANCH_OUT --stat"
fs_report "git log --oneline $(fs_short "$THEIR_BASE")..$(fs_short "$FORK_HEAD")   # the original commits"
fs_report '```'
fs_report ''
if [ "$DO_PUSH" -eq 0 ]; then
  fs_report '> **Review artifact** — produced by fork-replay.sh; nothing was pushed. The'
  fs_report '> replayed commits exist for reading: building and testing them is a human'
  fs_report '> step, exactly as for a synchronization candidate.'
else
  fs_report '> **Review artifact** — produced by fork-replay.sh. The replayed commits exist'
  fs_report '> for reading: building and testing them is a human step, exactly as for a'
  fs_report '> synchronization candidate. Nothing here is evidence that any of it works.'
fi
report_footer

# The pull request is created from the report just written; the report file is
# its body.  gh is invoked with the repository named explicitly, so a
# mis-resolved default can never direct a mutating call at another repository.
if [ "$DO_PR" -eq 1 ]; then
  command -v gh >/dev/null 2>&1 || fs_die "--pr needs the GitHub CLI (gh)"
  [ "$PUSHED" -eq 1 ] || fs_die "--pr needs --push (the head branch must exist on $REMOTE first)"
  [ -n "$PR_BASE" ] || PR_BASE=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')
  REPO_SLUG=$(gh repo view --json nameWithOwner --jq '.nameWithOwner')
  PR_TITLE="replay (review only): $FORK_SLUG — filtered squash of $RANGE_COUNT commit(s) from $FORK_URL"
  number=$(gh pr list --repo "$REPO_SLUG" --head "$BRANCH_OUT" --state open --json number --jq '.[0].number // empty')
  if [ -n "$number" ]; then
    # gh pr edit goes through GraphQL, whose projectCards query fails on
    # repositories where classic projects were used; the REST API is used
    # here so updating the body never depends on that.
    gh api "repos/$REPO_SLUG/pulls/$number" -X PATCH \
      -F title="$PR_TITLE" -F body="@$FS_REPORT_PATH" >/dev/null \
      || fs_die "updating pull request $number failed"
    PR_NUMBER=$number
    echo "fork-replay: pull request updated #$PR_NUMBER"
  else
    gh pr create --repo "$REPO_SLUG" --base "$PR_BASE" --head "$BRANCH_OUT" \
      --title "$PR_TITLE" --body-file "$FS_REPORT_PATH" >/dev/null \
      || fs_die "gh pr create failed (does the repository allow creating pull requests?)"
    PR_NUMBER=$(gh pr list --repo "$REPO_SLUG" --head "$BRANCH_OUT" --state open --json number --jq '.[0].number')
    echo "fork-replay: pull request opened  #$PR_NUMBER"
  fi
fi

echo "fork-replay: done; inspect with: git log --oneline $FS_BASE_REF..$BRANCH_OUT"
