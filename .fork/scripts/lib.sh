#!/usr/bin/env bash
# .fork/scripts/lib.sh -- shared helpers for the fork's upstream-sync tooling.
#
# Git plumbing only: the synchronization reads commits, trees and blobs, and
# never checks out, builds, imports or executes anything that came from upstream
# (see .fork/README.md for the security model).
#
# Sourced by sync-upstream.sh and validate-sync.sh.  Needs bash >= 4 and
# git >= 2.38 (git merge-tree --write-tree / --merge-base).

FS_MIN_GIT=2.38
FS_ERROR=''

# --------------------------------------------------------------- utilities --

fs_log()  { printf '%s\n' "$*" >&2; }
fs_warn() { printf 'fork-sync: warning: %s\n' "$*" >&2; }

fs_die() {
  printf 'fork-sync: error: %s\n' "$*" >&2
  FS_ERROR="$*"
  # Subshells (e.g. an error raised inside a $(command substitution)) cannot set
  # FS_ERROR for the parent, so the reason also goes to a shared file.
  if [ -n "${FS_ERROR_FILE:-}" ]; then
    printf '%s\n' "$*" > "$FS_ERROR_FILE" 2>/dev/null || true
  fi
  exit 1
}

# fs_run_dir -- the throwaway directory for this run.
#
# One directory per run, named after the top-level shell's pid.  The name is derived
# rather than remembered because of subshells: `$(fs_mktempdir ...)` runs in one, so an
# `FS_TMPDIRS+=` made in there never reaches the parent and the exit trap then cleans up
# nothing -- every run used to leak a directory into /tmp.  `$$` is stable inside
# subshells, so every sub-directory lands under the same root and the parent's trap can
# find the whole tree again.
fs_run_dir() { printf '%s' "${TMPDIR:-/tmp}/fork-sync.$$"; }

# fs_mktempdir <name> -- create `<run-dir>/<name>` and print it.
fs_mktempdir() {
  local d
  d="$(fs_run_dir)/$1"
  mkdir -p "$d" || fs_die "cannot create a working directory at $d"
  printf '%s' "$d"
}

fs_cleanup() {
  local d
  d=$(fs_run_dir)
  # Only ever remove this run's own directory: the basename must be fork-sync.<pid>.
  if [[ ${d##*/} =~ ^fork-sync\.[0-9]+$ ]]; then
    rm -rf -- "$d"
  fi
  return 0
}

fs_require_git() {
  command -v git >/dev/null 2>&1 || fs_die "git is required"
  local v maj min
  v=$(git --version 2>/dev/null | awk '{print $3}')
  maj=${v%%.*}
  min=${v#*.}; min=${min%%.*}
  case $maj in ''|*[!0-9]*) fs_die "cannot determine the git version ('$v')" ;; esac
  if [ "$maj" -lt 2 ] || { [ "$maj" -eq 2 ] && [ "$min" -lt 38 ]; }; then
    fs_die "git >= $FS_MIN_GIT is required (git merge-tree --write-tree); found $v"
  fi
}

# fs_short <sha> -- the abbreviated form used in messages.
fs_short() { printf '%s' "${1:0:12}"; }

# ---------------------------------------------------------------- metadata --

# fs_meta_parse <file>
# Sets FS_META_URL, FS_META_BRANCH, FS_META_COMMIT.  Non-zero (with a message on
# stderr) if the file is malformed.
fs_meta_parse() {
  local file=$1 line key val

  [ -f "$file" ] || { fs_log "fork-sync: metadata file not found: $file"; return 2; }

  FS_META_URL=''
  FS_META_BRANCH=''
  FS_META_COMMIT=''

  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    if [[ $line =~ ^[[:space:]]*$ ]]; then
      continue
    fi
    if [[ $line =~ ^[[:space:]]*([A-Za-z_][A-Za-z_0-9]*)[[:space:]]*=[[:space:]]*([^[:space:]]+)[[:space:]]*$ ]]; then
      key=${BASH_REMATCH[1]}
      val=${BASH_REMATCH[2]}
    else
      fs_log "fork-sync: $file: cannot parse line: $line"
      return 2
    fi
    case $key in
      repository)
        [ -z "$FS_META_URL" ] || { fs_log "fork-sync: $file: duplicate key '$key'"; return 2; }
        FS_META_URL=$val ;;
      branch)
        [ -z "$FS_META_BRANCH" ] || { fs_log "fork-sync: $file: duplicate key '$key'"; return 2; }
        FS_META_BRANCH=$val ;;
      commit)
        [ -z "$FS_META_COMMIT" ] || { fs_log "fork-sync: $file: duplicate key '$key'"; return 2; }
        FS_META_COMMIT=$val ;;
      *)
        fs_log "fork-sync: $file: unknown key '$key'"
        return 2 ;;
    esac
  done < "$file"

  local missing=''
  [ -n "$FS_META_URL" ]    || missing="$missing repository"
  [ -n "$FS_META_BRANCH" ] || missing="$missing branch"
  [ -n "$FS_META_COMMIT" ] || missing="$missing commit"
  if [ -n "$missing" ]; then
    fs_log "fork-sync: $file: missing key(s):$missing"
    return 2
  fi
  return 0
}

# --------------------------------------------------------------- path policy --

# fs_policy_load <file>
# Fills FS_POLICY_MODE[] / FS_POLICY_GLOB[] in file order.  A line `pruned` is
# the built-in rule for paths the accepted upstream revision had and this fork
# does not.
fs_policy_load() {
  local file=$1 line mode rest n=0

  FS_POLICY_MODE=()
  FS_POLICY_GLOB=()

  [ -f "$file" ] || { fs_log "fork-sync: policy file not found: $file"; return 2; }

  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    line=${line#"${line%%[![:space:]]*}"}   # ltrim
    line=${line%"${line##*[![:space:]]}"}   # rtrim
    [ -n "$line" ] || continue

    mode=${line%%[[:space:]]*}
    rest=${line#"$mode"}
    rest=${rest#"${rest%%[![:space:]]*}"}
    rest=${rest%"${rest##*[![:space:]]}"}

    case $mode in
      pruned)
        [ -z "$rest" ] || { fs_log "fork-sync: $file: 'pruned' takes no argument"; return 2; }
        FS_POLICY_MODE+=("pruned")
        FS_POLICY_GLOB+=("")
        n=$((n + 1)) ;;
      mirror|dropped|ours|protected)
        [ -n "$rest" ] || { fs_log "fork-sync: $file: '$mode' needs a glob"; return 2; }
        FS_POLICY_MODE+=("$mode")
        FS_POLICY_GLOB+=("$rest")
        n=$((n + 1)) ;;
      *)
        fs_log "fork-sync: $file: unknown rule mode '$mode'"
        return 2 ;;
    esac
  done < "$file"

  [ "$n" -gt 0 ] || { fs_log "fork-sync: $file: no rules"; return 2; }
  return 0
}

# fs_glob_match <glob> <path>
# Bash's own pattern matching: `*` also crosses '/', so `src/**` and `src/*`
# behave the same.  The path is quoted so its own glob characters stay literal.
fs_glob_match() {
  [ -n "$1" ] || return 1
  case "$2" in
    $1) return 0 ;;
    *)  return 1 ;;
  esac
}

# fs_path_sets_init <main-tree> <baseline-tree>
# Builds the two path sets the `pruned` rule needs.
fs_path_sets_init() {
  local p
  declare -gA FS_IN_MAIN=() FS_IN_BASELINE=()
  while IFS= read -r p; do
    [ -n "$p" ] && FS_IN_MAIN[$p]=1
  done < <(git ls-tree -r --name-only "$1")
  while IFS= read -r p; do
    [ -n "$p" ] && FS_IN_BASELINE[$p]=1
  done < <(git ls-tree -r --name-only "$2")
}

fs_path_in_main()     { [ -n "${FS_IN_MAIN[$1]+x}" ]; }
fs_path_in_baseline() { [ -n "${FS_IN_BASELINE[$1]+x}" ]; }

# fs_divergence_init <baseline-commit> <fork-ref>
# Records the paths whose content differs between the accepted upstream revision
# and the fork: the fork's own patches, deletions and additions.  Combined with
# fs_path_in_baseline()/fs_path_in_main() this distinguishes a pristine mirrored
# file from one that carries a local fork patch -- i.e. "upstream just edited
# this" from "upstream is about to merge on top of a change of ours".
fs_divergence_init() {
  local p
  declare -gA FS_DIVERGED=()
  while IFS= read -r p; do
    [ -n "$p" ] && FS_DIVERGED[$p]=1
  done < <(git diff --no-renames --name-only "$1" "$2")
}

fs_path_diverged() { [ -n "${FS_DIVERGED[$1]+x}" ]; }

# fs_classify <path> -> mirror | dropped | ours | protected | unknown
fs_classify() {
  local path=$1 i mode
  for ((i = 0; i < ${#FS_POLICY_MODE[@]}; i++)); do
    mode=${FS_POLICY_MODE[$i]}
    if [ "$mode" = pruned ]; then
      if fs_path_in_baseline "$path" && ! fs_path_in_main "$path"; then
        printf 'dropped'
        return 0
      fi
      continue
    fi
    if fs_glob_match "${FS_POLICY_GLOB[$i]}" "$path"; then
      printf '%s' "$mode"
      return 0
    fi
  done
  printf 'unknown'
}

# ------------------------------------------------------------------ merging --

# fs_merge_tree <base-commit> <ours-commit> <theirs-commit>
# Runs git's three-way merge in memory -- no checkout, no worktree, nothing
# written outside the object database -- using the *accepted upstream commit* as
# the merge base rather than git's own merge base.  That is what makes the lock
# file, and not the fork's history, the source of truth: after a synchronization
# has been squash-merged, git no longer knows that the fork contains upstream's
# older commits, but the lock file does.
#
# Sets: FS_MERGE_RC (0 clean, 1 conflicted), FS_MERGE_TREE, FS_MERGE_CONFLICTS,
#       FS_MERGE_OUTPUT.
fs_merge_tree() {
  local out rc
  set +e
  out=$(git merge-tree --write-tree --merge-base="$1" --messages "$2" "$3" 2>&1)
  rc=$?
  set -e

  FS_MERGE_RC=$rc
  FS_MERGE_OUTPUT=$out
  FS_MERGE_TREE=$(printf '%s\n' "$out" | sed -n '1p')
  # Conflicted paths come back as "<mode> <blob> <stage>\t<path>" lines.
  FS_MERGE_CONFLICTS=$(printf '%s\n' "$out" | awk -F'\t' 'NF >= 2 && $1 ~ /^[0-7]+ [0-9a-f]+ [1-3]$/ { print $2 }' | sort -u)

  case $rc in
    0|1) ;;
    *) fs_die "git merge-tree failed for $(fs_short "$1")..$(fs_short "$3") (exit $rc): $out" ;;
  esac
  [ -n "$FS_MERGE_TREE" ] || fs_die "git merge-tree produced no tree for $(fs_short "$1")..$(fs_short "$3")"
}

# fs_restore_from_main <main-tree> <path>
# Forces the index entry for <path> back to the accepted fork tree's version
# (or removes it when the fork does not have it).
fs_restore_from_main() {
  local tree=$1 path=$2 entry mode sha
  entry=$(git ls-tree "$tree" -- "$path")
  if [ -n "$entry" ]; then
    mode=${entry%% *}
    sha=${entry#* }; sha=${sha#* }; sha=${sha%%[[:space:]]*}
    git update-index --add --cacheinfo "$mode,$sha,$path"
  else
    git update-index --force-remove -- "$path" 2>/dev/null || true
  fi
}

# fs_build_candidate_tree <main-tree> <merged-tree> <conflict-paths>
# Applies the path policy to a merge result and prints the tree to use.
# Classifications that need a human decision stop the run, listing all of them.
fs_build_candidate_tree() {
  local main=$1 merged=$2 conflicts=$3
  local dir idx path cpath mode b n
  local -a blockers=()

  dir=$(fs_mktempdir candidate-tree)
  idx=$dir/index
  export GIT_INDEX_FILE=$idx

  git read-tree "$merged"

  while IFS= read -r -d '' path; do
    [ -n "$path" ] || continue
    mode=$(fs_classify "$path")
    case $mode in
      mirror) : ;;
      ours|dropped) fs_restore_from_main "$main" "$path" ;;
      protected) blockers+=("[protected] $path") ;;
      unknown)   blockers+=("[unclassified] $path") ;;
      *)         blockers+=("[internal] $path ($mode)") ;;
    esac
  done < <(git diff --no-renames --name-only -z "$main" "$merged")

  # A conflict on a path the fork does not take was resolved above; a conflict on a
  # mirrored path is a genuine disagreement and needs a human.
  while IFS= read -r cpath; do
    [ -n "$cpath" ] || continue
    mode=$(fs_classify "$cpath")
    case $mode in
      ours|dropped) : ;;
      mirror)    blockers+=("[conflict] $cpath") ;;
      protected) blockers+=("[protected conflict] $cpath") ;;
      unknown)   blockers+=("[unclassified conflict] $cpath") ;;
    esac
  done <<< "$conflicts"

  if [ "${#blockers[@]}" -gt 0 ]; then
    printf 'fork-sync: %d path(s) need a human decision:\n' "${#blockers[@]}" >&2
    n=0
    for b in "${blockers[@]}"; do
      n=$((n + 1))
      if [ "$n" -le 20 ]; then
        printf 'fork-sync:   %s\n' "$b" >&2
      fi
    done
    if [ "$n" -gt 20 ]; then
      printf 'fork-sync:   ... and %d more\n' "$((n - 20))" >&2
    fi
    fs_die "${#blockers[@]} path(s) need a human decision: $(printf '%s; ' "${blockers[@]:0:5}")"
  fi

  git write-tree
  unset GIT_INDEX_FILE
}

# fs_write_lock <base-ref> <new-upstream-commit>
# Rewrites the `commit` line of .fork/upstream.lock in the current index.
fs_write_lock() {
  local base=$1 new=$2 content blob
  content=$(git show "$base:.fork/upstream.lock")
  content=$(printf '%s\n' "$content" | sed -E "s|^(commit[[:space:]]*=[[:space:]]*)[0-9a-fA-F]{40}[[:space:]]*$|\1$new|")
  case $content in
    *"$new"*) ;;
    *) fs_die "could not update .fork/upstream.lock (no 'commit = ...' line matched)";;
  esac
  blob=$(printf '%s\n' "$content" | git hash-object -w --stdin)
  git update-index --add --cacheinfo "100644,$blob,.fork/upstream.lock"
}

# --------------------------------------------------------------- git commits --

fs_identity_setup() {
  local name email
  name=${FORK_SYNC_AUTHOR_NAME:-$(git config user.name 2>/dev/null || true)}
  email=${FORK_SYNC_AUTHOR_EMAIL:-$(git config user.email 2>/dev/null || true)}
  [ -n "$name" ]  || name='fork-sync bot'
  [ -n "$email" ] || email='fork-sync-bot@users.noreply.github.com'
  export GIT_AUTHOR_NAME=$name GIT_AUTHOR_EMAIL=$email
  export GIT_COMMITTER_NAME=$name GIT_COMMITTER_EMAIL=$email
}

# fs_commit_tree <tree> <base-commit> <upstream-commit> <message-file>
# Always two parents: the base branch, and the upstream tip.  The tree is the
# policy-filtered synchronization result; the second parent is an assertion that
# every change in `<accepted baseline>..<upstream-commit>` has been accounted for.
# Callers must therefore only reach this after the whole range was classified with
# no blocker left (see fs_build_candidate_tree).
fs_commit_tree() {
  git commit-tree "$1" -p "$2" -p "$3" -F "$4"
}

# fs_automation_owned <commit> <base-ref>
# True when <commit> is one this tooling wrote and may therefore be replaced.
#
# The shape is checked, not just a marker string: a candidate has exactly two
# parents, trailers naming those parents, a base that is on our line of history,
# and a lock file inside it that records the same upstream revision as its second
# parent.  A human commit -- or a forged one that merely copies the trailers --
# cannot satisfy that, so the rolling branch is left alone instead of overwritten.
fs_automation_owned() {
  local commit=$1 base=$2 body parents first second rec_base rec_up rec_lock

  body=$(git log -1 --format=%B "$commit" 2>/dev/null || true)
  case $body in
    *'Fork-Sync-Bot: sync-upstream.sh'*) ;;
    *) return 1 ;;
  esac
  case $body in
    *'Fork-Sync-Automated: true'*) ;;
    *) return 1 ;;
  esac

  parents=$(git rev-list --parents -n 1 "$commit" | wc -w)
  [ "$parents" -eq 3 ] || return 1        # exactly two parents
  first=$(git rev-parse "$commit^1")
  second=$(git rev-parse "$commit^2")

  rec_base=$(printf '%s\n' "$body" | sed -n 's/^Fork-Sync-Base: //p' | head -1)
  rec_up=$(printf '%s\n' "$body" | sed -n 's/^Fork-Sync-Upstream: //p' | head -1)
  [ -n "$rec_base" ] && [ "$rec_base" = "$first" ] || return 1
  [ -n "$rec_up" ] && [ "$rec_up" = "$second" ] || return 1

  # generated from our line of history
  git merge-base --is-ancestor "$first" "$base" 2>/dev/null || return 1

  # content and ancestry must agree about what was accepted
  rec_lock=$(git show "$commit:.fork/upstream.lock" 2>/dev/null \
             | sed -n 's/^commit[[:space:]]*=[[:space:]]*//p' | head -1)
  [ -n "$rec_lock" ] && [ "$rec_lock" = "$second" ]
}

# --------------------------------------------------------------- reporting --

# fs_report <line...> -- append markdown to the report file, if one was asked for.
fs_report() {
  [ -n "${FS_REPORT_PATH:-}" ] || return 0
  printf '%s\n' "$*" >> "$FS_REPORT_PATH"
}

# fs_report_stdin -- one report line per input line.
fs_report_stdin() {
  local line
  while IFS= read -r line; do
    fs_report "$line"
  done
}
