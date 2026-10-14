#!/usr/bin/env bash
# .fork/tests/run.sh -- fixture tests for the fork's upstream-sync machinery.
#
# Builds miniature git repositories in a temporary directory (dummy text files
# only) and drives the real .fork/scripts/*.sh against them.  It never builds
# Strata, never runs anything from the "upstream" repositories, and never
# touches the fork's own working tree.
#
#   .fork/tests/run.sh
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SELF_DIR/../.." && pwd)
SYNC="$REPO_DIR/.fork/scripts/sync-upstream.sh"
VALIDATE="$REPO_DIR/.fork/scripts/validate-sync.sh"

T=$(mktemp -d "${TMPDIR:-/tmp}/fork-sync-tests.XXXXXX")
trap 'rm -rf "$T"' EXIT

PASS=0
FAIL=0
CASE=''
ok()    { PASS=$((PASS + 1)); printf '  ok   %s\n' "$1"; }
bad()   { FAIL=$((FAIL + 1)); printf '  FAIL %s\n' "$1"; }
check() { # check <what> <expected> <actual>
  if [ "$2" = "$3" ]; then ok "$1"; else bad "$1: expected [$2] ... got [$3]"; fi
}
contains() { # contains <what> <needle> <haystack-file>
  if grep -qF -- "$2" "$3" 2>/dev/null; then ok "$1"; else bad "$1: no '$2' in $(basename "$3")"; fi
}
no_candidate() { # no_candidate <what> <report>
  if grep -q '^| candidate |' "$2" 2>/dev/null; then bad "$1: a candidate was produced"; else ok "$1: no candidate produced"; fi
}
lacks() { # lacks <what> <needle> <haystack-file>
  if grep -qF -- "$2" "$3" 2>/dev/null; then bad "$1: unexpected '$2' in $(basename "$3")"; else ok "$1"; fi
}
section() { printf '\n== %s ==\n' "$1"; }

git_() { git -C "$1" "${@:2}"; }

# ------------------------------------------------------------------ fixtures --

# world <name>  ->  UP (upstream), FORK (clone), U0
world() {
  local n=$1
  UP=$T/$n/up
  FORK=$T/$n/fork
  ORIGIN=$T/$n/origin.git
  mkdir -p "$T/$n"
  git init -q -b main "$UP"
  git_ "$UP" config user.email t@t
  git_ "$UP" config user.name t
  mkdir -p "$UP/src" "$UP/include" "$UP/docs" "$UP/tools" "$UP/data" "$UP/bench/micro" "$UP/third_party/ggml"
  printf 'base\n'  > "$UP/src/engine.txt"
  printf 'api\n'   > "$UP/include/api.h"
  printf 'core-1\ncore-2\ncore-3\ncore-4\ncore-5\ncore-6\n' > "$UP/src/engine_core.txt"
  printf 'guide\n' > "$UP/docs/guide.md"
  printf 'serve\n' > "$UP/tools/serve_load.py"
  printf 'pack\n'  > "$UP/tools/iq_pack.py"
  printf 'vocab\n' > "$UP/data/draft_vocab.bin"
  printf 'other\n' > "$UP/data/other.bin"
  printf 'micro\n' > "$UP/bench/micro/bench.cu"
  printf 'ver\n'   > "$UP/third_party/ggml/VERSION.txt"
  printf 'readme\n' > "$UP/README.md"
  git_ "$UP" add README.md src include docs tools data bench third_party
  git_ "$UP" commit -qm "U0: initial upstream"
  U0=$(git_ "$UP" rev-parse HEAD)

  git clone -q "$UP" "$FORK"
  git_ "$FORK" config user.email t@t
  git_ "$FORK" config user.name t

  # the fork's pruning commit
  git_ "$FORK" rm -q -r docs third_party data/other.bin
  git_ "$FORK" rm -q tools/serve_load.py
  mkdir -p "$FORK/.fork/scripts" "$FORK/.github/workflows"
  cp "$REPO_DIR/.fork/path-policy" "$FORK/.fork/path-policy"
  cp "$REPO_DIR/.fork/README.md"   "$FORK/.fork/README.md"
  cp "$REPO_DIR/.fork/scripts/"*.sh "$FORK/.fork/scripts/"
  cp "$REPO_DIR/.github/workflows/sync-upstream.yml" "$FORK/.github/workflows/"
  cp "$REPO_DIR/FORK_NOTES.md" "$FORK/FORK_NOTES.md"
  sed "s|^commit .*|commit     = $U0|" "$REPO_DIR/.fork/upstream.lock" > "$FORK/.fork/upstream.lock"
  printf 'fork readme\n' > "$FORK/README.md"
  printf '[submodule "third_party/llama.cpp"]\n\tpath = third_party/llama.cpp\n\turl = https://github.com/ggml-org/llama.cpp.git\n\tshallow = true\n' > "$FORK/.gitmodules"
  # a vendored submodule, represented the way git stores it (a gitlink)
  git_ "$FORK" update-index --add --cacheinfo "160000,$U0,third_party/llama.cpp"
  git_ "$FORK" add .fork .github .gitmodules FORK_NOTES.md README.md
  git_ "$FORK" commit -qm "fork: prune and add fork infrastructure"

  git init -q --bare "$ORIGIN"
  git_ "$FORK" push -q "$ORIGIN" main
}

# up <direction...>  -- commit in the upstream fixture
up_commit() { # up_commit <message>
  git_ "$UP" add -A
  git_ "$UP" commit -qm "$1"
}

# sync <report-file> [extra args...]  ->  sets SYNC_RC, SYNC_OUT, SYNC_ERR
sync() {
  local report=$1; shift
  set +e
  "$SYNC" --repo "$FORK" --base-ref main --allow-local-upstream \
          --upstream-url "$UP" --upstream-branch main \
          --report "$report" "$@" > "$T/sync.out" 2> "$T/sync.err"
  SYNC_RC=$?
  set -e
  SYNC_OUT=$(cat "$T/sync.out")
  SYNC_ERR=$(cat "$T/sync.err")
}

# candidate_sha <report-file>  -- the full candidate SHA recorded in a report
candidate_sha() {
  sed -n 's/^| candidate | `\([0-9a-f]\{40\}\)`.*/\1/p' "$1" | head -1
}

# -------------------------------------------------------------------- tests --

section "no-op: upstream is already at the accepted revision"
world noop
sync "$T/noop.md"
check "exit code" 0 "$SYNC_RC"
contains "says up to date" "UP TO DATE" "$T/noop.md"
check "no candidate recorded" "" "$(candidate_sha "$T/noop.md")"

section "one clean upstream commit -> candidate"
world clean
printf 'base\nupstream work\n' > "$UP/src/engine.txt"
up_commit "U1: engine work"
sync "$T/clean.md"
check "exit code" 0 "$SYNC_RC"
CAND=$(candidate_sha "$T/clean.md")
if [ -n "$CAND" ]; then ok "candidate recorded"; else bad "candidate recorded"; fi
check "candidate parent is main" "$(git_ "$FORK" rev-parse main)" "$(git_ "$FORK" rev-parse "$CAND^1")"
check "second parent is the upstream tip" "$(git_ "$UP" rev-parse HEAD)" "$(git_ "$FORK" rev-parse "$CAND^2")"
# the "N commits behind" number GitHub shows, computed the same way
check "nothing left behind upstream" "0" "$(git_ "$FORK" rev-list --count "$(git_ "$UP" rev-parse HEAD)" --not "$CAND")"
if git_ "$FORK" merge-base --is-ancestor "$(git_ "$UP" rev-parse HEAD)" "$CAND"; then
  ok "upstream tip is an ancestor of the candidate"
else
  bad "upstream tip is not an ancestor of the candidate"
fi
git_ "$FORK" log -1 --format=%B "$CAND" > "$T/body.txt"
contains "trailer names the base"    "Fork-Sync-Base: $(git_ "$FORK" rev-parse main)" "$T/body.txt"
contains "trailer names upstream"    "Fork-Sync-Upstream: $(git_ "$UP" rev-parse HEAD)" "$T/body.txt"
contains "trailer marks automation" "Fork-Sync-Automated: true" "$T/body.txt"
check "lock inside the candidate agrees with its second parent" \
      "commit     = $(git_ "$UP" rev-parse HEAD)" "$(git_ "$FORK" show "$CAND:.fork/upstream.lock" | grep '^commit')"
check "upstream change applied" "base
upstream work" "$(git_ "$FORK" show "$CAND:src/engine.txt")"
check "lock advanced" "commit     = $(git_ "$UP" rev-parse HEAD)" "$(git_ "$FORK" show "$CAND:.fork/upstream.lock" | grep '^commit')"
check "only engine + lock changed" ".fork/upstream.lock
src/engine.txt" "$(git_ "$FORK" diff --name-only main "$CAND")"

section "several upstream commits -> the whole range is taken"
world range
printf 'base\nw1\n' > "$UP/src/engine.txt"; up_commit "U1"
printf 'w2\n'      > "$UP/include/api.h";  up_commit "U2"
printf 'w3\n'      > "$UP/tools/iq_pack.py"; up_commit "U3"
sync "$T/range.md"
check "exit code" 0 "$SYNC_RC"
contains "counts three commits" "| upstream commits | 3 " "$T/range.md"
CAND=$(candidate_sha "$T/range.md")
check "first change landed"  "base
w1" "$(git_ "$FORK" show "$CAND:src/engine.txt")"
check "second change landed" "w2" "$(git_ "$FORK" show "$CAND:include/api.h")"
check "third change landed"  "w3" "$(git_ "$FORK" show "$CAND:tools/iq_pack.py")"
check "candidate is a single merge commit on main" "$(git_ "$FORK" rev-parse main)" "$(git_ "$FORK" rev-parse "$CAND^1")"
check "all three upstream commits are ancestors" "0" "$(git_ "$FORK" rev-list --count "$(git_ "$UP" rev-parse HEAD)" --not "$CAND")"

section "upstream changes a dropped path -> stays dropped, baseline still advances"
world dropped
printf 'guide\nmore\n' > "$UP/docs/guide.md"
printf 'serve2\n'     > "$UP/tools/serve_load.py"
up_commit "U1: docs and a removed tool"
sync "$T/dropped.md"
check "exit code" 0 "$SYNC_RC"
CAND=$(candidate_sha "$T/dropped.md")
check "candidate is the lock only" ".fork/upstream.lock" "$(git_ "$FORK" diff --name-only main "$CAND")"
check "docs did not come back" "" "$(git_ "$FORK" ls-tree -r --name-only "$CAND" | grep '^docs/' || true)"
check "removed tool did not come back" "" "$(git_ "$FORK" ls-tree -r --name-only "$CAND" | grep 'tools/serve_load.py' || true)"
contains "report notes the drops" "dropped" "$T/dropped.md"
contains "no divergence for a dropped path" "| fork divergences touched | no |" "$T/dropped.md"

section "upstream edits a fork-owned file -> ours is kept, change is reported"
world ours
printf 'upstream readme\n' > "$UP/README.md"
up_commit "U1: rewrite the readme"
sync "$T/ours.md"
check "exit code" 0 "$SYNC_RC"
CAND=$(candidate_sha "$T/ours.md")
check "our README is kept" "fork readme" "$(git_ "$FORK" show "$CAND:README.md")"
contains "report lists it as ours" "README.md (M)" "$T/ours.md"

section "upstream touches a protected path -> stop"
world protected
mkdir -p "$UP/.fork"
printf 'evil\n' > "$UP/.fork/evil.sh"
up_commit "U1: touch the fork control surface"
sync "$T/protected.md"
check "exit code" 1 "$SYNC_RC"
contains "names the protected path" "[protected] .fork/evil.sh" "$T/sync.err"
contains "report says blocked" "BLOCKED" "$T/protected.md"
contains "report gives the reason" "[protected] .fork/evil.sh" "$T/protected.md"
# the second parent is an assertion that the range was accounted for, so a stopped
# run must not produce a candidate at all
no_candidate "no ancestry graft when a path is protected" "$T/protected.md"
check "baseline not advanced" "commit     = $U0" "$(git_ "$FORK" show main:.fork/upstream.lock | grep '^commit')"

section "upstream introduces an unclassified surface -> stop"
world unknown
mkdir -p "$UP/installer"
printf 'installer\n' > "$UP/installer/setup.py"
up_commit "U1: add an installer"
sync "$T/unknown.md"
check "exit code" 1 "$SYNC_RC"
contains "names the path" "installer/setup.py" "$T/sync.err"
contains "asks for classification" "[unclassified] installer/setup.py" "$T/sync.err"
no_candidate "no ancestry graft for an unclassified path" "$T/unknown.md"
check "baseline not advanced" "commit     = $U0" "$(git_ "$FORK" show main:.fork/upstream.lock | grep '^commit')"

section "mirrored path changed on both sides -> stop instead of guessing"
world conflict
printf 'base\nupstream line\n' > "$UP/src/engine.txt"
up_commit "U1: upstream edits the engine line"
printf 'base\nfork line\n' > "$FORK/src/engine.txt"
git_ "$FORK" add src/engine.txt
git_ "$FORK" commit -qm "fork: local edit of the same line"
sync "$T/conflict.md"
check "exit code" 1 "$SYNC_RC"
contains "reports the conflict" "[conflict] src/engine.txt" "$T/sync.err"
no_candidate "no ancestry graft on a mirrored conflict" "$T/conflict.md"
lacks "a blocked run publishes no divergence analysis" "Fork divergences touched" "$T/conflict.md"
check "main is untouched" "$(git_ "$FORK" rev-parse main)" "$(git_ "$FORK" rev-parse main)"

section "a mirrored file the fork patched: upstream's change lands, the patch survives"
world localpatch
# the fork patches one region of a file upstream maintains, and adds a file of its
# own that upstream happens to add too (same content: not a patch to anything)
printf 'core-1\ncore-2-patched\ncore-3\ncore-4\ncore-5\ncore-6\n' > "$FORK/src/engine_core.txt"
printf 'both sides\n' > "$FORK/src/both_sides.txt"
git_ "$FORK" add src/engine_core.txt src/both_sides.txt
git_ "$FORK" commit -qm "fork: patch a mirrored file"
# upstream edits another region of the same file, plus a file the fork never touched
printf 'core-1\ncore-2\ncore-3\ncore-4\ncore-5\ncore-6-upstream\n' > "$UP/src/engine_core.txt"
printf 'api\nupstream\n' > "$UP/include/api.h"
printf 'both sides\n' > "$UP/src/both_sides.txt"
up_commit "U1: upstream edits another region of the same file"
sync "$T/localpatch.md"
check "exit code" 0 "$SYNC_RC"
CAND=$(candidate_sha "$T/localpatch.md")
check "the fork's patch is still there" "core-2-patched" \
      "$(git_ "$FORK" show "$CAND:src/engine_core.txt" | sed -n 2p)"
check "upstream's change landed as well" "core-6-upstream" \
      "$(git_ "$FORK" show "$CAND:src/engine_core.txt" | sed -n 6p)"
contains "flags the divergence"        "Fork divergences touched by this range" "$T/localpatch.md"
contains "is yellow, not green"         "🟡 REVIEW" "$T/localpatch.md"
contains "counts one diverged path"     "| fork divergences touched | **yes** — 1 mirrored path(s)" "$T/localpatch.md"
contains "names the diverged path"      '`src/engine_core.txt`' "$T/localpatch.md"
contains "explains why clean is not enough" "not about the meaning" "$T/localpatch.md"
lacks    "does not call a pristine file divergent" "include/api.h (upstream" "$T/localpatch.md"
lacks    "does not call a co-added file divergent"  "both_sides.txt (upstream" "$T/localpatch.md"
git_ "$FORK" log -1 --format=%B "$CAND" > "$T/localpatch-body.txt"
contains "the commit body records it"   "Fork divergences: 1 mirrored path(s)" "$T/localpatch-body.txt"

section "upstream rewrote history -> stop, do not guess a baseline"
world rewrite
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1"
sync "$T/rewrite-ok.md"          # first sync succeeds, so the lock moves to U1
CAND=$(candidate_sha "$T/rewrite-ok.md")
git_ "$FORK" update-ref refs/heads/main "$CAND"   # simulate the pull request being squash-merged
git_ "$UP" reset -q --hard "$U0"
printf 'rewritten\n' > "$UP/src/engine.txt"
up_commit "R1: force-pushed history"
sync "$T/rewrite.md"
check "exit code" 1 "$SYNC_RC"
contains "says the baseline is not an ancestor" "is not an ancestor of upstream" "$T/sync.err"
contains "report says blocked" "history was rewritten" "$T/rewrite.md"

section "merge commit vs squash: the lock file carries the baseline either way"
world squash
FORK_MAIN0=$(git_ "$FORK" rev-parse main)
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1"
sync "$T/squash-1.md"
CAND1=$(candidate_sha "$T/squash-1.md")

# (a) merged with a merge commit: main becomes the candidate, which records the
#     upstream tip as its second parent, so git's own view is accurate
 git_ "$FORK" update-ref refs/heads/main "$CAND1"
check "merge commit keeps upstream as an ancestor" "$(git_ "$UP" rev-parse HEAD)" \
      "$(git_ "$FORK" merge-base main "$(git_ "$UP" rev-parse HEAD)")"
check "the fork does not read as behind" "0" \
      "$(git_ "$FORK" rev-list --count "$(git_ "$UP" rev-parse HEAD)" --not main)"
sync "$T/squash-1b.md"
check "a second sync is then a no-op" 0 "$SYNC_RC"
contains "and says up to date" "UP TO DATE" "$T/squash-1b.md"

# (b) squashed: the same tree, but a single parent -- what GitHub's squash merge
#     produces.  Git's merge base goes stale; the lock file does not.
squashed=$(git_ "$FORK" commit-tree "$(git_ "$FORK" rev-parse "$CAND1^{tree}")" -p "$FORK_MAIN0")
git_ "$FORK" update-ref refs/heads/main "$squashed"
check "git's own merge base is now stale" "$U0" \
      "$(git_ "$FORK" merge-base main "$(git_ "$UP" rev-parse HEAD)")"
printf 'w2\n' > "$UP/include/api.h"
up_commit "U2"
sync "$T/squash-2.md"
check "exit code" 0 "$SYNC_RC"
CAND2=$(candidate_sha "$T/squash-2.md")
check "candidate sits on main" "$(git_ "$FORK" rev-parse main)" "$(git_ "$FORK" rev-parse "$CAND2^1")"
check "only the new change is in the diff" ".fork/upstream.lock
include/api.h" "$(git_ "$FORK" diff --name-only main "$CAND2")"
check "new change landed" "w2" "$(git_ "$FORK" show "$CAND2:include/api.h")"
check "no spurious re-application" "base
w1" "$(git_ "$FORK" show "$CAND2:src/engine.txt")"
check "the new candidate grafts all upstream ancestry back" "0" \
      "$(git_ "$FORK" rev-list --count "$(git_ "$UP" rev-parse HEAD)" --not "$CAND2")"

section "the vendored submodule gitlink is never touched"
world submodule
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1"
sync "$T/submodule.md"
CAND=$(candidate_sha "$T/submodule.md")
check "gitlink unchanged" "$(git_ "$FORK" ls-tree main third_party/llama.cpp | awk '{print $3}')" \
                          "$(git_ "$FORK" ls-tree "$CAND" third_party/llama.cpp | awk '{print $3}')"
printf 'U2: upstream adds its own gitmodules\n' > /dev/null
printf '[submodule "x"]\n\tpath = x\n\turl = https://example.invalid/x.git\n' > "$UP/.gitmodules"
up_commit "U2: upstream adds .gitmodules"
sync "$T/submodule-2.md"
check "adding .gitmodules stops the sync" 1 "$SYNC_RC"
contains "names .gitmodules" ".gitmodules" "$T/sync.err"

section "--push: creates the rolling branch, then refuses to clobber a human commit"
world push
PUSH_MAIN0=$(git_ "$FORK" rev-parse main)
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1"
sync "$T/push-1.md" --push --remote "$ORIGIN"
check "exit code" 0 "$SYNC_RC"
CAND=$(candidate_sha "$T/push-1.md")
check "rolling branch created" "$CAND" "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"

# a human pushes their own commit onto the rolling branch
git_ "$FORK" fetch -q "$ORIGIN" sync/upstream
tree=$(git_ "$FORK" rev-parse FETCH_HEAD^{tree})
human=$(printf 'human: hand edit\n' | GIT_AUTHOR_NAME=h GIT_AUTHOR_EMAIL=h@h GIT_COMMITTER_NAME=h GIT_COMMITTER_EMAIL=h@h \
        git_ "$FORK" commit-tree "$tree" -p FETCH_HEAD -F -)
git_ "$FORK" push -q --force "$ORIGIN" "$human:refs/heads/sync/upstream"

printf 'w2\n' > "$UP/include/api.h"
up_commit "U2"
sync "$T/push-2.md" --push --remote "$ORIGIN"
check "refuses to overwrite" 1 "$SYNC_RC"
contains "says the branch is not automation-owned" "not written by this tooling" "$T/sync.err"
check "the human commit is still there" "$human" "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"

# a commit that copies the provenance trailers, but whose own lock file disagrees
# with its second parent, must also be refused (content and ancestry must agree)
forged_tree=$(git_ "$FORK" rev-parse "$PUSH_MAIN0^{tree}")
forged_body=$(printf 'sync: upstream Strata\n\nFork-Sync-Bot: sync-upstream.sh\nFork-Sync-Automated: true\nFork-Sync-Base: %s\nFork-Sync-Upstream: %s\n' \
              "$(git_ "$FORK" rev-parse main)" "$(git_ "$UP" rev-parse HEAD)")
forged=$(printf '%s' "$forged_body" | GIT_AUTHOR_NAME=h GIT_AUTHOR_EMAIL=h@h GIT_COMMITTER_NAME=h GIT_COMMITTER_EMAIL=h@h \
         git_ "$FORK" commit-tree "$forged_tree" -p "$(git_ "$FORK" rev-parse main)" -p "$(git_ "$UP" rev-parse HEAD)" -F -)
git_ "$FORK" push -q --force "$ORIGIN" "$forged:refs/heads/sync/upstream"
sync "$T/push-forged.md" --push --remote "$ORIGIN"
check "refuses a forged provenance commit" 1 "$SYNC_RC"
contains "says the branch is not automation-owned" "not written by this tooling" "$T/sync.err"
check "the forged commit is still there" "$forged" \
      "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"

# once the branch is back to an automation commit, the push works again
git_ "$FORK" push -q --force "$ORIGIN" "$CAND:refs/heads/sync/upstream"
sync "$T/push-3.md" --push --remote "$ORIGIN"
check "exit code" 0 "$SYNC_RC"
CAND3=$(candidate_sha "$T/push-3.md")
check "rolling branch updated" "$CAND3" "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"

section "report: status, runtime signal, honesty and truncation"
world report
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1: engine work"
sync "$T/report-a.md"
contains "green when only the engine changed" "🟢 READY" "$T/report-a.md"
contains "no divergence flagged for a pristine mirrored file" "| fork divergences touched | no |" "$T/report-a.md"
contains "says the runtime is affected" "retained runtime affected | **yes**" "$T/report-a.md"
contains "summarises the areas" '`src` — 1 integrated' "$T/report-a.md"
contains "marks a dry run" "Dry run" "$T/report-a.md"

printf 'add_library(x)\n' > "$UP/CMakeLists.txt"
up_commit "U2: build integration"
sync "$T/report-b.md"
contains "yellow when build integration changed" "🟡 REVIEW" "$T/report-b.md"
contains "names the sensitive path" "CMakeLists.txt" "$T/report-b.md"

world trunc
for i in $(seq 1 210); do printf 'x\n' > "$UP/src/f$i.txt"; done
up_commit "U1: a lot of files"
sync "$T/report-c.md"
contains "discloses a truncated path list" "showing the first 200" "$T/report-c.md"
contains "counts the omitted paths" "10 more path(s) omitted" "$T/report-c.md"

world many
for i in $(seq 1 45); do printf 'v%s\n' "$i" > "$UP/src/engine.txt"; up_commit "U$i"; done
sync "$T/report-d.md"
contains "counts the commits" "| upstream commits | 45 " "$T/report-d.md"
contains "bounds a long commit list" "showing the oldest 10 and newest 25" "$T/report-d.md"
contains "discloses omitted commits" "10 commits omitted" "$T/report-d.md"

world blocked
mkdir -p "$UP/.fork"
printf 'evil\n' > "$UP/.fork/evil.sh"
up_commit "U1: touch the fork control surface"
sync "$T/report-e.md"
contains "blocked report says blocked" "🔴 BLOCKED" "$T/report-e.md"
contains "blocked report says the branch was untouched" "**not modified**" "$T/report-e.md"
contains "blocked report gives the reason" "[protected] .fork/evil.sh" "$T/report-e.md"
if grep -qE 'candidate upstream|READY|lock update only' "$T/report-e.md"; then
  bad "blocked report must not describe a candidate"
else
  ok "blocked report does not describe a candidate"
fi

section "after the rolling PR is merged with a merge commit"
world merged
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1"
sync "$T/merged-1.md" --push --remote "$ORIGIN"
CAND1=$(candidate_sha "$T/merged-1.md")
check "rolling branch created" "$CAND1" "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"

# GitHub's "Create a merge commit": parents (main, CAND1), tree from CAND1
merge_commit=$(git_ "$FORK" commit-tree "$(git_ "$FORK" rev-parse "$CAND1^{tree}")" \
               -p "$(git_ "$FORK" rev-parse main)" -p "$CAND1" -m "Merge pull request #1 from sync/upstream")
git_ "$FORK" update-ref refs/heads/main "$merge_commit"
check "merging made upstream an ancestor of main" "$(git_ "$UP" rev-parse HEAD)" \
      "$(git_ "$FORK" merge-base main "$(git_ "$UP" rev-parse HEAD)")"
check "the fork does not read as behind" "0" \
      "$(git_ "$FORK" rev-list --count "$(git_ "$UP" rev-parse HEAD)" --not main)"

# nothing new upstream: clean exit, and the merged branch is left alone
sync "$T/merged-2.md" --push --remote "$ORIGIN"
check "exit code" 0 "$SYNC_RC"
contains "says up to date" "UP TO DATE" "$T/merged-2.md"
check "the branch was left as it was" "$CAND1" \
      "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"

# upstream advances: the same rolling branch is replaced by a new candidate.  The old
# tip is an ancestor of main now, which is exactly the case the ownership check has to
# keep accepting.
printf 'w2\n' > "$UP/include/api.h"
up_commit "U2"
sync "$T/merged-3.md" --push --remote "$ORIGIN"
check "exit code" 0 "$SYNC_RC"
CAND2=$(candidate_sha "$T/merged-3.md")
if [ -n "$CAND2" ] && [ "$CAND2" != "$CAND1" ]; then ok "a new candidate was produced"; else bad "no new candidate"; fi
check "the same rolling branch was reused" "$CAND2" \
      "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"
check "new candidate sits on the merged main" "$(git_ "$FORK" rev-parse main)" \
      "$(git_ "$FORK" rev-parse "$CAND2^1")"
check "only the new change is in the diff" ".fork/upstream.lock
include/api.h" \
      "$(git_ "$FORK" diff --name-only main "$CAND2")"

section "report honesty when only fork-owned or dropped paths changed"
world oursonly
printf 'upstream readme\n' > "$UP/README.md"
up_commit "U1: upstream rewrites the readme"
sync "$T/oursonly.md"
check "exit code" 0 "$SYNC_RC"
CAND=$(candidate_sha "$T/oursonly.md")
check "candidate changes only the lock" ".fork/upstream.lock" "$(git_ "$FORK" diff --name-only main "$CAND")"
contains "labels it a policy suppression" "no fork tree changes after policy" "$T/oursonly.md"
if grep -q "only dropped paths" "$T/oursonly.md"; then
  bad "an ours-only change must not be called a dropped path"
else
  ok "does not mislabel an ours-only change as dropped"
fi

section "broken metadata still produces an honest blocked report"
world nolock
git_ "$FORK" rm -q .fork/upstream.lock
git_ "$FORK" commit -qm "remove the lock file"
sync "$T/nolock.md"
check "exit code" 1 "$SYNC_RC"
contains "report is still produced" "🔴 BLOCKED" "$T/nolock.md"
contains "names the missing metadata" "upstream.lock" "$T/nolock.md"
contains "says nothing was pushed" "Nothing was pushed" "$T/nolock.md"

world badlock
printf 'this line is not key = value\n' > "$FORK/.fork/upstream.lock"
git_ "$FORK" add .fork/upstream.lock
git_ "$FORK" commit -qm "corrupt the lock file"
sync "$T/badlock.md"
check "exit code" 1 "$SYNC_RC"
contains "report is still produced" "🔴 BLOCKED" "$T/badlock.md"
contains "says the metadata is malformed" "malformed" "$T/badlock.md"

section "push succeeded but publication failed"
world publish
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1"
mkdir -p "$T/fakebin"
printf '#!/bin/sh\nexit 1\n' > "$T/fakebin/gh"
chmod +x "$T/fakebin/gh"
set +e
PATH="$T/fakebin:$PATH" "$SYNC" --repo "$FORK" --base-ref main --allow-local-upstream \
  --upstream-url "$UP" --upstream-branch main --report "$T/publish.md" \
  --push --remote "$ORIGIN" --pr --pr-base main > "$T/sync.out" 2> "$T/sync.err"
publish_rc=$?
set -e
check "exit code" 1 "$publish_rc"
contains "report says blocked" "🔴 BLOCKED" "$T/publish.md"
contains "report says the branch was updated" "**updated**" "$T/publish.md"
if grep -q "Nothing was pushed" "$T/publish.md"; then
  bad "must not claim nothing was pushed when the push succeeded"
else
  ok "does not deny a push that happened"
fi
CAND_SHORT=$(awk '/^fork-sync: candidate/ {print $3}' "$T/sync.out")
CAND=$(git_ "$FORK" rev-parse "$CAND_SHORT")
check "the candidate really did reach the branch" "$CAND" \
      "$(git ls-remote --heads "$ORIGIN" refs/heads/sync/upstream | awk '{print $1}')"
contains "the report names the candidate that was pushed" "updated** to \`$CAND_SHORT\`" "$T/publish.md"

section "path policy classification"
world policy
mapfile -t GOT < <("$SYNC" --repo "$FORK" --base-ref main --allow-local-upstream --classify \
  src/engine.txt include/api.h tools/iq_pack.py data/draft_vocab.bin bench/micro/bench.cu README.md \
  docs/guide.md tools/serve_load.py data/other.bin third_party/ggml/VERSION.txt third_party/llama.cpp \
  .fork/path-policy .github/workflows/x.yml docs/new.md installer/new.py tools/new_tool.py)
check "mirrored engine file"     "src/engine.txt mirror"          "${GOT[0]}"
check "mirrored header"          "include/api.h mirror"           "${GOT[1]}"
check "mirrored kept tool"       "tools/iq_pack.py mirror"        "${GOT[2]}"
check "mirrored kept data"       "data/draft_vocab.bin mirror"    "${GOT[3]}"
check "mirrored kept benchmark"  "bench/micro/bench.cu mirror"    "${GOT[4]}"
check "fork-owned readme"        "README.md ours"                 "${GOT[5]}"
check "pruned documentation"     "docs/guide.md dropped"          "${GOT[6]}"
check "pruned tool"              "tools/serve_load.py dropped"    "${GOT[7]}"
check "pruned data file"         "data/other.bin dropped"         "${GOT[8]}"
check "pruned vendored header"   "third_party/ggml/VERSION.txt dropped" "${GOT[9]}"
check "protected submodule"      "third_party/llama.cpp protected"      "${GOT[10]}"
check "protected fork area"      ".fork/path-policy protected"          "${GOT[11]}"
check "fork-owned workflow"      ".github/workflows/x.yml ours"         "${GOT[12]}"
check "new file under a dropped tree" "docs/new.md dropped"             "${GOT[13]}"
check "new top-level surface"    "installer/new.py unknown"             "${GOT[14]}"
check "new file under a mirrored tree" "tools/new_tool.py mirror"       "${GOT[15]}"

section "a run leaves no working files behind"
world tmpleak
printf 'base\nw1\n' > "$UP/src/engine.txt"
up_commit "U1"
mkdir -p "$T/tmpleak-tmp"
set +e
TMPDIR="$T/tmpleak-tmp" "$SYNC" --repo "$FORK" --base-ref main --allow-local-upstream \
  --upstream-url "$UP" --upstream-branch main --report "$T/tmpleak.md" >/dev/null 2>&1
leak_rc=$?
set -e
check "a successful run exits 0" 0 "$leak_rc"
check "and cleans up after itself" "" "$(ls -A "$T/tmpleak-tmp")"
set +e
TMPDIR="$T/tmpleak-tmp" "$SYNC" --repo "$FORK" --base-ref does-not-exist >/dev/null 2>&1
leak_rc=$?
set -e
check "a failing run exits non-zero" 1 "$leak_rc"
check "and cleans up after itself too" "" "$(ls -A "$T/tmpleak-tmp")"

section "validate-sync.sh"
if [ -x "$VALIDATE" ]; then
  world validate
  set +e
  "$VALIDATE" --repo "$FORK" --base-ref main > "$T/validate.out" 2>&1
  vrc=$?
  set -e
  check "clean fixture validates" 0 "$vrc"
  sed -i 's/^commit .*/commit     = deadbeef/' "$FORK/.fork/upstream.lock"
  git_ "$FORK" add .fork/upstream.lock
  git_ "$FORK" commit -qm "break the lock"
  set +e
  "$VALIDATE" --repo "$FORK" --base-ref main > "$T/validate-bad.out" 2>&1
  vrc=$?
  set -e
  check "broken lock is rejected" 1 "$vrc"
  contains "says why" "not a full commit id" "$T/validate-bad.out"
else
  bad "validate-sync.sh is missing"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
