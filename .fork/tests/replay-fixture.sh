#!/usr/bin/env bash
# .fork/tests/replay-fixture.sh -- fixture test for fork-replay.sh: miniature
# repositories with dummy text files only, exercising every mechanic (mirror
# apply, deletion, drop/ours/protected filtering, unclassified exclusion,
# conflict markers, byte-exact pristine replay).  No network, no engine.
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$SELF_DIR/../scripts/fork-replay.sh
LIBSRC=$SELF_DIR/../scripts/lib.sh
POLICY=$SELF_DIR/../path-policy

ROOT=${TMPDIR:-/tmp}/fork-replay-fixture.$$
FX=$ROOT/fixture
mkdir -p "$FX"
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t

cleanup() { cd /; rm -rf -- "$ROOT"; }
trap cleanup EXIT

# ---------------------------------------------------------- upstream -----
git init -q --initial-branch=main "$FX/upstream"
cd "$FX/upstream"
mkdir -p src/core src/kernels include cmake docs serve tools data .github
printf 'int a(void) { return 0; }\n'            > src/core/a.c
printf '__global__ void k(void) {}\n'           > src/kernels/k.cu
printf '#pragma once\n#define H 1\n'            > include/h.hpp
printf 'add_subdirectory(src)\n'                > cmake/helper.cmake
cat > CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(strata LANGUAGES C CXX)
include(FetchContent)
FetchContent_Declare(llama GIT_REPOSITORY https://example/llama.git GIT_TAG 3cf0325)
add_subdirectory(src)
EOF
printf 'docs\n'  > docs/d.md
printf 'serve\n' > serve/s.py
printf 'tool\n'  > tools/t.py
printf 'BLOB\n'  > data/draft_vocab.bin
printf 'ci\n'    > .github/w.yml
printf 'readme\n' > README.md
git add -A && git commit -qm 'upstream initial'
U0=$(git rev-parse HEAD)

# ------------------------------------------------------------- fork ------
git clone -q "$FX/upstream" "$FX/fork"
cd "$FX/fork"
git checkout -q main
printf 'int a(void) { return 1; }  // their fix\n' > src/core/a.c
printf '__global__ void their_kernel(void) {}\n'   > src/kernels/new.cu
printf '#pragma once\n#define H 2\n'               > include/h.hpp
rm tools/t.py
printf 'docs changed\n' > docs/d.md
printf 'serve changed\n' > serve/s.py
printf 'extra data\n' > data/extra.bin
mkdir newdir
printf 'newdir\n' > newdir/n.txt
printf 'ci changed\n' > .github/w.yml
printf 'readme changed\n' > README.md
sed -i 's/GIT_TAG 3cf0325/GIT_TAG deadbeef/' CMakeLists.txt
git add -A && git commit -qm 'their work'

# ------------------------------------------------------------- ours ------
git clone -q "$FX/upstream" "$FX/ours"
cd "$FX/ours"
git checkout -q main
git rm -rq docs serve
python3 - <<'EOF'
s = open('CMakeLists.txt').read()
s = s.replace("FetchContent_Declare(llama GIT_REPOSITORY https://example/llama.git GIT_TAG 3cf0325)",
              "add_subdirectory(third_party/llama.cpp)  # vendored")
open('CMakeLists.txt','w').write(s)
EOF
mkdir -p .fork/scripts
cp "$LIBSRC" .fork/scripts/lib.sh
cp "$POLICY" .fork/path-policy
printf 'repository = %s/upstream\nbranch = main\ncommit = %s\n' "$FX" "$U0" > .fork/upstream.lock
git add -A && git commit -qm 'our fork state'

# ------------------------------------------------------------- run -------
cd "$ROOT"
echo "=== fixture run (dry) ==="
bash "$SCRIPT" --repo "$FX/ours" --fork "$FX/fork" --branch main \
  --allow-local-upstream --branch-out replay/fixture --dry-run 2>&1 | sed 's/^/  /'

echo "=== fixture run (replay) ==="
bash "$SCRIPT" --repo "$FX/ours" --fork "$FX/fork" --branch main \
  --allow-local-upstream --branch-out replay/fixture \
  --report "$ROOT/fixture-report.md" 2>&1 | sed 's/^/  /'

echo "=== fixture assertions ==="
cd "$FX/ours"
fail=0
chk() { if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1 (got: $2, want: $3)"; fail=1; fi; }

chk "commits produced"        "$(git rev-list --count main..replay/fixture)" "5"
chk "a.c replayed exactly"    "$(git show replay/fixture:src/core/a.c)" "$(git -C "$FX/fork" show main:src/core/a.c)"
chk "new.cu present"          "$(git show replay/fixture:src/kernels/new.cu)" "$(git -C "$FX/fork" show main:src/kernels/new.cu)"
chk "h.hpp replayed exactly"  "$(git show replay/fixture:include/h.hpp)" "$(git -C "$FX/fork" show main:include/h.hpp)"
chk "t.py deleted"            "$(git cat-file -e replay/fixture:tools/t.py 2>/dev/null && echo present || echo absent)" "absent"
chk "CMakeLists conflict-marked" "$(git show replay/fixture:CMakeLists.txt | grep -c '^<<<<<<<')" "1"
chk "our vendoring survives"  "$(git show replay/fixture:CMakeLists.txt | grep -c vendored)" "1"
chk "their change survives"   "$(git show replay/fixture:CMakeLists.txt | grep -c deadbeef)" "1"
chk "docs stay dropped"       "$(git cat-file -e replay/fixture:docs/d.md 2>/dev/null && echo present || echo absent)" "absent"
chk "serve stays dropped"     "$(git cat-file -e replay/fixture:serve/s.py 2>/dev/null && echo present || echo absent)" "absent"
chk "unknown excluded"        "$(git cat-file -e replay/fixture:newdir/n.txt 2>/dev/null && echo present || echo absent)" "absent"
chk "ours untouched (.github)" "$(git show replay/fixture:.github/w.yml)" "$(git show main:.github/w.yml)"
chk "ours untouched (README)"  "$(git show replay/fixture:README.md)" "$(git show main:README.md)"

echo
if [ "$fail" -eq 0 ]; then echo "ALL FIXTURE CHECKS PASSED"; else echo "FIXTURE FAILURES PRESENT"; exit 1; fi
