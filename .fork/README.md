# `.fork/` — keeping this fork in step with upstream

This directory is the fork's maintenance area: the metadata and tooling that bring
upstream Strata's changes into this repository without resurrecting what the fork
deliberately removed.  It is fork-owned: upstream has none of it, and the path
policy protects it from upstream changes.

Durable policy and divergences are recorded in [`FORK_NOTES.md`](../FORK_NOTES.md);
this file explains the mechanism.

## Files

| path | what it is |
|---|---|
| `upstream.lock` | upstream repository, branch, and the revision this fork's default branch already contains |
| `path-policy` | how an upstream change to a path is classified |
| `fork-candidates` | foreign forks worth replaying onto this fork (slug, URL, branch) |
| `scripts/sync-upstream.sh` | prepares (and optionally publishes) the synchronization candidate |
| `scripts/validate-sync.sh` | static checks for this setup, repository invariants and the workflow |
| `scripts/lib.sh` | shared helpers (policy parsing, `git merge-tree` plumbing, reporting) |
| `scripts/fork-replay.sh` | replays a foreign fork's retained-surface work as squashed, policy-filtered commits |
| `scripts/replay-candidates.sh` | runs `fork-replay.sh` over `fork-candidates` |
| `tests/run.sh` | fixture tests: miniature repositories exercising every sync outcome |
| `tests/replay-fixture.sh` | fixture test for the replay tool (dummy text repositories, no network) |

## What a synchronization run does

1. reads `upstream.lock` and `path-policy` **from the base branch's tree** (not from
   a working tree, and not from the candidate) — the branch being synchronized
   defines the rules;
2. asks upstream for its branch tip with `git ls-remote`; if it equals the accepted
   revision, stops immediately having downloaded nothing;
3. otherwise fetches that branch's git objects (`--no-tags --no-recurse-submodules`);
4. fails closed unless the accepted revision is present and is an **ancestor** of the
   new upstream tip (a rewrite needs a human);
5. classifies every changed path, then runs `git merge-tree --write-tree` — an
   in-memory three-way merge, no checkout, no worktree;
6. forces the classified paths to the fork's version (`ours`, `dropped`) and stops on
   anything that needs judgement (`protected`, `unknown`, conflicts on `mirror` paths);
7. writes the candidate as a single commit on top of the base branch, with
   `.fork/upstream.lock` advanced to the new upstream revision;
8. pushes it to the rolling branch `sync/upstream` (with `--force-with-lease`) and
   creates or refreshes the rolling pull request `sync: upstream Strata`.

Nothing is executed from upstream at any point: the run only reads commits, trees
and blobs.  It also never touches a working tree or the real index — the merge and
the candidate tree are built in a throwaway index file.

## Lifecycle

1. The run prepares a candidate and pushes it to `sync/upstream`, opening the rolling
   pull request if none is open.
2. You review the report, build the candidate by hand, and merge it — with a **merge
   commit** to keep the ancestry record.
3. The next run reads the new baseline from `main`, finds upstream unchanged, and exits
   "UP TO DATE" without touching anything: no branch, no push, no pull request.
4. When upstream does advance, the *same* `sync/upstream` branch is replaced
   (`--force-with-lease`) and a **new** pull request is opened; the merged one stays as the
   record of what was accepted.  The merged candidate is still recognised as
   automation-owned, so reusing the branch works normally.  Deleting the head branch on
   merge is fine too: the next run recreates it.
5. If a newer range cannot be integrated, the branch is left untouched but an already
   open pull request has its body rewritten to 🔴 BLOCKED — it can never keep showing a
   healthy candidate for a synchronization that has since stopped working.  The report
   states precisely what did and did not happen: "nothing was pushed" is distinguished from
   "the candidate did reach the branch but the run did not finish", and it is produced even
   when the metadata itself cannot be read.  The next successful run restores the 🟢/🟡 body.

The report (and therefore the pull-request body) is regenerated in full on every run.  It
is never appended to, and the script posts no comments: the body is the rolling dashboard.

### Reading the report

- **🟢 READY** — mechanically clean, nothing fork-sensitive touched.  The routine case.
- **🟡 REVIEW** — clean, but upstream touched something worth a look before merging: the
  build integration this fork edited (`CMakeLists.txt`, `cmake/`), upstream's own
  dependency integration (`third_party/`), the kept runtime data, or a mirrored file the
  fork has patched of its own accord (see `Fork divergences touched` below).
- **🔴 BLOCKED** — a human decision is needed and nothing was pushed.

`Retained runtime affected` answers the question a raw file count hides: whether any of
those changes actually reach the running engine (`src/`, `include/`, `cmake/`,
`CMakeLists.txt`) — "400 files changed" can mean 395 removed docs and five kernels.  Long
path lists and long commit ranges are explicitly marked when truncated, so the report
never silently omits what it is describing.

`Fork divergences touched` answers a different question: whether upstream changed a **mirrored
file this fork has patched**, where the merge is clean only because the two edits are on
different lines.  A clean merge says nothing about meaning — if upstream fixed the same thing
its own way, the merged file can hold a duplicated or contradictory fix — so the report lists
the paths and asks for the file to be read, not just its status.  The reason each divergence
exists is not derivable from git and lives in `FORK_NOTES.md` under "Active source divergences".

## Replaying foreign forks

Upstream has forks that carry real engine work, but hundreds of commits ahead
mixed with thousands of files this fork deliberately dropped is not reviewable.
`fork-replay.sh` makes them reviewable:

1. fetch the foreign fork and upstream (git objects only — nothing executed);
2. diff their fork point against their tip;
3. classify every changed path with **the same** `.fork/path-policy` the sync
   uses — dropped/ours/protected paths never enter the replay at all;
4. replay what survives onto this fork's branch as **one squashed commit per
   area**, applied with a three-way merge; each commit names the original
   authors and the original commit subjects it came from;
5. verify: only mirrored paths may change, and every file that was pristine on
   our side must come out byte-identical to the foreign fork's version.

Files whose hunks land on lines this fork has changed are **not discarded**:
they come in as three-way merges with conflict markers, exactly like a pull
request with conflicts — resolve them, do not throw the work away.  A group
whose merged result matches this fork's tree is reported as "already present".

A foreign fork whose history shares **no ancestry** with upstream (rewritten or
grafted commits) has no fork point to diff against.  Such a replay is flagged
**HISTORY REPLACED** in the report: the diff is their tree against the accepted
baseline and may contain upstream's own post-fork changes in reverse, so its
diffstat is an upper bound and its whole-file deletions are never replayed
automatically.  Read such branches per file; the byte-exact verification means
"matches their tree", not "safe to take".

The output is a review artifact, never evidence that anything works: the replay
is not built and not tested.  Nothing is pushed and no pull request is opened
unless `--push`/`--pr` is given; the pull request body is the run's report.

## The accepted revision is the merge base

`upstream.lock`'s `commit` is the source of truth for "what upstream state does this
fork already contain", and the run passes it to git explicitly:

```
git merge-tree --write-tree --merge-base=<lock commit> <base branch> <upstream tip>
```

This is what makes a **squash merge** of the synchronization pull request correct.
A squash merge produces a new commit with no upstream ancestry, so git's own idea of
the merge base goes stale (it stays at the original fork point).  The lock file does
not: the candidate always advances it, so the next run applies exactly
`<previous tip>..<new tip>` — never re-applying what `main` already has, and never
producing spurious conflicts from a stale base.

Because the content merge always uses the lock as its base, the invariant is: bring
upstream in **with this tooling**, not by hand.  A hand `git merge upstream/main`
uses git's own merge base — correct once a synchronization has been merge-committed,
stale after a squash — and knows nothing about the path policy, so it would happily
resurrect dropped paths.

## Ancestry, and the "N commits behind" badge

GitHub's "this branch is N commits behind upstream" comes from the commit graph, not
from content.  A fork that only ever squash-merges its synchronizations therefore
advertises itself as hundreds of commits behind even when its content is current,
which reads as "abandoned".

So the candidate commit carries two parents: the base branch, and the upstream tip it
integrates.  Its *tree* is still the policy-filtered merge described above (it is not
a plain `git merge` result), but the extra parent records upstream's ancestry:

- merged with **Create a merge commit**: `main` contains `upstream/main`, the badge
  reads "behind 0", `git merge-base main upstream/main` is correct again, and
  `git diff upstream/main` shows exactly this fork's divergences;
- merged with **squash** or **rebase**: the extra parent is dropped and nothing is
  broken — `.fork/upstream.lock` is still the merge base for content — but the badge
  goes back to "behind".

Either way is safe, so this is a display choice rather than a correctness one.  A
single merge-committed synchronization grafts all of upstream's history at once, so a
fork that was squashed previously is repaired by the next merge commit; there is no
separate repair mode.

### The ancestry assertion

The second parent is not cosmetic: it is a permanent claim that **every change in
`<recorded baseline>..<upstream tip>` has been accounted for**, so future git
operations are entitled to use that tip as a merge base and never offer those changes
to this fork again.  A candidate may therefore only be created once the whole range has
been classified and resolved deterministically.  The sync enforces that by
construction: a `protected` or unclassified path, or a conflict on a `mirrored` path,
stops the run before any commit object exists, so a failed run produces no candidate
and no ancestry graft.  An assertion of "accounted for" is never attached to something
a human still has to look at.

Content and ancestry are also checked against each other.  Every candidate carries
provenance trailers:

```
Fork-Sync-Bot: sync-upstream.sh
Fork-Sync-Automated: true
Fork-Sync-Base: <the candidate's first parent>
Fork-Sync-Upstream: <the candidate's second parent>
```

Before the rolling branch is replaced, those trailers are compared against the
commit's actual parents (exactly two), the first parent must be on this fork's line of
history, and the `.fork/upstream.lock` *inside the candidate* must record the same
revision as its second parent.  A human commit — or one that merely copies the
trailers — cannot satisfy that, so the branch is left alone; `--force-with-lease`
guards the push itself.

If the badge ever needs fixing sooner than the next synchronization, record only the
revision already accepted — never a tip whose changes have not been accounted for:

```sh
git merge -s ours "<the commit in .fork/upstream.lock>"   # ancestry only, tree unchanged
```

Because that record can always be rebuilt from `upstream.lock`, the lock file stays the
source of truth.

## Path policy

`path-policy` is a list of `<mode> <glob>` rules, matched in order, first match wins
(so exceptions sit above the rule they except).  `*` also crosses `/`, so `src/**`
and `src/*` behave the same.

| mode | meaning |
|---|---|
| `mirror` | take upstream's version: new files, edits, deletions |
| `dropped` | the path stays absent; the fork removed it on purpose |
| `ours` | the fork owns the file: keep ours, report upstream's change |
| `protected` | upstream must never touch this; a change stops the run |
| `unknown` | no rule matched — the run stops and asks for a classification |

The `pruned` line is the built-in rule for the fork's own deletions: a path present
in the accepted upstream revision but absent here was removed deliberately and stays
removed.  Its position matters — it keeps pruned files dropped even when a later
`mirror` rule covers their directory (for example `tools/**`).  It must appear
exactly once; `validate-sync.sh` enforces that, because losing it would let the next
synchronization quietly restore a removed file.

Deliberately *not* a rule: this fork follows upstream's engine (`src/`, `include/`,
`tests/`, `cmake/`, `ref/`, `tools/`, `bench/micro/`, `CMakeLists.txt`) and keeps its
own copies of the fork-control surface (`.fork/**`, `.gitmodules`,
`third_party/llama.cpp`) and the rewritten root documents.

A `mirror` path may still carry a fork patch: the merge is three-way, so the fork's version is
the *ours* side and upstream's change is applied on top of it — a clean merge keeps both, an
overlapping edit stops the run.  Keeping such a file `mirror` is the point: upstream's other
changes to it keep flowing in.  Classifying it `ours` instead would silently discard them and
leave the change visible only as a "does not take" list entry.  `CMakeLists.txt` is exactly this
case (the vendoring edits), and `FORK_NOTES.md` records the reason for every such divergence.

## The vendored llama.cpp submodule

`third_party/llama.cpp` is a fork decision, pinned in `.gitmodules` and as a gitlink.
The policy protects all three (`.gitmodules`, the gitlink path, and `.fork/**`), so an
upstream change to its own llama.cpp integration is reported, never applied.  The
synchronization never fetches submodules and never alters the gitlink.

## Running it locally

```sh
.fork/scripts/validate-sync.sh                  # setup + repository invariants
.fork/scripts/sync-upstream.sh                  # dry run against the base branch
.fork/scripts/sync-upstream.sh --classify src/  # what would happen to a path
.fork/scripts/fork-replay.sh --fork URL --branch NAME          # replay one fork
.fork/scripts/fork-replay.sh --fork URL --branch NAME --pr     # ... as a PR
.fork/scripts/replay-candidates.sh .fork/fork-candidates /tmp/reports  # the queue
.fork/tests/run.sh                              # fixture tests for the sync
.fork/tests/replay-fixture.sh                   # fixture test for the replay
```

The dry run fetches upstream's objects into `refs/remotes/fork-sync/*`, writes the
candidate commit as a dangling object, and reports what it would do — it creates no
branch, pushes nothing, and leaves the current branch, index and working tree
untouched.  Inspect the candidate with `git show <candidate>`.

Useful flags: `--repo DIR` (another checkout, used by the tests), `--base-ref REF`,
`--upstream-url URL` / `--upstream-branch NAME` (overrides for testing),
`--report FILE` (markdown, doubles as the pull-request body), `--push`, `--pr`.

## Safety properties

- **Fails closed.**  A rewritten upstream history, a protected path, an unclassified
  path, or a merge conflict on a mirrored path stops the run and publishes nothing;
  an existing valid rolling pull request is left alone.
- **Does not clobber human work.**  The rolling branch is only replaced when its tip
  carries the `Fork-Sync-Bot: sync-upstream.sh` trailer *and* sits on top of the base
  branch; the push uses `--force-with-lease`.  Otherwise the run stops and reports.
- **No upstream execution.**  No checkout of upstream content, no submodule fetch, no
  hook, no `make`/`cmake`/`python` from a candidate, no downloads from a candidate.
- **Trusted implementation.**  The workflow checks out the default branch and runs
  `.fork/scripts/*` from it; the candidate branch is never executed.
- **Few moving parts.**  One pinned action (`actions/checkout`), the repository's own
  `GITHUB_TOKEN` with `contents: write` and `pull-requests: write`, and the GitHub CLI
  that ships with the runner.

## Required repository setting

GitHub must be allowed to open the rolling pull request:

```
Settings -> Actions -> General -> Workflow permissions ->
  [x] Allow GitHub Actions to create and approve pull requests
```

## Out of scope

There is no engine CI here on purpose.  A synchronization candidate changes engine
sources and must be **built and tested by a human** before the pull request is
merged (`cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON`,
`ctest`).  A build job can be added on top of this workflow later; it does not belong
in the synchronization mechanism itself.
