# AGENTS.md

Operating rules for this repository and for the workspace around it.

This file is the **canonical copy**: it is versioned with the fork, alongside two companions
that it points at and must stay consistent with:

- [`.fork/README.md`](.fork/README.md) — how upstream synchronization actually works;
- [`FORK_NOTES.md`](FORK_NOTES.md) — the engineering ledger: upstream baseline, what is
  retained, what was removed, divergences, deferred work, uncertainties.

Read all three before changing anything.

## Fork notice

This is a **personal fork** of [`Niko1221/Strata`](https://github.com/Niko1221/Strata) — a
C++/CUDA/HIP mixture-of-experts inference engine (Qwen3.8-Flash-Next / Qwen3.6-35B-A3B) with
its Python installer, server, web app and platform ports — maintained as
[`k0valik/Strata`](https://github.com/k0valik/Strata) for one 16 GB NVIDIA GPU on 32 GB DDR4.

It is a **hard fork with intentional deletions**, not a contribution channel back upstream.
See "Never act on upstream" below.

## Never act on upstream

Upstream `Niko1221/Strata` is **read-only, always**:

- never open pull requests, issues, discussions, comments or releases there;
- never push branches or tags there;
- never make a mutating API call against it.

All repository-mutating work targets `k0valik/Strata` only. Pass `--repo k0valik/Strata`
explicitly to mutating `gh` commands, so a mis-resolved default can never act on the wrong
repository. *Reading* upstream is fine and expected: fetching git objects, `gh api` GETs,
browsing its history.

## Workspace

- `Strata/` — this repository, the working fork. The only repository you modify.
- `UnbiasedStrata/` — sibling directory, reference only. Read it; never modify it, and never
  copy from it wholesale: its choices (engine patches, frontend, removals) are its own
  reworkings against an older engine, not automatically this fork's choices.
- Loose files next to those repositories are scratch input.

Identify which repository you are in before acting; layouts may differ.

## Execution policy

Four tiers. They exist because this fork exists to *reduce* executable and supply-chain
surface: treating "the repository" as one trust domain is the mistake to avoid.

1. **Read anything, always.** Files, `rg`/`grep`/`find`, and read-only git plumbing (`log`,
   `show`, `diff`, `ls-tree`, `rev-list`, `merge-base`, `cat-file`, `blame`). `gh api` for
   repository metadata (default branch, size, action pins). No permission needed.
2. **Write and run your own code in a sandbox.** Scripts you just wrote, fixture suites, and
   experiments belong in throwaway directories — miniature git repositories under `/tmp`,
   temporary clones of the fork — and you are expected to *actually run* them to check that a
   design works rather than reasoning about git behaviour. Dummy data only: no engine, no GPU.
3. **Ask first, or leave it to the maintainer.** Building the engine or llama.cpp, running the
   repository's tests or scripts, anything needing a toolchain or GPU, anything long-running or
   network-heavy. Give the exact command instead of running it, unless the task authorises it.
   The commands such work would use are listed under "Building and verifying" below.
4. **Never.** Anything that came from upstream or another untrusted tree: upstream scripts,
   installers, Make/CMake targets, Python, binaries, hooks, Actions, CI configs, submodule
   contents. Also: package installs, Docker, model downloads, destructive cleanup. Fetching
   *git objects* from a public upstream is fine (tier 2); executing what is inside them is not.

Rule of thumb: read anything, run only what we wrote, never what came from upstream, leave
builds to the maintainer.

## Modification scope

- Modify only this repository; the reference checkout is read-only.
- Keep scratch work out of the fork. Throwaway clones, fixtures and generated reports live in
  temp directories, and this repository's working tree and index are left clean when you stop.
- Prefer a temporary clone over experimenting here, especially for anything that rewrites
  history or fetches. Do not leave the repository on a detached HEAD or a stray branch.

## Fork principles

Goal: a small, understandable fork that keeps the useful Strata inference implementation while
dropping unnecessary machinery and dependency surface.

- Understand a component before removing it. Never delete merely because UnbiasedStrata did.
- Prefer deletion over replacement when something is intentionally out of scope.
- Pruning needs no new abstractions. Fork *maintenance* infrastructure (`.fork/`, the sync
  workflow) is in scope and expected — but keep it small, dependency-free, auditable and
  documented, and stop expanding it once it works.
- Engine sources (`src/`, `include/`) stay byte-identical to upstream where possible, so
  `git diff upstream/main -- src include` stays meaningful. Deviate only when required to keep
  the tree buildable, keep the change minimal and mechanical, and record the porting guidance
  in `FORK_NOTES.md`.
- Keep every change easy to audit and easy to compare with upstream.

### Do not reintroduce

The installer/bootstrap machinery (`setup.py`, `setup.sh`, `*.bat`, `bench.sh`, `update.sh`,
flake files), the Python serving stack (`serve/`, `chat.py`), the Intel SYCL port, the Apple
Metal port, the `docs/` tree and translated READMEs, Docker packaging, benchmark archives
(`bench/results/`) and research data. The serving/API layer is deferred, and is planned to come
from llama.cpp rather than the removed Python stack.

Multi-GPU and the vision encoder are **planned features, not leftovers** — do not remove them.
Windows (MSVC) builds are supported and must keep working even though day-to-day work happens
from WSL2; HIP/AMD, older-CPU ISA floors and gfx906 paths are retained for upstream parity and
are untested here.

## Repository shape

- `src/`, `include/` — the engine (core, kernels, prefill, speculative decoding, plan).
- `tests/` — engine tests. `ref/` — reference GGUF loader. `cmake/` — build helpers.
- `bench/micro/` and `bench/gfx906_*` — micro-benchmarks and parity harnesses used by the build.
- `tools/` — model preparation (`iq_pack`, `mtp_*`, `make_profile*`, `pack_*`, `strata_pack`,
  `strata_tokenizer`, `draft_vocab`, `gguf_*`, `quant_formats`, `nvfp4_*`) plus `tools/vision`
  (mmproj encoder), `tools/hip` (tuning tables) and `tools/windows`. Their unit tests need no
  GPU and no downloads — running them is tier 3.
- `data/` — small runtime data (draft vocabularies, expert profiles).
- `.fork/`, `.github/workflows/sync-upstream.yml` — fork maintenance; the path policy treats
  them as fork-owned.

### Vendored dependency

llama.cpp is a git submodule at `third_party/llama.cpp`. `.gitmodules` records its URL and
`shallow = true`; the commit itself is pinned by the gitlink — `af0579c78a07a555d84830d01da7b61cfadbc7ef`,
tag `b11624` on `ggml-org/llama.cpp`, newer than upstream Strata's own pin. CMake takes ggml,
the MMQ kernel sources and `ggml-common.h` from it (`STRATA_GGML_DIR` overrides the location);
the build fetches nothing itself and fails with a `git submodule update --init` hint when the
submodule is not checked out. **The submodule stays pristine** — no patches — and the
engine-side adaptation required by this pin is described in `FORK_NOTES.md`. An upstream change
to its own llama.cpp integration is reported by the sync tooling, never applied.

### Building and verifying

Tier 3: hand these to the maintainer unless the task authorises running them.

```sh
git submodule update --init
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build
```

The parity tests catch layout and numerics drift (for example `prefill_fused_moe_test`,
`prefill_fused_iq_test`, `prefill_mmq_kquant_test`, `mmq_nvfp4_parity`, and the CPU parity
tests).

## Upstream synchronization

`.fork/README.md` is the reference for the mechanism; `FORK_NOTES.md` is the ledger.

- Never `git merge upstream/main` by hand, and never hand-edit the baseline in
  `.fork/upstream.lock`: `.fork/scripts/sync-upstream.sh` uses that recorded revision as the
  merge base and applies `.fork/path-policy`. Dry-run it before publishing anything.
- A mirrored file may carry a fork patch: the merge is three-way, so your patch is the *ours*
  side and upstream's change is applied on top of it — a clean merge keeps both, an overlapping
  edit stops the run. Keep such paths `mirror` (never `ours`, which would silently discard
  upstream's other changes), and record the reason under "Active source divergences" in
  `FORK_NOTES.md`; the candidate reports them as fork divergences that were touched.
- Treat upstream as untrusted: the tooling reads git objects only, fails closed, and must never
  attach a provenance claim (the second parent, the lock advance) to a range a human has not had
  a chance to review.
- The rolling pull request body is the current state. A run that cannot integrate a newer range
  must update it to say so rather than leaving a healthy-looking candidate. No per-run comments.
- Merge synchronization PRs with a **merge commit** (squash works but loses the ancestry
  record). Verifying a candidate by building and testing the engine is manual, and the
  maintainer's call.

## Git discipline

- One commit per phase of work, each self-describing and independently reviewable (pruning,
  vendoring, engine adaptations, sync infrastructure). Prefer few, large, coherent commits over
  a scatter of small ones; the maintainer dislikes fragmented history.
- A ledger or documentation update that describes a change belongs *in that change's commit*,
  not in a separate docs commit.
- Stage explicit paths; never `git add -A`.
- Inspect the complete diff before committing. No generated files, build artifacts,
  credentials, model data, or unrelated local changes.
- Rewriting history is allowed only while it is unpushed, and only when asked. Before: confirm
  the remote state (`git rev-parse origin/main`, `gh api repos/k0valik/Strata/branches/main`) —
  local commits are often well ahead of what GitHub has. After: confirm the tree is
  byte-identical (`git diff --stat <old> HEAD`), the tree is clean, and the old commits are
  still reachable in the reflog. Never rewrite published history.
- Never `git reset --hard`: it is blocked by policy and destructive. Move refs with
  `git update-ref -m <why> <ref> <new> <old>` (keeps a reflog entry) or `git reset --soft`.
- Commit messages: a subject saying what changed, a body saying why, and an explicit statement
  of what was verified and what was not.

## Verification and reporting

- Do not assert that untested mechanics work. Prototype them with real commands against
  miniature fixtures, then say plainly which parts were exercised and which were not.
- Give verification left to the maintainer as an exact command.
- Report concisely: what changed (with paths), what was verified, what is uncertain, and any
  decision that is genuinely the maintainer's (history shape, dependency pins, scope). Ask
  instead of assuming — the maintainer welcomes being consulted on real trade-offs.

## Fork notes

Maintain `FORK_NOTES.md` as an engineering ledger, not a diary.

Record: the upstream baseline/revision; what is retained; what was intentionally removed;
intentional divergences **with the porting guidance a future synchronization needs**;
dependency/vendor decisions; deferred work; and uncertainties.

Keep it factual, compact and current: a stale entry (completed work still listed as deferred,
obsolete pins, wrong counts) is a defect, not a cosmetic issue. Do not add per-run bot history;
the rolling PR and the git log already carry that.
