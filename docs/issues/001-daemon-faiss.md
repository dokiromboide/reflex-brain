# Issue #1 — P0-01: daemon embedding delivery to FAISS

Issue: https://github.com/dokiromboide/reflex-brain/issues/1

## Scope and workflow

First item in the established P0-01 → P0-02 → P0-03 → P0-04 → P0-05 sequence.
Implementation belongs exclusively to `fix/issue-1-daemon-faiss`, targeting `master`.
This scope-only commit opens the draft PR before implementation starts.

## Acceptance criteria

- A new source message is persisted before indexing.
- Every completed embedding reaches FAISS with the persisted object's stable ID.
- Semantic retrieval returns the newly persisted content.
- Restart preserves the index and vector-to-object mapping.
- Reprocessing a source message does not duplicate its stored object or vector.
- Embedding/index persistence failures remain retryable without losing the SQLite reference.
- Logs identify delivery/retry failures without exposing message content.

## Verification plan

1. Record existing test/lint failures before implementation.
2. Add and execute a failing regression test for each behavior, then implement its fix.
3. Exercise real temporary SQLite databases and real FAISS indexes; use a deterministic
   test encoder for offline failure-path tests, and a locally available real encoder
   for the semantic end-to-end check if available.
4. Exercise duplicate processing, restart, and failed embedding/index persistence.
5. Run the existing suite in an isolated working directory (existing tests use
   relative index paths), compare failures against baseline, and obtain independent review.
6. Commit/push only scoped changes, verify the remote PR, merge after verification,
   then verify `master` and remove the temporary branch.

## Boundaries

No writes to the user's live brain, source conversation database, or existing index files.
No inclusion or modification of the pre-existing untracked `DETAILED_PLAN.md` or `reflex_plan/`.
No unrelated MCP, scheduler, migration, global lint, or CI overhaul in this issue.

## Repository baseline observation

The checked-in CI workflow currently targets `main`/`develop`, not the actual default
branch `master`. A PR to `master` therefore may have no CI runs; absence of checks
must not be reported as passing CI. Local verification and its output are required.

## Results

Pending implementation and verification. No acceptance criterion is claimed complete yet.
