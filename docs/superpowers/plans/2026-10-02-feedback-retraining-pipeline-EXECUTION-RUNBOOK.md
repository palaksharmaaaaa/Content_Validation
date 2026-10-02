> **SUPERSEDED (2026-10-02):** this describes the earlier copy-into-a-queue design. The implemented design reads media in place by content hash; see `core/media_library.py`, `core/retrain_engine.py`, `<modality>_detector/retrain.py` and the README section "Train on your own media, in place".

# Execution Runbook: Feedback-Driven Retraining Pipeline (Subagent-Driven Development)

**Status:** Not yet executed. This document is the complete, ready-to-run dispatch sequence for implementing `docs/superpowers/plans/2026-10-02-feedback-retraining-pipeline.md` via `superpowers:subagent-driven-development`. Nothing in this document has been run — it's the plan for *how* to run the plan, written out in full per the user's explicit request before any subagent is dispatched.

**Branch strategy (explicit user decision):** Work directly on `main`, the same way the rest of this session has. No worktree, no feature branch. Every task commits straight to `main` as it completes.

**How to use this document:** Work top to bottom, one task at a time, never in parallel. For each task: dispatch the Implementer exactly as written, handle its status per the skill's rules, generate the review package, dispatch the Task Reviewer exactly as written, resolve any findings, update the progress ledger, then move to the next task. After Task 13, run the Final Whole-Branch Review, then `superpowers:finishing-a-development-branch`.

---

## Pre-flight scan (completed)

One conflict was found and resolved with the user before this document was written:

**Finding:** Tasks 4/7/10 (`retrain.py` × 3) and Tasks 3/6/9 (`learner.py` edits × 3) are deliberately near-identical code, mirroring this codebase's existing architecture (every other module — `scoring.py`, `batch.py`, `validator.py`, etc. — is already independently duplicated per modality rather than shared via `core/`). A task reviewer's standard rubric flags "verbatim duplication of a logic block" as Important.

**Resolution (user decision):** Each reviewer dispatch below includes an explicit note in its Global Constraints block that this per-modality duplication is plan-mandated and matches the project's existing architecture — reviewers should not flag the duplication *itself* as a defect. They remain free to flag anything else (bugs, missed requirements, poor quality *within* a single file).

No other conflicts found. Proceed task by task below.

---

## Model selection (per SKILL.md's Model Selection section)

| Task | Nature | Implementer model | Reviewer model |
|---|---|---|---|
| 1 | `core/corrections_store.py` — mechanical, complete code given | `haiku` | `sonnet` |
| 2 | image `config.py` + `trainer.py` extension — mechanical, complete code given | `haiku` | `sonnet` |
| 3 | image `learner.py` edit — mechanical, complete code given | `haiku` | `sonnet` |
| 4 | image `retrain.py` — new module, multi-file integration, real training run in tests | `sonnet` | `sonnet` |
| 5 | audio `config.py` + `trainer.py` extension — mechanical, complete code given | `haiku` | `sonnet` |
| 6 | audio `learner.py` edit — mechanical, complete code given | `haiku` | `sonnet` |
| 7 | audio `retrain.py` — new module, multi-file integration, real training run in tests | `sonnet` | `sonnet` |
| 8 | video `config.py` + `trainer.py` extension (incl. new validation split) — mechanical, complete code given, but changes established behavior | `sonnet` | `sonnet` |
| 9 | video `learner.py` edit — mechanical, complete code given | `haiku` | `sonnet` |
| 10 | video `retrain.py` — new module, multi-file integration, real training run in tests | `sonnet` | `sonnet` |
| 11 | Git LFS setup — environment-dependent, judgment on install/verification issues | `sonnet` | `sonnet` |
| 12 | UI integration — multi-file, manual Streamlit verification | `sonnet` | `sonnet` |
| 13 | README updates + final full-suite verification — mechanical text edits | `haiku` | `sonnet` |
| Final whole-branch review | broad judgment across the entire 13-task diff | — | `opus` |

---

## Progress ledger

Location: `.superpowers/sdd/progress.md` (git-ignored scratch — survives this session but not a fresh clone; recoverable from `git log` if lost per the skill's Durable Progress section).

Before starting Task 1, create it:

```bash
mkdir -p .superpowers/sdd
cat > .superpowers/sdd/progress.md <<'EOF'
# Progress ledger: feedback-driven retraining pipeline

Plan: docs/superpowers/plans/2026-10-02-feedback-retraining-pipeline.md
Branch: main (no worktree, per explicit user decision)

EOF
```

After each task's review comes back clean, append one line:
`Task N: complete (commits <base7>..<head7>, review clean)`

Task briefs already generated (via `scripts/task-brief`) at:
`.superpowers/sdd/task-1-brief.md` through `.superpowers/sdd/task-13-brief.md`

Report files (implementer writes, fix subagents append to) will be created at:
`.superpowers/sdd/task-1-report.md` through `.superpowers/sdd/task-13-report.md`

---

## The Global Constraints block (copy verbatim into every reviewer dispatch's `[GLOBAL_CONSTRAINTS]`)

```
- Retraining stays BINARY (ai_generated=0 / real=1) — matches every existing checkpoint's capacity exactly. No multi-class taxonomy work in this plan.
- Personal media must never be written anywhere git-tracked. data/corrections/{pending,archive}/ is already covered by the repo's existing blanket data/ gitignore rule — no new gitignore entries needed for it.
- models/*.pt is tracked via Git LFS (Task 11), not gitignored.
- A retrain only replaces the live checkpoint if candidate validation accuracy >= the current checkpoint's last-logged accuracy, or unconditionally if no checkpoint has ever been promoted (CHECKPOINT_LOG.md has no rows yet). On rollback: the candidate file is deleted, the pending queue is left untouched, nothing about the live checkpoint changes.
- Every new/modified function with a default parameter must preserve prior behavior exactly when called the old way.
- Validation accuracy is read directly from each trainer's own history return value (val_acc/val_accuracy) — never fabricated, never computed from an unrelated quantity like training loss. CHECKPOINT_LOG.md's "Train Loss" column is deliberately loss, not a fabricated accuracy (see plan's Self-Review Notes).
- Full test suite (pytest) must pass after every task.
- PLAN-MANDATED DUPLICATION: Tasks 3/6/9 (learner.py edits) and Tasks 4/7/10 (retrain.py modules) are deliberately near-identical per-modality code, matching this codebase's existing architecture (every other module in image_detector/audio_detector/video_detector is already independently duplicated rather than shared via core/). Do NOT flag this cross-task duplication itself as a defect — it was an explicit human decision made before implementation began. You remain free to flag any other issue, including quality problems within a single file.
```

---

## Task 1: `core/corrections_store.py`

**Dependencies:** None — first task.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 1: core/corrections_store.py"
  model: haiku
  prompt: |
    You are implementing Task 1: core/corrections_store.py -- shared feedback-queue
    and checkpoint-log bookkeeping

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-1-brief.md
    It contains the full task text from the plan, including complete code for
    every file.

    ## Context

    This is the foundation module for a larger feature: closing the loop between
    user feedback and the actual neural checkpoint across three sibling packages
    (image_detector, audio_detector, video_detector). This task only touches
    core/ -- a shared, presentation-independent layer that never imports from
    any of the three detector packages (one-directional dependency, verified
    elsewhere in this codebase). Nothing downstream exists yet; you are building
    in isolation and nothing else in the repo currently calls this module.

    core/atomic_io.py (existing, do not modify) provides:
    - atomic_read_json(file_path, default=None) -> Any
    - atomic_write_json(file_path, data, indent=2) -> None
    Use these for all file I/O that needs crash-safety, exactly as the brief's
    code shows.

    ## Before You Begin

    If you have questions about the requirements, approach, or anything unclear
    in the task description, ask them now.

    ## Your Job

    1. Implement exactly what the task brief specifies
    2. Write tests (TDD: the brief's Step 1 is the failing test, write it first)
    3. Verify implementation works
    4. Commit your work
    5. Self-review (see below)
    6. Report back

    Work fro<repo>

    **While you work:** If you encounter something unexpected or unclear, ask
    questions. Don't guess or make assumptions.

    While iterating, run the focused test for what you're changing; run the
    full suite once before committing, not after every edit.

    ## Code Organization

    - Follow the file structure defined in the brief exactly: one new file
      core/corrections_store.py, one new test file core/tests/test_corrections_store.py
    - If a file you're creating is growing beyond the plan's intent, stop and
      report it as DONE_WITH_CONCERNS -- don't split files on your own
    - Follow established patterns in core/atomic_io.py and core/security.py
      (both self-contained, no intra-project imports, this module should match)

    ## When You're in Over Your Head

    It is always OK to stop and say "this is too hard for me." Bad work is
    worse than no work.

    STOP and escalate when: the task requires architectural decisions with
    multiple valid approaches, you need to understand code beyond what was
    provided and can't find clarity, you feel uncertain whether your approach
    is correct, or you've been reading file after file without progress.

    How to escalate: report back with status BLOCKED or NEEDS_CONTEXT,
    describing specifically what you're stuck on.

    ## Before Reporting Back: Self-Review

    Check completeness (fully implemented everything in the brief? edge cases
    handled?), quality (clear names, clean code), discipline (YAGNI -- only
    what was requested, no extra helper functions or parameters not in the
    brief), and testing (tests verify real behavior, TDD followed, output
    pristine with no stray warnings).

    If you find issues during self-review, fix them now before reporting.

    ## Report Format

    Write your full report to .superpowers/sdd/task-1-report.md:
    - What you implemented
    - What you tested and test results
    - TDD Evidence: RED (command + failing output + why expected) and GREEN
      (command + passing output)
    - Files changed
    - Self-review findings (if any)
    - Any issues or concerns

    Then report back with ONLY (under 15 lines):
    - Status: DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
    - Commits created (short SHA + subject)
    - One-line test summary
    - Your concerns, if any
    - The report file path
```

### Controller steps after implementer reports

1. Record `BASE_SHA` = the `git rev-parse HEAD` value from **before** this dispatch (capture this before dispatching, not after).
2. Handle status per skill rules (DONE → proceed; DONE_WITH_CONCERNS → read concerns, address if correctness-related; NEEDS_CONTEXT/BLOCKED → resolve and re-dispatch).
3. Run: `"<home>/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/subagent-driven-development/scripts/review-package" BASE_SHA HEAD` — note the printed diff file path as `[DIFF_FILE]`.

### Task Reviewer dispatch

```
Subagent (general-purpose):
  description: "Review Task 1 (spec + quality)"
  model: sonnet
  prompt: |
    You are reviewing one task's implementation: first whether it matches its
    requirements, then whether it is well-built. This is a task-scoped gate,
    not a merge review -- a broad whole-branch review happens separately after
    all tasks are complete.

    ## What Was Requested

    Read the task brief: .superpowers/sdd/task-1-brief.md

    Global constraints from the spec/design that bind this task:
    [paste the "Global Constraints block" from this runbook's dedicated section above]

    ## What the Implementer Claims They Built

    Read the implementer's report: .superpowers/sdd/task-1-report.md

    ## Diff Under Review

    Base: [BASE_SHA recorded by controller]
    Head: [current HEAD SHA]
    Diff file: [DIFF_FILE path printed by review-package]

    Read the diff file once -- it contains the commit list, a stat summary,
    and the full diff with surrounding context. Do not Read a changed file
    separately unless a hunk is cut off mid-function. Do not re-run git
    commands. Do not crawl the broader codebase beyond one focused check per
    concrete, named risk.

    Your review is read-only on this checkout. Do not mutate the working
    tree, the index, HEAD, or branch state in any way.

    ## Do Not Trust the Report

    Treat the implementer's report as unverified claims. Verify against the
    diff. A stated rationale never downgrades a finding's severity.

    ## Tests

    The implementer already ran the tests and reported results with TDD
    evidence. Do not re-run the suite to confirm their report. Run a test
    only when reading the code raises a specific doubt no existing run
    answers -- a focused test, never the package-wide suite.

    Warnings or other noise in the implementer's reported test output are
    findings -- test output should be pristine.

    ## Part 1: Spec Compliance

    Compare the diff against What Was Requested: Missing, Extra, Misunderstood.
    If a requirement can't be verified from this diff alone, report it as a
    ⚠️ item instead of broadening your search.

    ## Part 2: Code Quality

    Code quality (clean separation, proper error handling, DRY without
    premature abstraction, edge cases), Tests (verify real behavior, cover
    the task's edge cases), Structure (one clear responsibility per file,
    independently testable units, matches the plan's file structure).

    Your report should point at evidence: file:line for every finding and
    for any check you'd otherwise answer with a bare "yes."

    ## Calibration

    Categorize by actual severity. Important means this task cannot be
    trusted until fixed. "Coverage could be broader" and polish are Minor.
    If the brief explicitly mandates something this rubric calls a defect,
    that IS a finding -- report it Important, labeled plan-mandated, EXCEPT
    for the specific plan-mandated duplication named in the Global
    Constraints block above, which you must not flag.
    Acknowledge what was done well before listing issues.

    ## Output Format

    ### Spec Compliance
    - ✅ Spec compliant | ❌ Issues found: [...]
    - ⚠️ Cannot verify from diff: [...]

    ### Strengths

    ### Issues
    #### Critical (Must Fix)
    #### Important (Should Fix)
    #### Minor (Nice to Have)

    ### Assessment
    **Task quality:** [Approved | Needs fixes]
    **Reasoning:** [1-2 sentences]
```

### Controller steps after review

- Clean (✅ spec, Approved quality) → append ledger line `Task 1: complete (commits <base7>..<head7>, review clean)`, move to Task 2.
- Issues found → dispatch a fix subagent (same model as implementer, `haiku`) with the complete findings list, re-run `review-package` with the new HEAD, re-dispatch the reviewer, repeat until clean.

---

## Task 2: `image_detector` — config additions + `trainer.py` extra-samples support

**Dependencies:** None (independent of Task 1's module internals, though it will be imported alongside it later).

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 2: image_detector config + trainer.py extra-samples"
  model: haiku
  prompt: |
    You are implementing Task 2: image_detector -- config additions + trainer.py
    extra-samples support

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-2-brief.md
    It contains the full task text from the plan, including the exact config
    additions and the complete replacement code for prepare_data().

    ## Context

    This is part of a larger feature (feedback-driven retraining) spanning
    image_detector, audio_detector, and video_detector. This task only touches
    image_detector/config.py and image_detector/trainer.py. A later task
    (Task 4, not yours) will build image_detector/retrain.py on top of the
    extra_samples parameter you're adding here -- so its exact signature and
    behavior matter even though nothing calls it yet within this task.

    image_detector/trainer.py's prepare_data() currently has two branches
    (pre-split train/val directories, or flat ai_generated/real directories)
    each with their own early return. Your brief's replacement code merges
    both branches to fall through to shared extra_samples-merging and loader-
    construction logic at the end -- read the full replacement carefully, it
    is a structural change to the method, not just an addition at the end.

    ## Before You Begin

    If anything is unclear, ask now.

    ## Your Job

    1. Add the three config constants to image_detector/config.py exactly as
       specified
    2. Replace prepare_data()'s body in image_detector/trainer.py exactly as
       specified in the brief (TDD: write the brief's failing test first)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test while iterating; run the full suite once before
    committing.

    ## Code Organization

    Only modify the two files named. Don't touch anything else in
    image_detector/, even if you notice something else you'd improve.

    ## When You're in Over Your Head

    Stop and escalate (BLOCKED or NEEDS_CONTEXT) rather than guess, especially
    if the existing prepare_data() code in the file doesn't match what the
    brief describes finding there -- that would mean the file has changed
    since the plan was written, and you need the controller to reconcile it.

    ## Before Reporting Back: Self-Review

    Specifically verify: does prepare_data(dataset_dir) called WITHOUT
    extra_samples behave identically to before your change? This is a hard
    requirement, not a nice-to-have.

    ## Report Format

    Write your full report to .superpowers/sdd/task-2-report.md with the same
    structure as before (what you implemented, test results with TDD
    evidence, files changed, self-review, concerns).

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern as Task 1: record BASE_SHA before dispatch, run `review-package`, dispatch reviewer with `model: sonnet`, brief `.superpowers/sdd/task-2-brief.md`, report `.superpowers/sdd/task-2-report.md`, the same Global Constraints block, and the same reviewer prompt body as Task 1's Task Reviewer dispatch (only the brief/report/description/diff paths change — reuse that exact template for every remaining task's reviewer dispatch in this document; it is not repeated verbatim below to keep this runbook's own length manageable, but every word of it applies unchanged except the bracketed placeholders).

---

## Task 3: `image_detector` — `learner.py` queues corrections

**Dependencies:** Task 1 (`core.corrections_store.queue_correction`) must be complete and committed.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 3: image_detector learner.py queues corrections"
  model: haiku
  prompt: |
    You are implementing Task 3: image_detector -- learner.py queues corrections
    for retraining

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-3-brief.md
    It contains the full task text, including the exact import changes,
    __init__ replacement, and record_feedback() insertion point and content.

    ## Context

    Task 1 (already complete, committed to main) added core/corrections_store.py
    with a queue_correction(corrections_dir, file_bytes, label, original_ext) -> bool
    function. This task wires ImageSelfImprover.record_feedback() to call it,
    so every user correction gets queued for a future retrain (Task 4, not
    yours, will consume this queue).

    image_detector/config.py already has CORRECTIONS_DIR defined (added in
    Task 2, already complete, committed to main).

    The brief shows an exact "find this code / replace with this code" diff
    for record_feedback() -- the insertion point is identified by the
    surrounding atomic_write_json try/except block already in the file. If
    that surrounding code doesn't match what's currently in
    image_detector/learner.py, STOP and report NEEDS_CONTEXT rather than
    guessing where to insert.

    ## Before You Begin

    If anything is unclear, ask now.

    ## Your Job

    1. Change the imports block and __init__ exactly as specified
    2. Insert the new try/except block into record_feedback() at the exact
       location shown in the brief (TDD: write the brief's failing test first)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test while iterating; run the full suite once before
    committing. The existing test_learner test (not the one you're adding)
    does not pass a corrections_dir, so it will default to the real
    CORRECTIONS_DIR and queue one throwaway correction as a side effect --
    this is expected and harmless (data/corrections/ is gitignored); do not
    try to "fix" this by changing the existing test.

    ## Code Organization

    Only modify image_detector/learner.py and
    image_detector/tests/test_image_detector.py. Don't touch anything else.

    ## When You're in Over Your Head

    Stop and escalate if the existing learner.py code doesn't match the
    brief's "find this" blocks.

    ## Before Reporting Back: Self-Review

    Verify: does record_feedback() still work identically for every existing
    caller that doesn't care about corrections_dir? Does the new
    corrections_dir constructor parameter default correctly to CORRECTIONS_DIR
    when not supplied?

    ## Report Format

    Write your full report to .superpowers/sdd/task-3-report.md (same
    structure as prior tasks).

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-3-brief.md`, report `.superpowers/sdd/task-3-report.md`), same reviewer template as Task 1.

---

## Task 4: `image_detector/retrain.py` — fine-tune, validate, promote-or-rollback

**Dependencies:** Task 1 (`core.corrections_store`) and Task 2 (`prepare_data(..., extra_samples=...)`) must both be complete and committed.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 4: image_detector/retrain.py"
  model: sonnet
  prompt: |
    You are implementing Task 4: image_detector/retrain.py -- fine-tune,
    validate, and promote-or-rollback a retrained checkpoint

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-4-brief.md
    It contains the full task text, including complete code for
    image_detector/retrain.py and image_detector/tests/test_retrain.py.

    ## Context

    This is the integration point for the feedback-retraining feature in the
    image_detector package. It depends on two already-complete, committed
    pieces:

    - core/corrections_store.py (Task 1): provides RetrainResult,
      append_checkpoint_log, count_pending, list_pending,
      next_checkpoint_version, promote_pending_to_archive,
      read_cumulative_samples, read_last_checkpoint_accuracy
    - image_detector/trainer.py's prepare_data(dataset_dir, ..., extra_samples=...)
      (Task 2): accepts a list of (Path, int) tuples to fold into training
      alongside the on-disk dataset

    image_detector/config.py already defines CORRECTIONS_DIR,
    CHECKPOINT_LOG_FILE, DATASET_DIR, DEFAULT_CHECKPOINT (the last two
    pre-existing, the first two added in Task 2).

    Critical behavioral contract: validation accuracy comes directly from
    trainer.train()'s own returned history["val_acc"] -- never compute or
    estimate accuracy any other way. A retrain promotes the candidate
    checkpoint only if that accuracy is >= the last value logged in
    CHECKPOINT_LOG.md (or unconditionally if the log has no rows yet). On
    rollback, delete the candidate file and leave the pending corrections
    queue completely untouched.

    The brief's tests monkeypatch module-level config constants (e.g.
    config.DATASET_DIR = tmp_path) AND the retrain module's own rebound
    copies (retrain.DATASET_DIR = config.DATASET_DIR) -- this is necessary
    because `from image_detector.config import DATASET_DIR` binds a local
    name at import time that a later `config.DATASET_DIR = ...` reassignment
    does not affect. Don't "simplify" this pattern away.

    ## Before You Begin

    If anything is unclear, ask now. In particular, if torchvision's
    pretrained ImageNet weights aren't cached locally and the test needs to
    download them, that's expected (this matches how the rest of this
    project's test suite already works, e.g. test_detector_predict) -- note
    it in your report if the download is slow, but don't treat it as a
    blocker.

    ## Your Job

    1. Create image_detector/retrain.py exactly as specified in the brief
    2. Create image_detector/tests/test_retrain.py exactly as specified
       (TDD: these are the failing tests, confirm they fail first with the
       expected ModuleNotFoundError, then implement)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test file while iterating (this task's tests do real,
    if tiny, model training -- expect each test to take real wall-clock
    time, not milliseconds); run the full suite once before committing.

    ## Code Organization

    Create exactly the two files named. This module intentionally does NOT
    import image_detector.benchmarks or image_detector.detector -- validation
    accuracy comes from the trainer's own history dict, not a separate
    benchmark pass. If you find yourself wanting to add either import,
    stop and reconsider against the brief first.

    ## When You're in Over Your Head

    Stop and escalate if: the existing prepare_data() signature doesn't
    match what Task 2 was supposed to produce (meaning Task 2 may not have
    landed correctly), or if real training behaves unexpectedly (e.g. a
    tiny 1-epoch fine-tune never produces a measurable val_acc difference
    between the rollback test's seeded 1.01 and a real candidate -- this
    is actually fine and expected, since any real accuracy is <= 1.0 and
    1.01 is deliberately unbeatable; don't "fix" the test by changing the
    seeded value).

    ## Before Reporting Back: Self-Review

    Verify the rollback path leaves config.DEFAULT_CHECKPOINT NOT existing
    on disk, and the promote path leaves pending corrections moved to
    archive/ with CHECKPOINT_LOG.md containing exactly one new row.

    ## Report Format

    Write your full report to .superpowers/sdd/task-4-report.md.

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-4-brief.md`, report `.superpowers/sdd/task-4-report.md`). **This reviewer dispatch should additionally be told:** "This diff includes real PyTorch training code executed in tests — confirm the implementer's report shows actual passing test output (not skipped/mocked), but do not re-run the training tests yourself (they're slow); a focused re-run is acceptable only for a specific, named correctness doubt." Add this one sentence to the reviewer prompt's `## Tests` section for Tasks 4, 7, and 10 specifically (the three `retrain.py` tasks) — every other task uses the unmodified Task 1 reviewer template.

---

## Task 5: `audio_detector` — config additions + `trainer.py` extra-samples support

**Dependencies:** None (parallel to Tasks 2-4, independent package).

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 5: audio_detector config + trainer.py extra-samples"
  model: haiku
  prompt: |
    You are implementing Task 5: audio_detector -- config additions +
    prepare_data_from_directory() extra-samples support

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-5-brief.md
    It contains the full task text, including the exact config additions and
    the complete replacement code for prepare_data_from_directory(), plus a
    note that this test may need a `import wave` added to the test file's
    imports if not already present -- check the top of
    audio_detector/tests/test_audio_detector.py first and add it if missing.

    ## Context

    This is part of the same larger feature as image_detector's Task 2
    (feedback-driven retraining), applied to the audio_detector package
    independently -- audio_detector has no cross-import relationship with
    image_detector, so nothing from that task is relevant here beyond the
    shared pattern. A later task (Task 7, not yours) will build
    audio_detector/retrain.py on top of the extra_audio_samples parameter
    you're adding here.

    Note: audio_detector's prepare_data_from_directory() currently returns
    (X, y) numpy arrays (feature vectors), unlike image_detector's
    prepare_data() which returns DataLoaders -- don't assume the same
    return shape, follow the brief's exact code.

    ## Before You Begin

    If anything is unclear, ask now.

    ## Your Job

    1. Add the three config constants to audio_detector/config.py exactly as
       specified
    2. Replace prepare_data_from_directory()'s body in audio_detector/trainer.py
       exactly as specified (TDD: write the brief's failing test first)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test while iterating; run the full suite once before
    committing.

    ## Code Organization

    Only modify audio_detector/config.py, audio_detector/trainer.py, and
    audio_detector/tests/test_audio_detector.py.

    ## When You're in Over Your Head

    Stop and escalate if the existing prepare_data_from_directory() code
    doesn't match what the brief describes finding there.

    ## Before Reporting Back: Self-Review

    Specifically verify: does prepare_data_from_directory(dataset_dir)
    called WITHOUT extra_audio_samples behave identically to before your
    change?

    ## Report Format

    Write your full report to .superpowers/sdd/task-5-report.md.

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-5-brief.md`, report `.superpowers/sdd/task-5-report.md`), same reviewer template as Task 1.

---

## Task 6: `audio_detector` — `learner.py` queues corrections

**Dependencies:** Task 1 and Task 5 complete and committed.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 6: audio_detector learner.py queues corrections"
  model: haiku
  prompt: |
    You are implementing Task 6: audio_detector -- learner.py queues
    corrections for retraining

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-6-brief.md
    It contains the full task text, including the exact import changes,
    __init__ replacement, and record_feedback() insertion point and content.

    ## Context

    Task 1 (already complete, committed to main) added core/corrections_store.py
    with queue_correction(). This task wires AudioSelfImprover.record_feedback()
    to call it. audio_detector/config.py already has CORRECTIONS_DIR defined
    (added in Task 5, already complete, committed to main).

    Note AudioSelfImprover's __init__ signature differs slightly from
    ImageSelfImprover's (no memory_dir parameter, only memory_file and
    calibration_file) -- follow the brief's exact replacement, don't copy
    ImageSelfImprover's shape from Task 3.

    The brief's test sets learner.corrections_dir as a direct attribute
    assignment after construction rather than via constructor kwarg, to
    specifically exercise that the attribute is read at call time, not
    cached -- this is intentional, not a simplification you should "fix" to
    match Task 3's test style.

    ## Before You Begin

    If anything is unclear, ask now.

    ## Your Job

    1. Change the imports block and __init__ exactly as specified
    2. Insert the new try/except block into record_feedback() at the exact
       location shown in the brief (TDD: write the brief's failing test first)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test while iterating; run the full suite once before
    committing.

    ## Code Organization

    Only modify audio_detector/learner.py and
    audio_detector/tests/test_audio_detector.py.

    ## When You're in Over Your Head

    Stop and escalate if the existing learner.py code doesn't match the
    brief's "find this" blocks.

    ## Before Reporting Back: Self-Review

    Verify record_feedback() still works identically for every existing
    caller, and the new corrections_dir constructor parameter defaults
    correctly to CORRECTIONS_DIR when not supplied.

    ## Report Format

    Write your full report to .superpowers/sdd/task-6-report.md.

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-6-brief.md`, report `.superpowers/sdd/task-6-report.md`), same reviewer template as Task 1.

---

## Task 7: `audio_detector/retrain.py` — fine-tune, validate, promote-or-rollback

**Dependencies:** Task 1 and Task 5 complete and committed.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 7: audio_detector/retrain.py"
  model: sonnet
  prompt: |
    You are implementing Task 7: audio_detector/retrain.py -- fine-tune,
    validate, and promote-or-rollback a retrained checkpoint

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-7-brief.md
    It contains the full task text, including complete code for
    audio_detector/retrain.py and audio_detector/tests/test_retrain.py.

    ## Context

    This is the audio-package counterpart to Task 4 (image_detector/retrain.py,
    already complete and committed to main by the time you run -- you do not
    need to read its code, but your task is structurally the same pattern
    applied to audio). It depends on:

    - core/corrections_store.py (Task 1, complete)
    - audio_detector/trainer.py's prepare_data_from_directory(dataset_dir, ...,
      extra_audio_samples=...) (Task 5, complete)

    Important and specific to audio: audio_detector currently has ZERO
    training data on disk and NO existing checkpoint at all (unlike
    image_detector). This means the "bootstrap" test case -- corrections
    alone, with no pre-existing dataset, producing the first-ever checkpoint
    -- is the REALISTIC default case for this package, not an edge case.
    Your brief's test_run_retrain_builds_first_checkpoint_from_corrections_alone
    test exercises exactly this.

    audio_detector/trainer.py's AudioDetectorTrainer.train(X, y, epochs, ...)
    already returns history["val_accuracy"] (a true held-out validation
    accuracy, pre-existing behavior, not something you're adding) -- use that
    directly, exactly as the brief shows. Do not add any separate benchmark
    or detector-reload step for validation.

    The brief's tests monkeypatch module-level config constants AND the
    retrain module's own rebound copies, same pattern as image_detector's
    Task 4 -- necessary because of how Python binds `from module import NAME`.

    ## Before You Begin

    If anything is unclear, ask now.

    ## Your Job

    1. Create audio_detector/retrain.py exactly as specified in the brief
    2. Create audio_detector/tests/test_retrain.py exactly as specified
       (TDD: confirm these fail first with the expected ModuleNotFoundError)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test file while iterating (real, if tiny, model training
    -- expect real wall-clock time per test); run the full suite once before
    committing.

    ## Code Organization

    Create exactly the two files named.

    ## When You're in Over Your Head

    Stop and escalate if prepare_data_from_directory()'s signature doesn't
    match what Task 5 was supposed to produce, or if AudioClassifierNet /
    AudioDetectorTrainer behave unexpectedly.

    ## Before Reporting Back: Self-Review

    Verify the rollback path leaves config.DEFAULT_AUDIO_CHECKPOINT NOT
    existing on disk, and the bootstrap-from-corrections-alone path
    successfully produces a checkpoint with zero on-disk training data.

    ## Report Format

    Write your full report to .superpowers/sdd/task-7-report.md.

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-7-brief.md`, report `.superpowers/sdd/task-7-report.md`), reviewer template as Task 1 **plus** the Task 4 addendum about real-training tests (see Task 4 section above).

---

## Task 8: `video_detector` — config additions + `trainer.py` extra-samples + validation split

**Dependencies:** None (parallel to Tasks 2-7, independent package). This is the one task in this runbook that changes established behavior (video's `train()` currently has no validation split at all) rather than being purely additive at the call-site level — flagged in the plan's own Global Constraints as needing care.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 8: video_detector config + trainer.py extra-samples + val split"
  model: sonnet
  prompt: |
    You are implementing Task 8: video_detector -- config additions,
    prepare_data_from_videos() extra-samples support, and a NEW optional
    held-out validation split in train()

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-8-brief.md
    It contains the full task text, including the exact config additions and
    complete replacement code for both prepare_data_from_videos() and train().

    ## Context

    This is part of the same larger feature as image_detector's Task 2 and
    audio_detector's Task 5, applied to video_detector. A later task
    (Task 10, not yours) will build video_detector/retrain.py on top of both
    changes here.

    The important part of this task: video_detector/trainer.py's existing
    train() method currently has NO validation split at all -- it trains on
    100% of the frame pairs passed to it, with no accuracy tracking
    (history = {"loss": []} only). This is being changed because Task 10's
    retrain.py needs a real held-out validation accuracy to decide whether a
    candidate checkpoint is safe to promote, and image_detector/audio_detector
    both already have this (their trainers already compute history["val_acc"]
    / history["val_accuracy"]). This is the one task in the whole plan that
    changes EXISTING behavior rather than being purely additive at the
    call-site level.

    The backward-compatibility contract is strict: the new val_split
    parameter defaults to 0.0, and at that default, the method's behavior
    (train on all pairs, no held-out set, history["loss"] populated exactly
    as before) must be byte-for-byte unchanged from before your edit. The
    ONLY observable difference at the default is that history now always
    also contains a "val_accuracy" key, which will be an empty list []　when
    val_split=0.0. Check whether any existing test or caller asserts the
    exact shape of the returned history dict (e.g. assertEqual(history,
    {"loss": [...]}) rather than assertIn("loss", history)) -- if you find
    one, it will break from the added key, and you should report this in
    your self-review rather than silently changing the test's assertion
    style without flagging it.

    ## Before You Begin

    If anything is unclear, ask now. In particular, if you find existing
    code elsewhere in the repo calling VideoDetectorTrainer.train() and
    relying on the exact old history shape, flag it before proceeding --
    don't silently patch call sites outside this task's two files.

    ## Your Job

    1. Add the three config constants to video_detector/config.py exactly as
       specified
    2. Replace prepare_data_from_videos()'s body exactly as specified
    3. Replace train()'s body exactly as specified, adding the val_split
       parameter (TDD: write the brief's two failing tests first)
    4. Verify implementation works
    5. Commit your work
    6. Self-review
    7. Report back

    Work fro<repo>

    Run the focused tests while iterating; run the full suite once before
    committing -- this is the task where a full-suite run matters most,
    since train()'s behavior change could silently affect something outside
    this task's own new tests.

    ## Code Organization

    Only modify video_detector/config.py, video_detector/trainer.py, and
    video_detector/tests/test_video_detector.py.

    ## When You're in Over Your Head

    Stop and escalate if the existing prepare_data_from_videos() or train()
    code doesn't match what the brief describes finding there, or if you
    find an existing caller of train() whose behavior your change would
    alter.

    ## Before Reporting Back: Self-Review

    Specifically verify: with val_split=0.0 (the default), is
    history["val_accuracy"] always [] and is history["loss"]'s content
    identical to what the old code would have produced for the same input?
    Did you check for existing callers relying on the old exact history
    shape?

    ## Report Format

    Write your full report to .superpowers/sdd/task-8-report.md.

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-8-brief.md`, report `.superpowers/sdd/task-8-report.md`). **Reviewer addendum for this task specifically:** add to the reviewer prompt's `## Part 2: Code Quality` section: "This task modifies an existing method's behavior (video_detector/trainer.py's train()), not just adds a new one. Specifically verify the backward-compatibility claim: with val_split=0.0, does the diff show history["loss"] computed identically to the pre-change code? Check this against the diff's removed/added lines directly."

---

## Task 9: `video_detector` — `learner.py` queues corrections

**Dependencies:** Task 1 and Task 8 complete and committed.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 9: video_detector learner.py queues corrections"
  model: haiku
  prompt: |
    You are implementing Task 9: video_detector -- learner.py queues
    corrections for retraining

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-9-brief.md
    It contains the full task text, including the exact import changes,
    __init__ replacement, and record_feedback() insertion point and content.

    ## Context

    Task 1 (already complete, committed to main) added core/corrections_store.py
    with queue_correction(). This task wires VideoSelfImprover.record_feedback()
    to call it. video_detector/config.py already has CORRECTIONS_DIR defined
    (added in Task 8, already complete, committed to main).

    VideoSelfImprover's __init__ shape matches ImageSelfImprover's (both take
    memory_dir, memory_file, calibration_file) -- Task 3's pattern for
    image_detector is a reasonable reference for the shape, but follow this
    brief's exact code, don't copy Task 3's diff verbatim since the
    surrounding file content differs.

    Note video_detector's record_feedback() signature is more complex than
    image/audio's -- it accepts video_metrics, temporal_metrics, notes, and
    **kwargs, with video_path as the file-path parameter (not image_path or
    audio_path). Use video_path for the file read.

    ## Before You Begin

    If anything is unclear, ask now.

    ## Your Job

    1. Change the imports block and __init__ exactly as specified
    2. Insert the new try/except block into record_feedback() at the exact
       location shown in the brief (TDD: write the brief's failing test first)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test while iterating; run the full suite once before
    committing.

    ## Code Organization

    Only modify video_detector/learner.py and
    video_detector/tests/test_video_detector.py.

    ## When You're in Over Your Head

    Stop and escalate if the existing learner.py code doesn't match the
    brief's "find this" blocks.

    ## Before Reporting Back: Self-Review

    Verify record_feedback() still works identically for every existing
    caller, and the new corrections_dir constructor parameter defaults
    correctly to CORRECTIONS_DIR when not supplied.

    ## Report Format

    Write your full report to .superpowers/sdd/task-9-report.md.

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-9-brief.md`, report `.superpowers/sdd/task-9-report.md`), same reviewer template as Task 1.

---

## Task 10: `video_detector/retrain.py` — fine-tune, validate, promote-or-rollback

**Dependencies:** Task 1 and Task 8 complete and committed (Task 8 specifically, since this task relies on both the new `extra_video_samples` parameter and the new `val_split` parameter it added).

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 10: video_detector/retrain.py"
  model: sonnet
  prompt: |
    You are implementing Task 10: video_detector/retrain.py -- fine-tune,
    validate, and promote-or-rollback a retrained checkpoint

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-10-brief.md
    It contains the full task text, including complete code for
    video_detector/retrain.py and video_detector/tests/test_retrain.py.

    ## Context

    This is the video-package counterpart to Task 4 (image_detector/retrain.py)
    and Task 7 (audio_detector/retrain.py), both already complete and
    committed to main by the time you run. It depends on:

    - core/corrections_store.py (Task 1, complete)
    - video_detector/trainer.py's prepare_data_from_videos(dataset_dir, ...,
      extra_video_samples=...) AND train(pairs, ..., val_split=...) (Task 8,
      complete) -- Task 8 added BOTH of these; this task is the reason Task 8
      needed to add a validation split to train() at all, since train()
      previously had none.

    Like audio_detector, video_detector currently has ZERO training data on
    disk and NO existing checkpoint. The bootstrap test case (corrections
    alone producing the first-ever checkpoint) is the realistic default case.

    Call train() with val_split=0.2 explicitly (the brief shows this) --
    video's train() defaults to val_split=0.0 for backward compatibility
    (Task 8), so retrain.py must pass it explicitly to get a real validation
    accuracy out of history["val_accuracy"].

    The brief's tests monkeypatch module-level config constants AND the
    retrain module's own rebound copies, same pattern as Tasks 4 and 7.

    ## Before You Begin

    If anything is unclear, ask now.

    ## Your Job

    1. Create video_detector/retrain.py exactly as specified in the brief
    2. Create video_detector/tests/test_retrain.py exactly as specified
       (TDD: confirm these fail first with the expected ModuleNotFoundError)
    3. Verify implementation works
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run the focused test file while iterating (real, if tiny, model training
    on synthetic video frames via cv2.VideoWriter -- expect real wall-clock
    time per test); run the full suite once before committing.

    ## Code Organization

    Create exactly the two files named.

    ## When You're in Over Your Head

    Stop and escalate if prepare_data_from_videos() or train()'s signatures
    don't match what Task 8 was supposed to produce, or if
    VideoTemporalTransitionModel / VideoDetectorTrainer behave unexpectedly.

    ## Before Reporting Back: Self-Review

    Verify the rollback path leaves config.DEFAULT_VIDEO_CHECKPOINT NOT
    existing on disk, and the bootstrap-from-corrections-alone path
    successfully produces a checkpoint with zero on-disk training data.
    Confirm you passed val_split=0.2 to train() -- without it,
    history["val_accuracy"] would be empty and the promote/rollback decision
    would be meaningless.

    ## Report Format

    Write your full report to .superpowers/sdd/task-10-report.md.

    Report back with ONLY (under 15 lines): Status, commits, one-line test
    summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-10-brief.md`, report `.superpowers/sdd/task-10-report.md`), reviewer template as Task 1 **plus** the Task 4 addendum about real-training tests, **plus** one more check specific to this task: add to the reviewer's Part 1 Spec Compliance checks: "Confirm run_retrain() calls trainer.train(pairs, epochs=epochs, val_split=0.2) with val_split explicitly passed — video's train() defaults to val_split=0.0, so omitting this argument would silently produce an empty val_accuracy list and break the promote/rollback decision. Check this against the diff directly."

---

## Task 11: Git LFS — track `models/*.pt` instead of ignoring it

**Dependencies:** None strictly, but logically should run after Task 4 (so there's an actual `image_detector/models/ai_detector.pt` worth tracking — it already exists on disk regardless, so this ordering is about narrative clarity, not a hard technical dependency).

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 11: Git LFS tracking for models/*.pt"
  model: sonnet
  prompt: |
    You are implementing Task 11: Git LFS -- track models/*.pt instead of
    gitignoring it

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-11-brief.md
    It contains the full task text with exact commands and verification
    steps for every step.

    ## Context

    This task is pure repository configuration (.gitattributes, .gitignore),
    not application code. Its purpose: today models/*.pt is blanket-ignored
    by .gitignore's existing *.pt rule, so a fresh clone of this repo never
    gets the trained checkpoint and the detector silently falls back to
    heuristic-only mode. This task makes models/*.pt trackable via Git LFS
    instead, so a clone + "git lfs pull" reproduces the exact same checkpoint
    bytes (and therefore the exact same detection results) without ever
    receiving the personal media that trained it.

    This is Windows + Git Bash (per this project's established toolchain --
    confirmed by prior session git commands in this repo using bash syntax
    successfully). Use bash-compatible commands as the brief shows.

    ## Before You Begin

    If `git lfs version` fails (Git LFS not installed on this machine), STOP
    and report BLOCKED immediately -- do not attempt to install it yourself,
    this requires the human to install Git LFS first
    (https://git-lfs.com).

    If anything else about the sequence is unclear, ask now.

    ## Your Job

    Follow the brief's 7 steps exactly, in order:
    1. Verify git lfs version
    2. git lfs install
    3. Create .gitattributes
    4. Edit .gitignore (exact find/replace shown in brief)
    5. Verify via git check-ignore and git lfs track that the negation works
    6. Stage and commit .gitattributes, .gitignore, AND
       image_detector/models/ai_detector.pt together
    7. Run pytest to confirm nothing else broke

    Work fro<repo>

    This task has no new Python tests -- "testing" here means following the
    brief's verification commands exactly and confirming their expected
    output, plus the final full pytest run.

    ## Code Organization

    Only create/modify .gitattributes and .gitignore, and stage the one
    existing checkpoint file as instructed. Do not touch any Python source.

    ## When You're in Over Your Head

    Stop and escalate (BLOCKED) if: Git LFS isn't installed, the gitignore
    negation doesn't work as expected (git check-ignore still reports the
    path as ignored after your edit), or `git lfs track` doesn't show *.pt
    as a tracked pattern after creating .gitattributes.

    ## Before Reporting Back: Self-Review

    Confirm `git check-ignore -v image_detector/models/ai_detector.pt`
    produces NO output and a non-zero exit code (meaning no longer ignored)
    -- this is the single most important verification in this task.

    ## Report Format

    Write your full report to .superpowers/sdd/task-11-report.md, including
    the actual output of each verification command from the brief (git lfs
    version, git lfs install, git check-ignore -v, git lfs track, git status
    after staging).

    Report back with ONLY (under 15 lines): Status, commits, one-line
    verification summary, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-11-brief.md`, report `.superpowers/sdd/task-11-report.md`). **Reviewer note for this task:** the diff for this task includes a binary `.pt` file via LFS — the reviewer should treat this as expected (not flag "large binary committed" as an issue) and focus review on whether `.gitattributes`/`.gitignore` are correct and whether the implementer's reported verification command output actually shows the negation working.

---

## Task 12: UI integration — "Retrain Now" in the Continuous Learning dashboard

**Dependencies:** Tasks 2, 4 (image), 5, 7 (audio), 8, 10 (video) — all three `retrain.py` modules and their `config.RETRAIN_SUGGEST_THRESHOLD` constants must exist.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 12: UI Retrain Now integration"
  model: sonnet
  prompt: |
    You are implementing Task 12: ui/feedback_ui.py -- surface queued
    corrections and a Retrain Now button in the Continuous Learning dashboard

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-12-brief.md
    It contains the full task text, including the exact import changes and
    the exact "find this code / replace with this code" diff for
    render_learning_dashboard().

    ## Context

    By the time you run, all three retrain.py modules exist and are
    committed to main:
    - image_detector.retrain.count_pending_corrections() -> Dict[str, int]
    - image_detector.retrain.run_retrain(min_new=1) -> RetrainResult
    - (same two functions on audio_detector.retrain and video_detector.retrain)
    - image_detector.config.RETRAIN_SUGGEST_THRESHOLD (and the audio/video
      equivalents) -- an int, default 15

    RetrainResult (from core.corrections_store) has fields: promoted: bool,
    new_accuracy: float, samples_folded_in: int, cumulative_samples: int,
    message: str, old_accuracy: Optional[float].

    This task modifies ui/feedback_ui.py, which per this project's existing
    convention has NO automated tests (every other render_* function in this
    file is verified by manually running the Streamlit app, not pytest) --
    your verification step is running the app, not writing a test file.

    ## Before You Begin

    If anything is unclear, ask now. In particular, if the existing imports
    block or render_learning_dashboard() code doesn't match what the brief
    describes finding there, stop and report NEEDS_CONTEXT.

    ## Your Job

    1. Add the new imports exactly as specified
    2. Insert the new "Retrain on Corrections" section into
       render_learning_dashboard() at the exact location shown in the brief
    3. Manually verify by running the Streamlit app (see brief's Step 3 for
       the exact verification procedure: run streamlit run app.py, check
       the Continuous Learning tab renders with no corrections queued,
       submit one correction via the Image tab's feedback panel, confirm
       the banner and button appear, click it, confirm either a success or
       warning message renders)
    4. Commit your work
    5. Self-review
    6. Report back

    Work fro<repo>

    Run `pytest` once before committing to confirm nothing else broke (this
    task adds no new automated tests, but existing ones must still pass).

    ## Code Organization

    Only modify ui/feedback_ui.py.

    ## When You're in Over Your Head

    Stop and escalate if the Streamlit app fails to start, or if clicking
    "Retrain Now" produces an unhandled exception rather than either a
    success or warning message -- that would indicate a real integration
    bug between the UI and one of the three retrain.py modules.

    ## Before Reporting Back: Self-Review

    Confirm you actually ran the app and saw the banner + button appear and
    function, not just that the code looks plausible -- paste the key
    observations (not full screenshots, just what you saw) into your report.

    ## Report Format

    Write your full report to .superpowers/sdd/task-12-report.md, including
    what you observed during manual Streamlit verification (not just "it
    worked" -- describe what rendered at each step).

    Report back with ONLY (under 15 lines): Status, commits, one-line
    verification summary (manual + pytest), concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-12-brief.md`, report `.superpowers/sdd/task-12-report.md`). **Reviewer note for this task:** since there's no automated test diff to inspect for this task's core behavior, the reviewer's Part 1 Spec Compliance should weight the implementer's reported manual-verification observations more heavily than usual — but still flag it as a ⚠️ Cannot verify from diff if the report's manual-verification description is vague or missing concrete observations (e.g. just "it worked" with no detail).

---

## Task 13: README updates + final end-to-end verification

**Dependencies:** All prior tasks (1-12) complete — this task documents the finished feature and does a final whole-suite sanity check.

### Implementer dispatch

```
Subagent (general-purpose):
  description: "Implement Task 13: README updates + final verification"
  model: haiku
  prompt: |
    You are implementing Task 13: README.md updates + final end-to-end
    verification

    ## Task Description

    Read your task brief first: .superpowers/sdd/task-13-brief.md
    It contains the full task text, including the exact "find this text /
    replace with this text" edits for README.md.

    ## Context

    This is the final task of the feedback-driven retraining pipeline
    feature. All code (Tasks 1-12) is already implemented and committed to
    main. This task only updates documentation and runs a final
    verification -- no source code changes.

    ## Before You Begin

    If the exact text the brief says to find in README.md doesn't match
    what's actually there, stop and report NEEDS_CONTEXT rather than
    guessing where to make the edit.

    ## Your Job

    1. Make the two README.md edits exactly as specified in the brief
    2. Run `pytest -v` and confirm every test across core/tests,
       image_detector/tests, audio_detector/tests, video_detector/tests,
       and tests/ passes
    3. Run `git status` and confirm only README.md is modified/unstaged at
       this point (everything else should already be committed from prior
       tasks)
    4. Commit your work
    5. Report back

    Work fro<repo>

    ## Code Organization

    Only modify README.md.

    ## When You're in Over Your Head

    Stop and escalate if git status shows anything unexpected uncommitted
    (that would mean an earlier task didn't fully commit its work) -- do not
    try to clean this up yourself, report it.

    ## Report Format

    Write your full report to .superpowers/sdd/task-13-report.md, including
    the full pytest -v output summary (pass count) and the git status
    output.

    Report back with ONLY (under 15 lines): Status, commits, final pytest
    pass count, concerns, report file path.
```

### Controller steps + Task Reviewer dispatch

Same mechanical pattern (model: `sonnet`, brief `.superpowers/sdd/task-13-brief.md`, report `.superpowers/sdd/task-13-report.md`), same reviewer template as Task 1 — but note this reviewer's job is almost entirely "did the README text match the brief, and does the implementer's reported final pytest count look complete" rather than code-quality judgment, since there's no source diff.

---

## After Task 13: Final Whole-Branch Review

Once all 13 tasks show `review clean` in the progress ledger:

1. Find the merge-base: since this ran directly on `main` with no branch, the "start of the branch" is the commit immediately before Task 1's implementer was first dispatched — record this SHA at the very start of execution (before Task 1) as `RUNBOOK_START_SHA`, alongside creating the progress ledger.
2. Run: `"<home>/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/subagent-driven-development/scripts/review-package" RUNBOOK_START_SHA HEAD` — note the printed path.
3. Dispatch the final reviewer using `superpowers:requesting-code-review`'s `code-reviewer.md` template, **model: `opus`** (most capable, per this skill's Model Selection), pointing it at the printed diff package path and this runbook's Global Constraints block. Include in its prompt: "This diff spans 13 tasks implementing a feedback-driven retraining pipeline across core/, image_detector/, audio_detector/, and video_detector/. Each task already passed an individual task-scoped review; this is the broad, cross-cutting pass — look specifically for: inconsistencies BETWEEN the three per-modality retrain.py implementations that individual task reviews couldn't see (e.g. did Task 10 actually pass val_split=0.2 the way Task 4 and 7's packages didn't need to? do all three CHECKPOINT_LOG.md files end up with a consistent column format?); whether the Git LFS checkpoint (Task 11) and the retrain.py promotion logic (Tasks 4/7/10) interact correctly (does a promoted checkpoint actually end up trackable, not re-ignored?); and whether the UI integration (Task 12) correctly reflects all three modalities' actual function signatures."
4. If the final review finds issues: dispatch **one** fix subagent (not one per finding) with the complete findings list, re-run the review package, re-dispatch the final reviewer.
5. Once clean, append to the ledger: `Final whole-branch review: complete (clean)`.

## After Final Review: Finishing

Invoke `superpowers:finishing-a-development-branch`. Since this work was done directly on `main` (not a feature branch), expect that skill to present options accordingly (e.g. "already on main, nothing to merge" rather than its usual branch-merge flow) — follow its guidance for that case rather than assuming a PR/merge step is needed.
