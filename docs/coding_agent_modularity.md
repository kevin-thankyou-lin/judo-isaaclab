# Coding-agent modularity budget

Iterative simulator repair must not turn the task runner into an archive of
every attempted controller.  Git history, append-only attempt receipts, and
immutable artifacts preserve experiments.  Production code retains only the
active strategy and reusable, evidence-backed primitives.

## Responsibility boundaries

- CLI runner: argument parsing, component assembly, execution, and receipt writing.
- Semantic geometry: object-local frames and geometry-conditioned relationships.
- Controller strategy: task-specific action generation behind a narrow interface.
- Diagnostics: render overlays, trace analysis, and failure classification.
- Runtime: simulator lifecycle, worker protocol, and shutdown handling.
- Evidence: artifact serialization, hashes, and acceptance receipts.

Tests mirror those boundaries.  New repair families belong in separate strategy
modules rather than conditionals inside a runner.  After a repair succeeds,
distill the minimum reusable behavior and remove superseded production paths.

## Ratcheted enforcement

Run this before and after each repair:

```bash
python scripts/check_modularity_budget.py
```

`configs/modularity_budget.json` pins the revision at which enforcement was
introduced.  New production modules may not exceed 1,200 lines, new functions
may not exceed 250 lines, and a top-level `main()` may not exceed 200 lines.
Warnings begin at 800 module lines and 100 function lines.  Tests use a separate
1,500-line module budget.

Legacy violations are allowed only at or below their pinned baseline size.  A
future change cannot grow an inherited oversized file or function: it must first
extract a coherent responsibility.  Net production growth above 200 lines is a
review warning because a multi-module feature can legitimately be larger.

These are decomposition gates, not limits on task sophistication.  Controllers
can remain flexible through composable strategy modules and explicit interfaces.

## Pasteable coding-agent requirement

```text
Maintain a modularity budget throughout implementation.

Before editing:
- run `python scripts/check_modularity_budget.py`
- report line counts for task-specific runners, source modules, and tests
- identify the responsibility boundary for each planned change

During implementation:
- do not add task logic directly to a large CLI/main function
- keep CLI runners limited to parsing, assembly, execution, and receipt writing
- separate semantic geometry, controller strategies, diagnostics, runtime
  lifecycle, and evidence serialization
- isolate experimental repairs behind a common strategy interface
- preserve obsolete experiments in Git and immutable artifacts, not production code
- if a production change adds over 200 lines, review decomposition before proceeding

Before finishing:
- rerun `python scripts/check_modularity_budget.py`
- report final line counts and largest functions
- run architectural and behavioral regression tests
- verify extraction did not alter the action trace or coded success predicate
```

For an already active campaign, first freeze a known-good action trace, receipt,
and artifact hashes.  Refactor in an isolated branch and compare those behavioral
receipts before promoting the decomposition.
