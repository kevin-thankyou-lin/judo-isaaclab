# Flexible semantic execution boundary

Semantic geometry and control remain task-owned.  The shared execution layer
provides instrumentation and optional observation events without requiring a
universal `grasp -> transport -> release` state machine.

```text
campaign harness     sequencing, artifacts, immutable ledgers
task runner          semantic frames, controls, reanchoring, task-specific gates
task manager         authoritative stage and final-success predicates
```

## Measured protocol receipt

`SemanticProtocolRecorder` records runner-visible operations:

- task-environment resets and their reasons;
- the initial target-dataset state restoration;
- any state write or reset after rollout start;
- every environment step and its termination/truncation flags; and
- aggregate grasp-detector observations with their named source.

The receipt states its measurement scope as
`runner_operations_and_env_step_flags`.  It does not claim to observe hidden
simulator mutations.  Existing compatibility fields such as `scene_resets`,
`inter_stage_resets`, and `teleports_after_reset` are now derived from the same
receipt instead of being literal constants.

## Optional events

`SemanticExecutionHooks` emits read-only lifecycle observations:

- `rollout_start`;
- `before_step` and `after_step`;
- task-authored waypoint `milestone`; and
- `rollout_end`.

An empty hook set is the default and is a no-op.  Callbacks cannot replace the
action returned by the task runner, so adding instrumentation does not change
the delivered controller.  A future task adapter may consume these events for
diagnosis or propose a separately reviewed readiness/reanchoring policy without
moving task semantics into the generic campaign harness.

## Compatibility rule

The shared layer must not define task phases, contact allowances, geometry
thresholds, or success.  Those remain in task adapters and the authoritative
task manager.  Fixed-duration open-loop programs remain supported, while tasks
may opt into additional state-aware logic in their own runner.
