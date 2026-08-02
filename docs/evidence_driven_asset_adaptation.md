# Evidence-driven coding-agent asset adaptation

The source demonstration proposes a semantic strategy. It is not an exact joint
trajectory for every asset instance. The coding agent implements task-space
skills, while this harness owns evidence, sequencing, and fail-closed acceptance.

The agent does not directly author an accepted demonstration. It writes and
repairs a semantic skill program; the simulator produces the demonstration only
after the uninterrupted rollout passes the existing coded task predicate.

```text
source demo + task predicate + asset geometry
                    |
          semantic strategy and frames
                    |
          deterministic skill program
                    |
       continuous target-asset simulation
                    |
     numeric checks + rendered diagnosis
                    |
        accepted recorded demonstration
```

## Design principles

- Start from the task predicate. Preserve its thresholds and stable windows.
- Use the demonstration for subtask order, ownership, contact intent, and
  object-relative relationships—not as a target joint trajectory.
- Express targets in semantic frames: grasp contacts, support surfaces, handle
  holes, branch tangents, drawer cavities, and articulation axes.
- Keep skills explicit and editable: `grasp`, `handover`, `transport`,
  `insert/place`, `release`, `close`, and `settle`.
- Diagnose with both state traces and synchronized cameras. Rendered evidence
  reveals wrong-side approaches, occlusion, collisions, and unstable support;
  coded success remains the final authority.
- Test a repaired stage locally, but accept only a one-reset, uninterrupted
  end-to-end rollout. Mid-contact resets are not valid substitutes for natural
  contact history.
- Treat task completion and motion quality as separate gates. A successful
  path can still need shorter motion, fewer internal stops, better clearance,
  or a more useful observer view.

## Required proof

The final claim requires four accepted runs:

1. `source_skill`: the coded skill succeeds on the source assets.
2. `target_replay`: source actions fail on the selected target assets.
3. `target_skill`: the adapted coded skill succeeds on that same target.
4. `final_render`: the same target skill revision succeeds in one continuous,
   technically valid rendered rollout.

The target-replay failure matters. If source replay already succeeds, the target
is useful regression coverage but does not prove adaptation.

## Agent loop

1. Audit the authoritative task manager, stage latches, assets, dataset, and a
   previously proven analogous skill before editing code.
2. Extract the source strategy and semantic frames from the demonstration and
   collision geometry.
3. Implement a continuous graph of small deterministic primitives such as
   `grasp`, `handover`, `transport`, `insert/place`, `release`, and `close`.
4. Prove the coded skill on the source assets.
5. Rank target candidates geometrically and run unchanged source actions until
   a validated replay-failure target is found.
6. Execute the skill on that target and classify the first failed predicate as
   provenance, correspondence, contact, reachability/tracking, clearance,
   release/support, or acceptance instrumentation.
7. Change only the implicated primitive. Do not weaken a threshold or disturb
   stages that already pass.
8. Keep every attempt in the append-only ledger with its command, revision,
   asset identity, parameters, reset/teleport counts, stage events, metrics,
   checks, rejection reason, and artifact hashes.
9. Finish with one reset, no inter-stage teleport, coded success, full video
   decode, visual inspection, tests, clean worktrees, and matching remote hash.

The final demonstration must retain actions, states, observations, asset and
dataset hashes, controller parameters, stage/success traces, and the exact code
revision. A filename or successful process exit is not acceptance evidence.

## Diagnosis policy

- Large EEF error: reachability/controller problem. Use target-relative IK or a
  different waypoint.
- Accurate endpoint but failed contact: interaction-path or grasp-transform
  problem. Preserve the contact transform and move along the articulation or
  insertion axis.
- Failed final placement: adjust support-frame clearance, release timing, and
  the stable-support window.
- Correct task outcome but poor motion: preserve the successful interaction and
  optimize a separate quality gate such as path length, internal stops,
  clearance, duration, or observer visibility. Do not claim lower jerk unless
  it is actually measured.
- Successful target replay: reject the target as adaptation evidence.
- Missing provenance or artifacts: repair infrastructure before motion code.

## Scaling beyond one target

Run unchanged source actions over many same-category instances, retain genuine
replay failures, and group them by the first failed primitive. Adapt one
representative from each failure group, promote the correction into
geometry-conditioned skill defaults, and rerun every solved pair as a
regression suite. Hold out additional pairs to measure whether the skill—not an
asset identifier or memorized action sequence—generalizes.

## PutPot deployment

`configs/putpot_evidence_agent.json` defines the first deployment. Its semantic
frames are the two pot handle contacts, the pot bottom support plane, and the
cooktop top support frame. The coding-only strategy is:

```text
bimanual handle grasp
        -> lift and transport
        -> align pot support frame over cooktop support frame
        -> lower, unload, release, and settle
```

The authoritative success check remains the task's existing two-stage predicate:
latched bimanual pick followed by released, stable on-top placement.

## HangMug deployment

`configs/hangmug_evidence_agent.json` defines the HangMug proof. Its semantic
frames are the left and right grasp contacts, mug handle hole, branch tip and
tangent, support normal, and a left-wrist observer pose. The coding-only
strategy is:

```text
left handle grasp
        -> physical right-hand handover
        -> smooth transport while the left wrist moves to observe the branch
        -> align the handle hole beyond the branch tip
        -> insert inward along the branch tangent
        -> unload, release, and validate stable support
```

The observer pose is copied from the source insertion view in the source-tree
frame, transferred into the target-tree frame, and reached during transport.
The left arm then stays at that pose throughout final alignment, insertion,
release, and support validation. This makes the target branch visible without
changing the successful right-hand path or treating camera visibility as a
task-success substitute.

## CLI

Initialize a ledger:

```bash
export PUTPOT_DATA_ROOT=/path/to/PutPotOnCooktop
python examples/evidence_adaptation_harness.py \
  --bundle configs/putpot_evidence_agent.json \
  --workspace /tmp/putpot_adaptation init
```

Run one foreground simulator attempt and ingest it atomically:

```bash
python examples/evidence_adaptation_harness.py \
  --bundle configs/putpot_evidence_agent.json \
  --workspace /tmp/putpot_adaptation run \
  --phase target_skill \
  --result /tmp/putpot_target_skill.json \
  --log /tmp/putpot_target_skill.log \
  --trace /tmp/putpot_target_skill.npz \
  --revision "$(git rev-parse HEAD)" \
  --source-id putpot_000 \
  --target-id putpot_017 -- \
  python examples/run_putpot_skill_program.py ...
```

`verify` exits nonzero until all four proof runs exist, share source/target
identity, and the final render uses the accepted target-skill revision.
