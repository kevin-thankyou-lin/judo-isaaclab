# Evidence-driven coding-agent asset adaptation

The source demonstration proposes a semantic strategy. It is not an exact joint
trajectory for every asset instance. The coding agent implements task-space
skills, while this harness owns evidence, sequencing, and fail-closed acceptance.

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

1. Read the coded task-success predicate and stage latches.
2. Extract semantic frames from the source demo and asset geometry.
3. Implement small primitives such as `grasp`, `transport`, `insert/place`,
   `release`, and `close`.
4. Prove the source skill before changing assets.
5. Rank target candidates geometrically, then run replay until a validated
   failure target is found.
6. Change only the primitive implicated by the latest evidence.
7. Keep every attempt in the ledger with command, revision, asset identity,
   metrics, checks, and artifact hashes.
8. Finish with one reset, no inter-stage teleport, coded success, full video
   decode, visual inspection, tests, clean worktrees, and matching remote hash.

## Diagnosis policy

- Large EEF error: reachability/controller problem. Use target-relative IK or a
  different waypoint.
- Accurate endpoint but failed contact: interaction-path or grasp-transform
  problem. Preserve the contact transform and move along the articulation or
  insertion axis.
- Failed final placement: adjust support-frame clearance, release timing, and
  the stable-support window.
- Successful target replay: reject the target as adaptation evidence.
- Missing provenance or artifacts: repair infrastructure before motion code.

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
