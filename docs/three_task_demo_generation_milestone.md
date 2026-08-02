# Three-task deterministic demo-generation milestone

This milestone freezes the latest accepted code-only strategies for
PutPotOnCooktop, PutMarkerInDrawer, and HangMugOnTree, and defines how they are
scaled to forty official asset pairs per task.

## Acceptance contract

Every accepted output is one uninterrupted rollout with exactly one initial
scene reset and no inter-stage reset or state teleport. The existing coded task
success predicate is authoritative. A successful pair must include:

- the exact official target-asset paths and source/target dataset hashes;
- all executed actions and full simulator scene states (`T` actions, `T+1` states);
- numeric policy/proprioceptive and semantic observations for every action;
- a continuous, fully decodable H.264 MP4;
- a result receipt containing stages, terminal predicates, metrics, parameters,
  physics device, assistance configuration, and artifact hashes.

RGB is stored in the MP4 rather than duplicated inside HDF5. The HDF5 remains
compatible with the project's `data/demo_0/{actions,states,initial_state,obs}`
layout and is rejected unless its success attribute is true.

## Frozen strategies

### PutPotOnCooktop

Use the source demonstration to identify bimanual handle acquisition and the
cooktop support frame. When replay already reaches stable support but misses the
stricter center gate, preserve its contact-rich prefix and perform only a
right-held supported center repair before release. For replay failures, execute
one smooth collision-aware bimanual semantic transport, lower onto support, use
closed-loop centering feedback, release, and validate a stable centered terminal
pot. No grasp assistance or sampled optimization is used.

### PutMarkerInDrawer

Use sparse semantic keyframes for marker grasp, drawer-handle acquisition,
opening, cavity placement, release, drawer closure, and handle release. Transfer
drawer axis, handle, and cavity geometry to the target pair and track the
deterministic program with Cartesian DLS plus the calibrated rigid handle pull.
No MPC or candidate sampling is used.

### HangMugOnTree

Transfer the left grasp, physical handover, handle-hole, branch tangent/support,
and left-wrist observer frames. Use the datagen-supported fingertip friction
assist for the left pickup, release the assist during handover, align the handle
beyond the target branch tip, insert inward along the branch tangent, then
release and validate stable support. No MPC or candidate sampling is used.

## Forty-pair campaign loop

For each official pair:

1. Run the source actions once in a fresh target scene as a classification run.
2. If the coded task predicate and stricter task-specific terminal gates pass,
   retain that uninterrupted rollout as the successful target demonstration.
3. Otherwise run the frozen deterministic semantic skill against the same
   target, with the failed replay receipt attached as evidence.
4. Validate the demonstration HDF5, video, result, exact assets, and hashes.
5. Atomically mark the pair accepted in the task ledger. Accepted pairs are
   revalidated before resume and are never silently rerun.
6. Continue through all pairs even if adaptation fails. Failure clusters are
   inputs to the next code revision; a task is terminal only at `40/40`.

The campaign runner is `examples/run_three_task_asset_campaign.py`; its exact
dataset sets and devices are declared in
`configs/three_task_40_asset_campaign.json`. It deliberately runs one Isaac
process at a time to avoid hidden resource contention and preserves both failed
source-replay evidence and the final successful adapted rollout.

Example:

```bash
export PUTPOT_DATA_ROOT=/path/to/PutPotOnCooktop
export PUTMARKER_DATA_ROOT=/path/to/PutMarkerInDrawer/random40_0519
export HANGMUG_DATA_ROOT=/path/to/HangMugOnTree

python examples/run_three_task_asset_campaign.py \
  --config configs/three_task_40_asset_campaign.json \
  --gear-repo /path/to/gear-dc-study \
  --output-root /path/to/three_task_40_asset_results
```

Use `--dry-run` to verify enumeration and all generated commands without
starting Isaac. Use repeated `--task putpot|putmarker|hangmug` to scope a run.
