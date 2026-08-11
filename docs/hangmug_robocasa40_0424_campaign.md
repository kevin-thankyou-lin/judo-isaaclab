# HangMug RoboCasa assets40 campaign

This lane consumes the immutable dataset at
`$HANGMUG_ROBOCASA40_ROOT`.  Its original HDF5 files use `obj_0` and
`obj_1`; the runner maps those labels to task-native `mug` and `mug_tree` only
in memory.  Source file bytes and hashes are never rewritten.

The lane uses `task_config` grasp assistance to match the successfully replayed
source environment.  Any later mechanism comparison must use a separate receipt.

The input inventory contains 40 `teleop/mug_*.hdf5` paths. The original strict
campaign retains the pinned 96-byte `teleop/mug_003.hdf5` as an explicit input
blocker and therefore has 39 runnable inputs.

The success-first campaign recovers the 40th *target scene* without inventing a
second action source. `examples/create_hangmug_target_descriptor.py` copies only
the first state from the hash-pinned official `mug_003_rescale_0.hdf5`, relabels
it to the official base `objaverse_mug_003` and `MugTree_011` assets, and writes
one synthetic zero action. Its provenance marks it as a reset descriptor that
can never be admitted as a source demonstration. All 40 rollouts still use only
`teleop/mug_001.hdf5` as their action/keyframe authority.

`configs/hangmug_robocasa40_0424_success_first_campaign.json` also enforces:

- original insertion/support controller gains (`1.0`, with zero gain lead);
- waypoint retiming, branch selection, and geometry reanchoring as the only
  adaptation controls; and
- an exact-mesh minor-contact budget of at most 8 total frames, 3 consecutive
  frames, and 1 mm penetration. Stable release and coded task success remain
  mandatory; deeper or sustained body contact still fails closed.

Preflight without Isaac:

```bash
export HANGMUG_ROBOCASA40_ROOT=/mnt/amlfs-02/shared/tianyuand_yam/datasets/old/HangMugOnTree/robocasa_assets40_0424
python examples/run_three_task_asset_campaign.py \
  --config configs/hangmug_robocasa40_0424_campaign.json \
  --output-root /path/to/immutable/output \
  --gear-repo /path/to/gear-dc-study \
  --task hangmug_robocasa40_0424 \
  --dry-run
```

Start with `--max-pairs 1` to reproduce the source replay and extract
hash-bound keyframes.  Increase the prefix by one only after the prior pair is
accepted.  Keep one Isaac process active, preserve every pair directory, and
stop for diagnosis at the first non-accepted result.
