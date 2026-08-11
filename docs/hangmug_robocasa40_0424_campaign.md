# HangMug RoboCasa assets40 campaign

This lane consumes the immutable dataset at
`$HANGMUG_ROBOCASA40_ROOT`.  Its original HDF5 files use `obj_0` and
`obj_1`; the runner maps those labels to task-native `mug` and `mug_tree` only
in memory.  Source file bytes and hashes are never rewritten.

The lane uses `task_config` grasp assistance to match the successfully replayed
source environment.  Any later mechanism comparison must use a separate receipt.

The input inventory contains 40 `teleop/mug_*.hdf5` paths.  The pinned
`teleop/mug_003.hdf5` is a 96-byte truncated HDF5 and is retained as an explicit
input blocker.  Thirty-nine inputs are runnable.  The campaign cannot report
40/40 until an asset-matched, hash-verified recovery of that canonical file is
provided.

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
