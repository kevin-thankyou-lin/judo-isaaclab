"""Compose contiguous HangMug MPC stages into one replayable controls file."""

import argparse

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--output", required=True)
    parser.add_argument("stages", nargs="+")
    args = parser.parse_args()

    loaded = [np.load(path) for path in args.stages]
    try:
        starts = [int(stage["start_state"]) for stage in loaded]
        controls = [
            np.asarray(
                stage[
                    "best_executed_actions"
                    if "best_executed_actions" in stage.files
                    else "best_sample"
                ],
                dtype=np.float32,
            )
            for stage in loaded
        ]
        ends = [start + len(control) for start, control in zip(starts, controls)]
        if starts[1:] != ends[:-1]:
            raise ValueError(
                f"Stages are not contiguous: starts={starts}, ends={ends}"
            )
        target_trees = {str(stage["target_mug_tree"]) for stage in loaded}
        if len(target_trees) != 1:
            raise ValueError(f"Target trees differ: {sorted(target_trees)}")

        first = loaded[0]
        last = loaded[-1]
        with h5py.File(args.dataset, "r") as handle:
            nominal = np.asarray(
                handle[f"data/{args.episode}/actions"][starts[0] : ends[-1]],
                dtype=np.float32,
            )
        best = np.concatenate(controls)
        if nominal.shape != best.shape:
            raise ValueError(
                f"Nominal/best shapes differ: {nominal.shape} versus {best.shape}"
            )

        np.savez_compressed(
            args.output,
            nominal=nominal,
            best_sample=best,
            best_executed_actions=best,
            optimized_mean=best,
            checkpoint_state=np.int64(first["checkpoint_state"]),
            start_state=np.int64(starts[0]),
            target_state=np.int64(last["target_state"]),
            target_name=np.asarray(str(last["target_name"])),
            history_actions=np.asarray(
                first["history_actions"], dtype=np.float32
            ),
            history_control_overrides=np.asarray(args.stages),
            tree_offset_xyz=np.asarray(
                last["tree_offset_xyz"], dtype=np.float32
            ),
            tree_yaw_deg=np.float32(last["tree_yaw_deg"]),
            source_mug_tree=np.asarray(str(last["source_mug_tree"])),
            target_mug_tree=np.asarray(str(last["target_mug_tree"])),
            tree_root_z_adjustment=np.float32(
                last["tree_root_z_adjustment"]
            ),
            source_branch_points=np.asarray(
                last["source_branch_points"], dtype=np.float32
            ),
            target_branch_points=np.asarray(
                last["target_branch_points"], dtype=np.float32
            ),
        )
    finally:
        for stage in loaded:
            stage.close()


if __name__ == "__main__":
    main()
