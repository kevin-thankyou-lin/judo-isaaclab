# Agent Instructions

## Grasp

Choose the safest approach axis in the live object or affordance frame. Resting objects usually use world top-down; held objects and handovers use the object's current frame. Move to a clear standoff, orient while clear, approach open along that axis, verify pad straddle and palm/wrist clearance, close in place, and wait for stable contact or assist before lifting or releasing. For better sim-to-real transfer, prefer grasps that place as much usable gripper-pad area as possible on broad, stable regions of the intended object or affordance; favor well-distributed contact over marginal fingertip or edge contact, while preserving force closure, collision clearance, task controllability, and clean release. If grasping fails, repair the first failed phase using geometry and live state—never controller or IK gains. Combine rotation and translation only when the swept path is verified collision-free.

## HangMug choreography

Preserve the demonstrated arm roles unless an immutable task manifest explicitly reverses them. In the pinned Task 2 source, the left arm picks and gives the mug, the right arm receives it and performs branch transport/insertion, and therefore the non-carrying left arm—not the right carrier—must retreat to its demonstrated rest pose after handover. Apply the rest rule by role for a reversed-hand variant; never send the arm currently holding the mug to rest.

Treat insertion as one continuous closed-carrier phase. Enter it with the giver already open and the carrier closed, issue no gripper transition during transport/alignment/insertion/unload, and open the carrier exactly once only at the final supported release. Do not use intermediate open/close pulses to repair alignment or collision failures.

## HangMug quality regeneration

Generate a new quality-qualified trajectory even when an older direct replay or accepted artifact exists. Always plan against a second-row branch. This quality campaign explicitly interprets “right start” as a collision-screened carrying staging configuration, not a release or idle rest: after physical handover, keep the right gripper closed around the mug while the right arm returns to its demonstrated start configuration; only then move the open left arm to the target-branch observer pose, and only after that begin right-arm transport and insertion. Hold the left observer pose during insertion. The right gripper must remain closed from handover through branch unload and execute one monotone opening only at final supported release. Require sustained interior two-pad contact for both pick and carrier grasps; fingertip or edge-only contact is diagnostic failure, not quality acceptance. Preserve one reset, zero inter-stage resets, unchanged controller and IK gains, decoded video, aligned HDF5/trace evidence, and stable terminal hang.

## Parallel HangMug lanes

Every simulator lane must receive one explicit zero-based asset pair index and a unique lane ID. Never infer ownership from the first missing shared-ledger entry. Give each lane its own Git worktree and `results/task2` tree; do not share or concurrently edit a ledger, attempt directory, repair candidate, branch, or output root. Reconcile hash-verified accepted receipts centrally before assigning or merging more work.
