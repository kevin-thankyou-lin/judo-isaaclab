# Agent Instructions

## Grasp

Choose the safest approach axis in the live object or affordance frame. Resting objects usually use world top-down; held objects and handovers use the object's current frame. Move to a clear standoff, orient while clear, approach open along that axis, verify pad straddle and palm/wrist clearance, close in place, and wait for stable contact or assist before lifting or releasing. If grasping fails, repair the first failed phase using geometry and live state—never controller or IK gains. Combine rotation and translation only when the swept path is verified collision-free.

## HangMug choreography

Preserve the demonstrated arm roles unless an immutable task manifest explicitly reverses them. In the pinned Task 2 source, the left arm picks and gives the mug, the right arm receives it and performs branch transport/insertion, and therefore the non-carrying left arm—not the right carrier—must retreat to its demonstrated rest pose after handover. Apply the rest rule by role for a reversed-hand variant; never send the arm currently holding the mug to rest.

Treat insertion as one continuous closed-carrier phase. Enter it with the giver already open and the carrier closed, issue no gripper transition during transport/alignment/insertion/unload, and open the carrier exactly once only at the final supported release. Do not use intermediate open/close pulses to repair alignment or collision failures.

## Parallel HangMug lanes

Every simulator lane must receive one explicit zero-based asset pair index and a unique lane ID. Never infer ownership from the first missing shared-ledger entry. Give each lane its own Git worktree and `results/task2` tree; do not share or concurrently edit a ledger, attempt directory, repair candidate, branch, or output root. Reconcile hash-verified accepted receipts centrally before assigning or merging more work.
