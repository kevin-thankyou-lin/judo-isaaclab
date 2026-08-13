# Agent Instructions

## PutPot quality choreography

Quality-wave work is opt-in through `configs/putpot_quality_wave_v1.json`. Preserve the
existing source-first worker, task predicate, controller gains, and generic repair machinery.
Do not turn a pair-specific correction into a default or weaken the coded PutPot success
predicate.

Acquire the pot sequentially. Approach the left handle through a collision-clear pregrasp,
orient while clear, and close only when both left pads can contact broad interior handle
surfaces. The left grasp must be flush, force-backed, and stable before the right arm begins
its final approach. Keep the pot still while the right gripper acquires the opposite handle
with the same broad two-pad standard. Fingertip, edge-only, or one-pad grasps are diagnostic
failures. Require all four pads to remain force-backed and interior for 15 consecutive frames
before issuing any lift or transport action.

After the four-pad latch, plan object-first: derive both wrist paths from the live pot-to-wrist
transforms. Lift bimanually, transfer to the cooktop with smooth coordinated task-space motion,
and lower onto stable support. Preserve bounded translation, rotation, acceleration, and grasp
transform drift; do not insert internal stops or independently drag one wrist. Open both
grippers once, together, only after supported placement. Keep them open while both arms move
away along a collision-clear path and return to their demonstrated start configurations.

## Robustness and collision evidence

Prefer broad, well-distributed contact with margin to every pad edge. A quality acceptance
must include a measurable contact receipt and a deterministic fixed-seed perturbation audit
over grasp pose, end-effector pose, and joint actions. Small perturbations must not turn the
grasp or downstream semantic task into a failure. Do not tune controller or IK gains to pass
the audit.

Audit the complete swept trajectory, not only endpoints. Include both arms, both grippers,
and both wrist-camera bodies. Unnecessary robot-robot, gripper-camera, and camera-camera
collisions fail quality acceptance. Exclude only declared structural adjacency and intended
gripper-to-assigned-handle contact; never add an exclusion merely because an attempted path
collides.

## Parallel lanes and receipts

Every simulator worker must receive one explicit zero-based asset-pair index, one lane ID, and
exactly one `CUDA_VISIBLE_DEVICES` entry. A GPU lease is exclusive. Never infer ownership from
a shared ledger frontier. Each pair owns a separate Git worktree, branch, output root,
attempt directory, quality receipt, and repair candidate. Terminal failure means diagnose and
retry in the same pair lane; it is not permission to stop an unfinished pair or start a second
worker. Accepted receipts are immutable and centrally reconciled only after hashes, video,
aligned HDF5/trace, semantic success, quality audits, and worker cleanup all pass.
