# judo-isaaclab

Experimental IsaacLab rollout support for
[Judo](https://github.com/rai-opensource/judo) sampling-based MPC.

The core idea is to avoid cold-resetting into a contact-rich grasp. Every planning
clone starts from the same safe checkpoint, replays the same executed action
history, and only then branches into candidate controls:

```text
contact-free checkpoint
        |
        v
identical action-history replay in every IsaacLab clone
        |
        v
contact manifold reconstructed naturally
        |
        v
Judo candidate controls branch in parallel
```

## What this repository provides

- `HistoryConditionedIsaacLabBackend`: Judo `RolloutBackend` implementation.
- `BranchContext`: checkpoint, action history, rigid-body and assist state.
- `JudoIsaacLabMPC`: small optimizer loop using Judo's existing CEM/MPPI/PS API.
- Duplicate-nominal scoring and a configurable improvement gate for rejecting
  changes below the measured PhysX noise floor.

The backend expects a planning-only runner with the interface already implemented
by `gear-dc-study`'s `CpuBatchedPlanningRunner`.
Judo's high-level `Controller` currently assumes a MuJoCo task/model, so this
package uses Judo's optimizer API directly while retaining its `RolloutBackend`
contract.

## Installation

Install Judo and this package in the same IsaacLab Python environment:

```bash
git clone https://github.com/rai-opensource/judo.git
pip install -e ./judo
pip install -e .
```

IsaacLab itself is intentionally not declared as a pip dependency; use the Python
environment belonging to your IsaacLab installation.

## Backend usage

```python
import numpy as np
from judo_isaaclab import BranchContext, HistoryConditionedIsaacLabBackend

backend = HistoryConditionedIsaacLabBackend(
    runner,
    state_encoder=lambda env: encode_scene_state(env),
)
backend.set_branch_context(
    BranchContext(
        checkpoint_state=checkpoint,
        action_history=executed_actions_since_checkpoint,
        rigid_object_states=rigid_object_states,
        is_relative=True,
    )
)

states, sensors, _ = backend.rollout(
    x0=np.empty(0),
    controls=candidate_actions,  # (rollouts, horizon, action_dim)
)
```

## MPC usage

```python
from judo.optimizers.cem import CrossEntropyMethod, CrossEntropyMethodConfig
from judo_isaaclab import JudoIsaacLabMPC

optimizer = CrossEntropyMethod(
    CrossEntropyMethodConfig(
        num_rollouts=64,
        num_nodes=8,
        num_elites=8,
    ),
    nu=14,
)
mpc = JudoIsaacLabMPC(
    optimizer,
    backend,
    objective,
    duplicate_nominal=4,
    min_improvement=0.04,
)
plan = mpc.plan(context, nominal_knots)
main_env.step(plan.action)
```

## Operational rules

- Use the same assets, solver, timestep, controller, and actuator targets in main
  and planning scenes.
- Prefer a contact-free checkpoint 8-64 control steps before grasp formation.
- Replay the exact actions executed by the main environment.
- Keep MPC horizons short and replan frequently.
- Advance the checkpoint after release or another contact-free boundary.
- Measure duplicate-candidate score spread and reject improvements inside it.

Exact equality between independent PhysX scenes is not expected. In the initial
PutPot validation, seven of eight clones were identical; the remaining clone
differed by less than 0.85 mm in pot position over an eight-action horizon.

## Real IsaacLab smoke

`examples/putpot_identical_rollout_smoke.py` runs the backend against the
official PutPot demo with eight CPU-batched planning clones. It resets at a
contact-free checkpoint, replays 40 shared actions, and applies eight identical
candidate actions:

```bash
python examples/putpot_identical_rollout_smoke.py
```

The checked-in result is
`validation/putpot_backend_smoke.json`: all eight clones retained both grasps,
with maximum internal pot spread of 0.848 mm and 0.002679 rad.

`examples/hangmug_near_contact_smoke.py` exercises a tighter branch boundary.
It starts from contact-free HangMug state 110, replays six shared actions, and
forms contact, grasp, and friction assistance at action 115 in every clone.
All eight clones retain the mug over the eight-action candidate horizon:

- maximum internal mug spread: 0.172 mm and 0.001803 rad;
- maximum mug drop relative to the gripper: 0.276 mm; and
- maximum terminal divergence from the recorded main trajectory: 0.930 mm and
  0.006407 rad.

The exact receipt is `validation/hangmug_near_contact_smoke.json`.

The same test also passes with a 32-action candidate horizon. All clones retain
the grasp; internal spread remains 0.244 mm and 0.005684 rad, while terminal
recorded-main divergence grows to 10.679 mm and 0.010304 rad. See
`validation/hangmug_near_contact_h32.json`.

## Keyframe MPC starter

`examples/extract_hangmug_keyframes.py` replays the canonical source demo and
extracts simulator-verified task boundaries: grasp, pick, handover, tree
approach, release, and stable hang. The resulting timeline is checked in at
`validation/hangmug_source_keyframes.json`.

`examples/hangmug_grasp_keyframe_mpc.py` then performs actual Judo CEM sampling,
not duplicate-action validation. The checked experiments use 16 rollouts and
three optimizer rounds from contact-free state 110:

- grasp target at state 116: all candidates grasped; the accepted best sample
  reached 0.350 mm mean position error;
- pick target at state 148: the accepted best sample reduced mean position
  error from 10.602 mm to 1.015 mm while retaining the grasp in every repeat.

The multi-round acceptance gate compares the best evaluated sample across all
rounds with duplicate evaluations of the original nominal. This avoids
incorrectly rejecting a useful accumulated CEM update merely because the final
round has converged close to its already-improved nominal.

The same script supports history-conditioned handover targets. Both experiments
reset at contact-free state 110 and replay the shared source-demo prefix before
candidate controls branch:

- right grasp from state 300: 190 history actions reconstruct the left grasp;
  the best sample reduces mean mug-position error from 7.948 mm to 2.169 mm
  and reaches two-arm grasp in all six repeats;
- stable handover from state 323: 213 history actions reconstruct the contact
  history; the best sample reduces mean error from 7.410 mm to 3.130 mm and
  reaches right-only grasp with stage 2 latched in all six repeats.

See `validation/hangmug_right_grasp_mpc.json` and
`validation/hangmug_handover_mpc.json`.

Pass `--controls-npz /tmp/controls.npz` to retain the nominal, optimized mean,
and best evaluated control sequences. The comparison renderer then runs nominal
and MPC controls concurrently in two clones of the same CPU PhysX scene:

```bash
python examples/render_hangmug_mpc_comparison.py \
  --controls-npz /tmp/controls.npz \
  --output /tmp/hangmug_nominal_vs_mpc.mp4 \
  --result-json /tmp/hangmug_nominal_vs_mpc.json
```

The checked-in video receipts verify dynamic 1280x480 H.264 output, full decode,
terminal grasp state, and origin-relative mug divergence.

For the longer pre-tree transport window, each sampled control is evaluated in
two clones and assigned their mean reward (`candidate_repeats=2`). From state
600 to the tree-approach keyframe at state 639, this robust CEM run preserves
right grasp and latched handover in all six repeats while reducing mean target
position error from 5.379 mm to 4.317 mm. The parallel video finishes with
2.036 mm origin-relative separation between nominal and MPC.

See `validation/hangmug_tree_approach_mpc.json` and
`validation/hangmug_tree_approach_mpc_video.json`.

The inserted/held target is state 744: the last simulator-verified right-held
state immediately before release at state 745; continuous source replay reaches
stable hang at state 774. Robust MPC from state 700 reaches this target in all
six repeats, reducing mean position error from 7.330 mm to 5.239 mm and mean
rotation error from 0.0586 rad to 0.0466 rad. Both video lanes visibly finish
with the handle seated around the branch and the right gripper still closed.

See `validation/hangmug_inserted_held_mpc.json` and
`validation/hangmug_inserted_held_mpc_video.json`.

Tree approach and insertion targets are evaluated in the mug-tree frame, so
the same target remains valid when the tree moves. A simulator proof translates
the tree by `(10, -5, 0)` mm and rotates it by `1` degree, replays 490 history
steps, and branches at state 600. All six robust repeats retain the right grasp
and handover latch at the held-insertion target. The best samples finish with
7.508 mm mean relative-position error and 0.1561 rad mean relative-rotation
error. The side-by-side render fully decodes and ends with 0.760 mm mug
separation between nominal and MPC.

See `validation/hangmug_inserted_held_moved_tree_mpc.json` and
`validation/hangmug_inserted_held_moved_tree_mpc_video.json`.

For trees whose branch geometry differs, pass three aligned non-collinear
points in each tree's local frame:

```bash
--source-branch-points X0 Y0 Z0 X1 Y1 Z1 X2 Y2 Z2 \
--target-branch-points X0 Y0 Z0 X1 Y1 Z1 X2 Y2 Z2
```

The mug target is transferred from the source branch frame to the corresponding
target branch frame. This supports, for example, a branch that is higher
relative to the new tree root; the unit suite verifies a 20 cm local branch
height change. For `hang_complete`, pass/fail requires stage-3 success, both
grippers released, at most 30 mm branch-relative position error, at most
50 mm/s mug speed, and all conditions retained for 30 consecutive control
steps.

The translation/rotation simulator proof moves the tree by `(10, -5, 20)` mm and
rotates it by `1` degree, uses explicit matched branch points, and plans from
state 600 through stable hang state 774. All 16 sampled candidates and all six
best-sample repeats complete the hang. The video confirms both nominal and MPC
lanes release the mug and latch stage 3; terminal mug divergence is 1.220 mm
and 0.0496 rad. This checks target-frame equivariance, not changed geometry.

See `validation/hangmug_corresponded_branch_hang_complete_mpc.json` and
`validation/hangmug_corresponded_branch_hang_complete_mpc_video.json`.

## Actual taller-tree adaptation

`examples/create_taller_mugtree_asset.py` creates a real USD variant whose
visual and collision geometry are both scaled in Z. The checked experiment uses
the official `mug_tree_000` source at 1.2x Z scale:

- asset height: 350.275 mm to 420.330 mm;
- demonstrated branch center: 48.75 mm higher in world coordinates;
- nominal source-demo controls: 0/10 successful repeats, 175.9 mm mean terminal
  branch-relative error;
- MPC best sample: 10/10 successful repeats, 8.29 mm mean terminal error,
  and a complete 30/30-step stability window.

The 225-step rollout is optimized as four smooth right-arm correction knots.
This searches 24 coherent variables instead of perturbing 1,350 independent
joint targets. The side-by-side video runs nominal and MPC controls concurrently
in the same two-clone CPU PhysX scene. The nominal mug drops to the table; MPC
releases it on the taller branch. Independent render-time acceptance reports
0/30 stable frames for nominal and 30/30 for MPC.

See `validation/hangmug_actual_taller_tree_mpc.json` and
`validation/hangmug_actual_taller_tree_mpc_video.json`.

### 1.5x geometry boundary

The same one-shot budget fails when the full tree geometry is scaled to 1.5x Z:
the matched branch is 121.9 mm higher and also physically thicker/steeper.
History-conditioned staged MPC improves the trajectory without teleporting:

- lift/approach at state 639: 10/10 within the 60 mm capture region;
- replans at states 700, 720, and 735: each 10/10 with the grasp retained;
- held insertion at state 744: fails the 30 mm gate, with 42.6 mm mean error;
- release/stabilization: 0/10 stable hangs; the mug drops.

`--history-controls-npz` replays accepted earlier-stage controls before a later
branch. `examples/compose_hangmug_staged_controls.py` composes contiguous stages
for a full side-by-side render. This result is a tested failure boundary, not
evidence that 1.5x is unreachable with a redesigned target asset, controller,
or insertion objective.

See `validation/hangmug_z150_staged_mpc.json` and
`validation/hangmug_z150_staged_mpc_video.json`.

### Hierarchical task-space MPC

The environment consumes 14-D absolute joint-position targets. The original
HangMug search therefore optimized smooth joint-target corrections. J-PARSE is
not on this rollout path: it is only used by the MimicGen end-effector-to-action
adapter, so its SVD cannot explain MPC wall time.

`--search-space task` adds a hierarchical alternative:

1. an outer semantic program interpolates a six-dimensional right-end-effector
   residual between stage offsets;
2. CEM searches local Cartesian residuals around that program;
3. a batched damped-least-squares adapter maps all candidate residuals to the
   14-D joint targets required by IsaacLab, using `torch.linalg.solve` rather
   than an SVD;
4. later stages replay the exact executed joint actions from accepted earlier
   stages before branching.

On the fully Z-scaled 1.5x tree, the approach stages at states 639, 700, and
735 each passed all six fresh-clone repeats, with mean matched-branch errors of
13.56, 19.54, and 33.49 mm. The strict inserted/held stage found one sampled
28.49 mm success, but the identical program reproduced at 39.14 mm mean and
0/8 successes in fresh clones. Release/stabilization remained 0/6. Thus the
hierarchy improves long-range adaptation substantially, but does not make the
thicker/steeper 1.5x contact geometry robustly insertable under this physics
and action interface.

The dominant runtime is still repeated CPU PhysX history reconstruction and
candidate simulation, not task-space conversion. Retaining or checkpointing a
history-conditioned planning scene is the next meaningful speed optimization.

See `validation/hangmug_z150_hierarchical_task_mpc.json`.

#### Closed-loop pose tracking

The initial task-space adapter still added a bounded Jacobian correction to
each source joint target. When the source trajectory retracted, that absolute
joint target could pull the arm downward despite the Cartesian offset.

`--task-controller pose_tracking` removes that joint anchor. It first records
the source end-effector trajectory from unmodified history, then computes
`desired pose - current pose` online and applies batched DLS to the current
joint positions. On the same 1.5x tree and history:

- late approach improved from 33.49 mm mean to 20.42 mm mean, with 6/6 repeats;
- strict held insertion improved from 0/8 to 8/8 repeats, with 21.67 mm mean
  and 28.29 mm maximum error;
- release search found one 16.31 mm stable-hang sample, but that contact outcome
  did not reproduce in the independent six-clone evaluation (0/6).

The video confirms the right arm remains near the transformed branch rather
than springing back toward the source joint targets. The mug still fails robust
release/stabilization, so release contact—not approach or insertion—is now the
remaining boundary.

See `validation/hangmug_z150_closed_loop_task_mpc.json`.

#### Semantic-keyframe pose tracking

`--task-controller semantic_pose` removes the remaining source-trajectory
anchor from the right-arm tree-interaction stages. The demonstration supplies
only:

- the stage boundary and final semantic end-effector keyframe;
- the keyframe's transformation to the matched target-tree branch;
- discrete grasp/hold/release intent.

At each branch, the controller measures the live end-effector pose and creates
a new smooth Cartesian trajectory to the transformed semantic keyframe. CEM
searches bounded Cartesian residuals around that trajectory, and batched DLS
maps closed-loop pose errors from the current joint state to IsaacLab's joint
targets. The source joint trajectory and intermediate source end-effector
trajectory are not controller references. Earlier accepted pickup/handover
stages are still replayed as physical history before the tree branch.

On the fully Z-scaled 1.5x tree:

- late approach passed 6/6 repeats at 24.71 mm mean and 24.94 mm maximum
  branch-relative error;
- held insertion passed 8/8 repeats at 23.19 mm mean and 24.59 mm maximum
  error;
- release/stabilization remained 0/6, with the mug dropping to 302.83 mm mean
  terminal error.

This confirms that semantic-keyframe MPC solves the source-path spring-back
for approach and insertion. Stable release remains a separate contact-planning
problem.

See `validation/hangmug_z150_semantic_task_mpc.json`.

#### Editable insertion primitive and frame diagnostics

For `semantic_pose` plus `inserted_held`, the planner now calls an explicit
`insert()` primitive instead of using one straight Cartesian interpolation.
Its editable defaults are expressed in the matched target-branch frame:

```text
pre-insert offset: (+50, 0, 0) mm along branch root-to-tip tangent
seat offset:       (  0, 0, 0) mm at stable-hang-derived target
phase fractions:   approach=0.35, seat=0.80, hold=0.20
```

The corresponding CLI overrides are
`--insert-approach-offset-branch`,
`--insert-seat-offset-branch`, `--insert-approach-fraction`, and
`--insert-seat-fraction`. This makes small geometry-specific corrections easy
to revise without binding the controller to the source joint path.

`render_hangmug_mpc_comparison.py --draw-coordinate-axes` draws RGB=`XYZ`
frames for the matched branch (5 cm), desired EEF (3.5 cm), and live EEF
(2 cm). The three-point correspondence transfers the full desired EEF pose
(position and orientation), not only a world-space offset.

The original state-744 validation was a false positive: it accepted a latched
handover, retained right grasp, and up to 30 mm mug-pose error, but never proved
that the branch occupied the mug handle. The corrected primitive derives its
EEF target from the stable released-hang mug pose at state 774 while preserving
the live EEF-to-mug grasp transform. Acceptance now requires the mug to match
that stable branch-relative pose within 10 mm and 0.15 rad for three consecutive
steps. Released-hang acceptance uses the existing task stage-3 latch, which
already requires 30 stable steps, plus both grippers open and mug speed at most
0.05 m/s. Branch-relative pose and orientation remain diagnostics for changed
geometry rather than duplicating the task's stability gate.

On the 1.5x Z-scaled tree, the 40-step tangent insertion reproduced the strict
held-pose gate in 6/8 fresh clones. Its all-camera MPC lane finished at 3.58 mm
and 0.075 rad with the handle visibly around the branch. Continuing those exact
controls through release passed the existing stable-hang task check in 8/8
fresh clones with both grippers open and zero terminal mug speed. Contact-heavy
release candidates use `--candidate-repeat-reducer min` so the worst duplicate,
not a lucky mean score, determines the CEM update.

See `validation/hangmug_z150_insert_primitive.json`.

The same staged controller also completes a 2.0x Z-scaled tree whose matched
branch is 243.73 mm above the source branch. Three semantic approach stages
retain the right grasp, then the tangent insertion passes the strict held-pose
gate in 10/10 fresh clones (7.39 mm mean, 7.95 mm maximum position error).
Continuing those exact controls through release passes the existing 30-step
stable-hang task check in 8/8 fresh clones with both grippers open and zero
terminal mug speed. All-camera renders were fully decoded and visually checked:
the handle is threaded around the branch before release and remains supported
after both arms retract.

See `validation/hangmug_z200_staged_mpc.json`.

## Resumable asset-adaptation agent

`examples/hangmug_task_adaptation_agent.py` executes a versioned task bundle as
an evidence-gated loop: reproduce the source strategy on substituted assets,
refine semantic insertion from branch correspondences, plan release, and then
validate the promoted controls in one continuous simulator episode. Failed
trials are retained in a JSON ledger, and `--resume` continues from the first
unfinished trial rather than repeating completed simulation.

The included `configs/hangmug_mug029_tree037.json` adapts the source
`mug_000`/`mug_tree_000` demonstration to the official
`mug_029`/`mug_tree_037` instances:

```bash
python examples/hangmug_task_adaptation_agent.py \
  --bundle configs/hangmug_mug029_tree037.json \
  --workspace /tmp/hangmug-adaptation \
  --gear-repo /path/to/gear-dc-study \
  --render-failures \
  --resume
```

An insertion that narrowly misses the strict 10 mm/0.15 rad diagnostic gate
may be promoted only through an explicit, bounded provisional gate. It still
must retain the grasp in every repeat, and both the downstream release search
and final continuous episode must pass the existing coded task-success check.
The final validator additionally requires one process, one initial reset, zero
inter-stage resets, exact control-history continuity, target-asset provenance,
a fully decodable H.264 render, and stage-3 success.

The checked proof is `validation/hangmug_mug029_tree037_adaptation.json`.
