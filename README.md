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
