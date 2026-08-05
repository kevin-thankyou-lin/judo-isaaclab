# Evidence-driven coding-agent asset adaptation

The source demonstration proposes a semantic strategy. It is not an exact joint
trajectory for every asset instance. The coding agent implements task-space
skills, while this harness owns evidence, sequencing, and fail-closed acceptance.

The agent does not directly author an accepted demonstration. It writes and
repairs a semantic skill program; the simulator produces the demonstration only
after the uninterrupted rollout passes the existing coded task predicate.

```text
source demo + task predicate + asset geometry
                    |
          semantic strategy and frames
                    |
          deterministic skill program
                    |
       continuous target-asset simulation
                    |
     numeric checks + rendered diagnosis
                    |
        accepted recorded demonstration
```

## Design principles

- Start from the task predicate. Preserve its thresholds and stable windows.
- Use the demonstration for subtask order, ownership, contact intent, and
  object-relative relationships—not as a target joint trajectory.
- Generate target code directly from the current target state and authored
  target-local part geometry. A failed semantic run on the source asset does
  not block target repair; source-skill runs are optional diagnostics.
- Express targets in semantic frames: grasp contacts, support surfaces, handle
  holes, branch tangents, drawer cavities, and articulation axes.
- Preserve task-native interaction mechanisms used by the authoritative datagen
  environment, including configured grasp assistance. Record their exact config
  and state transitions instead of silently disabling or replacing them.
- Keep skills explicit and editable: `grasp`, `handover`, `transport`,
  `insert/place`, `release`, `close`, and `settle`.
- Diagnose with both state traces and synchronized cameras. Rendered evidence
  reveals wrong-side approaches, occlusion, collisions, and unstable support;
  coded success remains the final authority.
- Test a repaired stage locally, but accept only a one-reset, uninterrupted
  end-to-end rollout. Mid-contact resets are not valid substitutes for natural
  contact history.
- Reanchor from observed state after contact-critical milestones, then preserve
  the observed object-to-gripper contact frames through transport.
- Scale duration and clearance from measured geometry, preserve hash-verified
  successes, and stop a sequential lane at its first failed pair for diagnosis.
- Treat task completion and motion quality as separate gates. A successful
  path can still need shorter motion, fewer internal stops, better clearance,
  or a more useful observer view.

## Required proof

The final claim requires three accepted runs:

1. `target_replay`: source actions fail on the selected target assets.
2. `target_skill`: the adapted coded skill succeeds on that same target.
3. `final_render`: the same target skill revision succeeds in one continuous,
   technically valid rendered rollout.

`source_skill` remains an optional diagnostic phase. It is not required when
source action replay and an earlier semantic program both fail.

The target-replay failure matters. If source replay already succeeds, the target
is useful regression coverage but does not prove adaptation.

## Agent loop

1. Audit the authoritative task manager, stage latches, assets, dataset, and a
   previously proven analogous skill before editing code.
2. Extract the source strategy and semantic frames from the demonstration and
   collision geometry.
3. Implement a continuous graph of small deterministic primitives such as
   `grasp`, `handover`, `transport`, `insert/place`, `release`, and `close`.
4. Optionally run the skill on source assets to diagnose correspondence; do not
   gate target generation on source semantic success.
5. Rank target candidates geometrically and run unchanged source actions until
   a validated replay-failure target is found.
6. Execute the skill on that target and classify the first failed predicate as
   provenance, correspondence, contact, reachability/tracking, clearance,
   release/support, or acceptance instrumentation.
7. Change only the implicated primitive. Do not weaken a threshold or disturb
   stages that already pass.
8. Keep every attempt in the append-only ledger with its command, revision,
   asset identity, parameters, reset/teleport counts, stage events, metrics,
   checks, rejection reason, and artifact hashes.
9. Finish with one reset, no inter-stage teleport, coded success, full video
   decode, visual inspection, tests, clean worktrees, and matching remote hash.

The final demonstration must retain actions, states, observations, asset and
dataset hashes, controller parameters, stage/success traces, and the exact code
revision. A filename or successful process exit is not acceptance evidence.

## Diagnosis policy

- Large EEF error: reachability/controller problem. Use target-relative IK or a
  different waypoint.
- Accurate endpoint but failed contact: interaction-path or grasp-transform
  problem. Preserve the contact transform and move along the articulation or
  insertion axis.
- Failed final placement: adjust support-frame clearance, release timing, and
  the stable-support window.
- Correct task outcome but poor motion: preserve the successful interaction and
  optimize a separate quality gate such as path length, internal stops,
  clearance, duration, or observer visibility. Do not claim lower jerk unless
  it is actually measured.
- Successful target replay: reject the target as adaptation evidence.
- Missing provenance or artifacts: repair infrastructure before motion code.

## Scaling beyond one target

Run unchanged source actions over many same-category instances, retain genuine
replay failures, and group them by the first failed primitive. Adapt one
representative from each failure group, promote the correction into
geometry-conditioned skill defaults, and rerun every solved pair as a
regression suite. Hold out additional pairs to measure whether the skill—not an
asset identifier or memorized action sequence—generalizes.

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

### PutPot repair runtime and render policy

One PutPot asset visit owns one persistent Isaac worker. The worker loads Kit,
the task, and one target asset pair once. The campaign creates an interactive
session and appends only cycle 1 to `requests.jsonl`. After the durable attempt
ack appears in `worker_receipts.jsonl`, the worker stays alive and idle while
the supervisor diagnoses that receipt. It never receives a static batch of
prebuilt retries.

Semantic trajectory/controller parameters live in the versioned, validated
`configs/putpot_semantic_program_v4.json` program spec. Schema v2 added a bounded
object-local receiving-jaw centering translation fraction; schema v3 makes the
pre-close missing-finger correction limit reloadable. Schema v4 adds a
contact-triggered `receiving_jaw_close_horizon_steps`: zero preserves the slow
full-contact-hold default used by pot 023, while a positive value closes over a
bounded number of control steps after the measured single-pad latch. Older
schemas remain preserved for provenance but are no longer accepted for new requests. The worker
reloads that JSON file on every request after a full environment reset and fresh task
predicates, controller state, recorder, trace, and encoder state. A versioned
Python controller runs behind a subprocess protocol boundary and is likewise
reloaded for every request without reloading Isaac. The SHA-256 of the exact
immutable spec and controller copies is recorded in the result, runtime receipt,
and worker ack. Parameter and Python-controller revisions therefore reuse the
loaded Isaac process. A changed asset pair, device, camera capability, or
Isaac-host code commit remains a worker boundary and requires a new process.

A measured receiving-jaw translation also retimes the close command by its
object-local distance at the conservative 1 mm/control-step loaded-contact
horizon, keeping the jaw open while the wrist tracks the correction.

Every attempt records both its immutable lifetime attempt number and its
one-based repair-epoch cycle number. The epoch limit is four diagnose-to-repair
cycles, not four identical-code retries. A same-spec repeat is rejected unless
the preceding receipt is classified `ambiguous_failure` and the submitted
request includes an explicit ambiguity reason. After four nonaccepting cycles
the scheduler rotates the asset into hard-case review; it never reuses or
overwrites an attempt directory.

A different spec hash is not sufficient evidence of a material repair. For a
changed-parameter request, every changed parameter must appear in the previous
worker ack's `failed_stage_program_parameter_observations`. The close-horizon
observation includes its requested and applied step counts plus close-start,
close-end, and grasp-end steps. Missing or mismatched observations fail closed.

To append exactly one revised request to a waiting worker:

```bash
python examples/submit_putpot_program_request.py \
  --session-json /path/to/interactive_session.json \
  --program-spec-json /path/to/revised_program.json
```

Use `--ambiguity-reason '...'` only for a justified same-spec stochastic probe.
Use `--shutdown-reason '...'` at an artifact-safe worker boundary. The append
CLI refuses a new cycle until the previous request has an ack, refuses a fifth
cycle, and copies each submitted spec into the repair epoch before enqueueing.

PutPot repair cycles are rendered inside one persistent Isaac process. Every
diagnose-to-repair cycle writes a synchronized H.264 MP4 beside its trace, and
the coding agent inspects both before submitting one revised semantic spec or
Python controller plugin. Cameras remain instantiated for the asset visit, but
the scene is reset between revisions, so visual evidence does not require a new
Isaac startup. A rendered strict success merges directly; merging still
requires a continuous, fully decoded H.264 video plus every existing task,
bimanual transport, stability, provenance, trace, and HDF5 gate.
Typed deterministic controller or configuration exceptions are neither physics
failures nor visual ambiguities: the worker records their exception provenance,
stays at the acknowledged boundary for supervisor action, and never spends a
rendered retry on them. Python controller plugins and parameter specs are both
copied immutably and reloaded between resets without restarting the Isaac host.

The tmux-hosted coding agent is guarded by
`examples/watch_putpot_repair_handoffs.py`. The watchdog polls durable remote
receipts, allows a normal diagnosis grace period, and then steers the existing
agent once if an acknowledged repair boundary has not produced a new request.
It uses a separate short grace after a completed visit so round-robin rotation
cannot silently stop or incur diagnosis-scale idle time. The watchdog never
writes the worker queue: the coding
agent must still inspect the synchronized MP4 and trace and use the guarded
submission CLI, which
continues to reject unacknowledged, unchanged, duplicate, and fifth-cycle
requests. Wake attempts are recorded in an atomic state file and are bounded
per receipt. Remote snapshot calls have a finite timeout and transient SSH,
timeout, or malformed-response failures are logged and retried; a connection
blip cannot terminate the daemon.

Each watchdog diagnosis wake carries a task-general progress contract. Before
submitting a revision, the repair agent must record the first failed stage, the
primary metric before and after the preceding revision, one falsifiable causal
hypothesis, an expected metric delta, the repair family, and whether the change
is local or structural. A new controller or spec hash establishes distinct
provenance, not physical progress. Local parameter or feedback changes are
appropriate only when synchronized video and trace evidence show that the
upstream semantic approach and contact frame are already correct and the
measured tracking residual is within the correction's authority. If two
attempts from the same repair family fail at the same stage without materially
improving the primary metric, that family is exhausted for the visit. The next
cycle must change the upstream Python approach, contact-frame construction, or
trajectory primitive, or the asset must rotate when no evidence-backed
structural change fits the remaining cycle budget. Strict acceptance predicates
are never adjusted by this escalation rule.

Every PutPot grasp diagnosis also starts with a causal acquisition audit across
both arms, rather than inspecting only the arm that ultimately misses. The agent
records first-contact arm and step, contact order, per-pad force/contact windows,
pregrasp object motion, and object motion before the peer arm arrives. Material
object displacement before peer contact makes the earliest contacting arm's
approach or sequencing the upstream failure; downstream wrist tuning is not an
admissible repair while that displacement remains. The next revision must state
an evidence-derived object-motion abort threshold and use an appropriate safer
standoff, contact-gated stop/backoff, acquisition ordering, synchronized
acquisition, or observed-pose reanchoring. Transport remains gated until both
hands sustain dual-pad contact for an explicitly stated validation window.

Runtime receipts report wall-clock seconds for app startup, asset/environment
load, reset, trajectory build, rollout, render/encode, trace/demo,
validation/decode/hash, and shutdown. They additionally report the full
per-attempt wall time, sum of those named phases, and the unattributed remainder;
worker bootstrap/import/validation time is reported separately instead of being
folded into a named phase. CPU deployment is sized to the remote
cgroup quota of eight cores; host CPU inventory is not treated as usable quota.

## HangMug deployment

`configs/hangmug_evidence_agent.json` defines the HangMug proof. Its semantic
frames are the left and right grasp contacts, mug handle hole, branch tip and
tangent, support normal, and a left-wrist observer pose. The coding-only
strategy is:

```text
left handle grasp
        -> physical right-hand handover
        -> smooth transport while the left wrist moves to observe the branch
        -> align the handle hole beyond the branch tip
        -> insert inward along the branch tangent
        -> unload, release, and validate stable support
```

The observer pose is copied from the source insertion view in the source-tree
frame, transferred into the target-tree frame, and reached during transport.
The left arm then stays at that pose throughout final alignment, insertion,
release, and support validation. This makes the target branch visible without
changing the successful right-hand path or treating camera visibility as a
task-success substitute.

HangMug uses the datagen environment's grasp-assist interface. The task config
defaults to a left-arm fixed joint, while the same config exposes a fingertip
friction mechanism. The evidence-agent config selects friction (`high=100`,
`low=0.5`) to avoid runtime topology changes. It engages only after `0.3 s` of
contact-backed left grasp and returns to low friction during the physical
two-hand handover. The right-hand transport and insertion remain unassisted.
Evidence must show that the selected mechanism existed, actually engaged, and
was released before the final unsupported hang.

Use `--grasp-assist-mechanism friction` for the selected evidence lane. Use
`task_config` to reproduce the YAML's active mechanism or `fixed_joint` for an
explicit topology-owning comparison.

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
