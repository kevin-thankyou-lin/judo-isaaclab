# Secondary-arm active perception

For precise bimanual tasks, the arm that is not manipulating can optionally
act as an embodied camera operator. During HangMug insertion, the left wrist
camera can point toward the selected branch while the right arm carries and
releases the mug.

This is an optional task-adapter capability, not a generic harness policy and
not an acceptance requirement for the success-first 40-demo campaign. A future
implementation should:

- accept a task-provided interaction target, such as the branch support point;
- solve a collision-cleared camera look-at waypoint without touching the task
  objects or constraining the manipulating arm;
- log enable/disable steps, optical-axis angular error, target visibility, and
  minimum arm/object clearance;
- fall back to a neutral observer pose for embodiments without a spare arm or
  wrist camera; and
- keep task success and evidence validation independent of the optional view.

The same hook can support peg insertion, drawer placement, pouring, assembly,
and other contact-sensitive tasks where an unused limb can improve viewpoint
quality for the learned policy.
