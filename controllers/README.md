# Reloadable PutPot controllers

PutPot repair attempts may replace controller Python without restarting Isaac.
The Isaac host never imports these files.  It starts one child process per
attempt, verifies the immutable file hash, and exchanges JSONL messages through
`putpot_controller_protocol.py`.

A plugin must define `create_controller()`.  The returned object implements:

- `initialize(context)`: receives the base trajectory, target geometry,
  program parameters, attempt identity, and reset observation.  It returns the
  protocol version, program name, unchanged evidence horizon, and metadata.
- `command(request)`: receives the current observation and base command.  It
  returns either authoritative Cartesian targets for both wrists and grippers,
  or one direct 14-joint action.
- `close()` (optional): releases controller-only resources.

The child may maintain arbitrary Python state and implement new phases,
branches, geometry logic, or feedback.  The host retains scene ownership,
Cartesian IK, physics stepping, recording, and strict task/evidence checks.
Timeouts, malformed output, non-finite commands, plugin exceptions, and hash
mismatches fail the attempt closed.

`putpot_passthrough.py` is the reference implementation.  It reproduces the
host base trajectory while proving the subprocess boundary is active.
