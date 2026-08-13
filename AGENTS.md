# Agent Instructions

## Grasp

Choose the safest approach axis in the live object or affordance frame. Resting objects usually use world top-down; held objects and handovers use the object's current frame. Move to a clear standoff, orient while clear, approach open along that axis, verify pad straddle and palm/wrist clearance, close in place, and wait for stable contact or assist before lifting or releasing. For better sim-to-real transfer, prefer grasps that place as much usable gripper-pad area as possible on broad, stable regions of the intended object or affordance; favor well-distributed contact over marginal fingertip or edge contact, while preserving force closure, collision clearance, task controllability, and clean release. If grasping fails, repair the first failed phase using geometry and live state—never controller or IK gains. Combine rotation and translation only when the swept path is verified collision-free.
