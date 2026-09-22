"""Play a packaged demo sequence on a hand.

The poses themselves ship with orca_core
(:func:`orca_core.demo_poses.load_demo_poses`)
"""

from orca_core import OrcaJointPositions
from orca_core.constants import NUM_STEPS, STEP_SIZE
from orca_core.demo_poses import load_demo_poses


def _drop_joints(joint_pos: OrcaJointPositions, joints: set[str]) -> OrcaJointPositions:
    """Return *joint_pos* without *joints*, so they're left uncommanded."""
    if not joints:
        return joint_pos
    return OrcaJointPositions.from_dict(
        {joint: value for joint, value in joint_pos.data.items() if joint not in joints}
    )


def run_demo(
    hand,
    demo_name: str = "main",
    cycles: int = 1,
    num_steps: int = NUM_STEPS,
    step_size: float = STEP_SIZE,
    return_to_neutral: bool = True,
) -> tuple[str, ...]:
    """Play a named demo sequence on *hand* and return the pose names played."""
    demos = load_demo_poses()
    if demo_name not in demos:
        available = ", ".join(sorted(demos))
        raise ValueError(f"Unknown demo '{demo_name}'. Available demos: {available}.")

    demo = demos[demo_name]
    disabled_joints = set(hand.config.disabled_joint_ids)
    if disabled_joints:
        print(f"disabled_motor_ids in config: skipping {', '.join(sorted(disabled_joints))}")

    poses = {
        name: _drop_joints(hand.pose_from_fractions(fractions), disabled_joints)
        for name, fractions in demo.pose_fractions.items()
    }
    # Built manually rather than hand.set_neutral_position() so disabled joints
    # stay excluded from the return-to-neutral command too.
    neutral = _drop_joints(
        OrcaJointPositions.from_dict(dict(hand.config.neutral_position)), disabled_joints
    )

    for _ in range(cycles):
        for name in demo.sequence:
            hand.set_joint_positions(poses[name], num_steps=num_steps, step_size=step_size)

        if return_to_neutral:
            hand.set_joint_positions(neutral, num_steps=num_steps, step_size=step_size)

    return demo.sequence
