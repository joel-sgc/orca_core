"""Hybrid calibration: place each joint at its limit by hand (torque off),
then let the motor take over for a short current-driven "tug" that seats it
firmly against the true hardstop under real string tension, before recording
that final motor position. Walks every joint the same way calibrate.py's
automated sequence does, just with a human placement step per direction
instead of driving the whole range under motor power.

This splits the difference between full automation (which can plateau early
on friction/stiction well short of a joint's true range) and pure manual
jogging (which misses the string-stretch-under-load / human-caution gap
between "feels resistant" and "truly at the hard stop"). The human gets the
motor into the right neighborhood fast, without it ever having to fight
through friction across the whole range; the motor then does a short,
small-step "tug" using the same step+stability logic as
maintenance/calibration_routine.py, so the final recorded position is a real
tension-loaded reading, not a hand-guessed one -- and for non-wrist joints,
mirrors the real routine's torque-release-and-reread step afterward, so the
recorded value has the same tension-bias handling as automated calibration.

Writes into calibration.yaml after every joint (not just at the end), using
the same merge/persist helpers the real calibration routine uses -- an
interrupted session (Ctrl+C between joints) never loses completed work.
Joints that already have valid motor limits + a nonzero ratio are skipped by
default; pass --force to redo them anyway.

Usage:
    uv run python hybrid_calibrate_joint.py CONFIG
    uv run python hybrid_calibrate_joint.py CONFIG --joints thumb_mcp thumb_abd
    uv run python hybrid_calibrate_joint.py CONFIG --force --dry-run
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from orca_core import OrcaHand
from orca_core.constants import CURRENT_BASED_POSITION, EXTEND, FLEX, TINY_SLEEP, WRIST
from orca_core.maintenance.calibration_routine import (
    _build_calibration_result,
    _persist_calibration,
)

MAX_TUG_ITERATIONS = 1000
"""Safety cap: the tug should settle in well under this if the human
placement was close. Hitting the cap means something's wrong -- abort loudly
rather than run indefinitely."""


def tug_to_stable(hand, motor_id, sign, step_size, step_period, num_stable, threshold):
    """Drive small steps in `sign`'s direction until position stabilizes, then
    release torque and reread to remove tension bias. Returns the recorded
    motor position, or None if it never stabilized. Not used for the wrist --
    see the note in calibrate_direction().
    """
    idx = hand.config.motor_id_to_idx_dict[motor_id]
    buffer = deque(maxlen=num_stable)
    last_pos = None

    for _ in range(MAX_TUG_ITERATIONS):
        hand._set_motor_pos({motor_id: sign * step_size}, rel_to_current=True)
        time.sleep(step_period)
        with hand._motor_lock:
            state = hand.motor_client.read_position_velocity_current()
            read_ok = hand.motor_client.last_read_ok
        if not read_ok:
            continue
        last_pos = float(state.position[idx])
        buffer.append(last_pos)
        if len(buffer) == num_stable and np.allclose(buffer, buffer[0], atol=threshold):
            break
    else:
        return None

    failed = hand.disable_torque([motor_id])
    if failed:
        print(f"  warning: torque release not acknowledged for motor {motor_id}; "
              "recorded value may include tension bias")
        return last_pos
    time.sleep(TINY_SLEEP)
    with hand._motor_lock:
        pos = hand.get_motor_pos()
        read_ok = hand.motor_client.last_read_ok
    if not read_ok:
        print(f"  warning: post-release read failed for motor {motor_id}; using pre-release value")
        return last_pos
    return float(pos[idx])


def direction_sign(joint, direction, config) -> int:
    sign = 1 if direction == FLEX else -1
    if config.joint_inversion_dict.get(joint, False):
        sign = -sign
    return sign


def calibrate_direction(hand, joint, motor_id, direction, config) -> float | None:
    hand.disable_torque([motor_id])
    input(f"\nMove {joint} to its {direction.upper()} limit by hand, then press Enter...")

    if joint == WRIST:
        # The wrist's calibration mode (multi_turn_position, forced by
        # set_control_mode's wrist special-case) doesn't respect Goal_Current
        # as a torque limit the way current_based_position does -- a
        # motor-driven tug here isn't actually current-limited, it drives at
        # full available torque and can slip the belt. Record the hand-placed
        # position directly instead.
        with hand._motor_lock:
            pos = hand.get_motor_pos()
            read_ok = hand.motor_client.last_read_ok
        if not read_ok:
            print(f"  {direction}: position read failed -- aborting.")
            return None
        result = float(pos[config.motor_id_to_idx_dict[motor_id]])
        print(f"  {direction}: recorded hand-placed position {result:.4f} rad (no motor tug)")
        return result

    sign = direction_sign(joint, direction, config)
    hand.set_control_mode(CURRENT_BASED_POSITION, [motor_id])
    hand.set_max_current(config.calibration_current)
    hand.enable_torque([motor_id])

    with hand._motor_lock:
        seed_pos = hand.get_motor_pos()[config.motor_id_to_idx_dict[motor_id]]

    result = tug_to_stable(
        hand, motor_id, sign,
        config.calibration_step_size, config.calibration_step_period,
        config.calibration_num_stable, config.calibration_threshold,
    )
    if result is None:
        print(f"  {direction}: never stabilized within {MAX_TUG_ITERATIONS} iterations -- aborting.")
        hand.disable_torque([motor_id])
        return None

    print(f"  {direction}: seeded at {seed_pos:.4f} rad, settled at {result:.4f} rad "
          f"(tug moved it {abs(result - seed_pos):.4f} rad)")
    return result


def joint_already_calibrated(hand, joint) -> bool:
    motor_id = hand.config.joint_to_motor_map[joint]
    limits = hand.calibration.motor_limits_dict.get(motor_id)
    ratio = hand.calibration.joint_to_motor_ratios_dict.get(motor_id)
    return bool(limits) and None not in limits and bool(ratio)


def calibrate_joint(hand, joint, config) -> "tuple[list[float], float] | None":
    """Run flex+extend hybrid capture for one joint.

    Returns (motor_limits, joint_to_motor_ratio), or None if a direction
    failed to stabilize.
    """
    motor_id = config.joint_to_motor_map[joint]
    limits = {}
    for direction in (FLEX, EXTEND):
        pos = calibrate_direction(hand, joint, motor_id, direction, config)
        if pos is None:
            return None
        bound = 1 if direction_sign(joint, direction, config) == 1 else 0
        limits[bound] = pos

    motor_limits = [limits[0], limits[1]]
    delta_motor = motor_limits[1] - motor_limits[0]
    delta_joint = config.joint_roms_dict[joint][1] - config.joint_roms_dict[joint][0]
    ratio = float(delta_motor / delta_joint)
    return motor_limits, ratio


def persist_joint_result(hand, joint, motor_limits, ratio) -> None:
    config = hand.config
    motor_id = config.joint_to_motor_map[joint]

    new_limits = dict(hand.calibration.motor_limits_dict)
    new_ratios = dict(hand.calibration.joint_to_motor_ratios_dict)
    new_limits[motor_id] = motor_limits
    new_ratios[motor_id] = ratio

    result = _build_calibration_result(
        motor_limits=new_limits,
        joint_to_motor_ratios=new_ratios,
        wrist_calibrated=hand.calibration.wrist_calibrated or joint == WRIST,
        joint_encoder_calibration_dict=hand.calibration.joint_encoder_calibration_dict,
    )
    hand.calibration = result
    _persist_calibration(config.calibration_path, result=result, include_encoder=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("config_path")
    parser.add_argument(
        "--joints", nargs="+", default=None,
        help="joint names to calibrate, in this order (default: every joint, in config order)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recalibrate joints even if they already have valid motor limits + a ratio",
    )
    parser.add_argument("--dry-run", action="store_true", help="don't write calibration.yaml")
    args = parser.parse_args()

    hand = OrcaHand(config_path=args.config_path)
    success, msg = hand.connect()
    if not success:
        raise SystemExit(f"connect failed: {msg}")

    try:
        config = hand.config
        joints = list(args.joints) if args.joints is not None else list(config.joint_ids)
        unknown = [j for j in joints if j not in config.joint_to_motor_map]
        if unknown:
            raise SystemExit(f"unknown joint(s) {unknown}; expected one of {config.joint_ids}")

        if not args.force:
            skipped = [j for j in joints if joint_already_calibrated(hand, j)]
            joints = [j for j in joints if j not in skipped]
            if skipped:
                print(f"Skipping already-calibrated joints: {skipped} (pass --force to redo them)")

        if not joints:
            print("Nothing to calibrate.")
            return

        print(f"\nCalibrating {len(joints)} joint(s): {joints}")
        print("(Ctrl+C is safe between joints -- progress is saved after each one.)\n")

        for i, joint in enumerate(joints, start=1):
            print(f"=== [{i}/{len(joints)}] {joint} ===")
            outcome = calibrate_joint(hand, joint, config)
            if outcome is None:
                print(f"  {joint}: a direction failed to stabilize -- skipping this joint, "
                      "continuing with the rest.\n")
                continue

            motor_limits, ratio = outcome
            motor_id = config.joint_to_motor_map[joint]
            print(f"  motor {motor_id} ({joint}): motor_limits={motor_limits}, "
                  f"joint_to_motor_ratio={ratio:.6f}")

            if args.dry_run:
                print("  (--dry-run: not writing calibration.yaml)\n")
                continue

            persist_joint_result(hand, joint, motor_limits, ratio)
            print("  wrote calibration.yaml\n")

        print("Done.")
    finally:
        hand.set_max_current(hand.config.max_current)
        hand.disable_torque()
        hand.disconnect()


if __name__ == "__main__":
    main()
