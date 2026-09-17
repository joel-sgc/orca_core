"""Read each motor's Min/Max_Voltage_Limit (EEPROM) and live Present_Input_Voltage.

Min/Max_Voltage_Limit is a firmware-level window: if the motor's actual
supply voltage falls outside it, the motor latches a Hardware Error Status
"Input Voltage" bit and orca_core's alert handling reboots it. Comparing
this window and the live reading across all motors distinguishes two very
different problems:

- If Min/Max_Voltage_Limit is *inconsistent* across motors (like we found
  with Current_Limit earlier), some motors trip on a normal sag simply
  because their configured window is tighter -- a leftover-bring-up config
  problem, not a wiring problem.
- If Present_Input_Voltage genuinely dips lower on specific motors under
  load (rerun this while the hand is mid-grasp, or right after a
  main_demo.py run that trips the error), that points at real voltage sag
  on whatever wiring/connector feeds those motors.

Usage:
    uv run python check_voltage_limits.py /dev/ttyUSB0 3000000 1,2,3,...,17
"""
import argparse

import dynamixel_sdk as dxl

PROTOCOL_VERSION = 2.0
ADDR_MAX_VOLTAGE_LIMIT = 32
ADDR_MIN_VOLTAGE_LIMIT = 34
ADDR_PRESENT_INPUT_VOLTAGE = 144
VOLT_SCALE = 0.1  # V per LSB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("baudrate", type=int)
    parser.add_argument("motor_ids", help="comma-separated, e.g. 1,2,3,...,17")
    args = parser.parse_args()

    motor_ids = [int(x) for x in args.motor_ids.split(",")]

    port_handler = dxl.PortHandler(args.port)
    packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise SystemExit(f"failed to open {args.port}")
    if not port_handler.setBaudRate(args.baudrate):
        raise SystemExit(f"failed to set baud {args.baudrate}")

    try:
        print(f"{'motor':>5}  {'Min_V':>6}  {'Max_V':>6}  {'Present_V':>10}")
        for motor_id in motor_ids:
            min_raw, c1, _ = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_MIN_VOLTAGE_LIMIT)
            max_raw, c2, _ = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_MAX_VOLTAGE_LIMIT)
            pres_raw, c3, _ = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_PRESENT_INPUT_VOLTAGE)
            if c1 != dxl.COMM_SUCCESS or c2 != dxl.COMM_SUCCESS or c3 != dxl.COMM_SUCCESS:
                print(f"{motor_id:5d}  read failed")
                continue
            print(f"{motor_id:5d}  {min_raw * VOLT_SCALE:6.1f}  {max_raw * VOLT_SCALE:6.1f}  "
                  f"{pres_raw * VOLT_SCALE:10.1f}")
    finally:
        port_handler.closePort()


if __name__ == "__main__":
    main()
