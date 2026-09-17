"""Read each motor's raw Hardware Error Status byte (address 70).

Dynamixel firmware latches hardware errors -- once tripped, the bit stays
set (and torque is typically disabled by the firmware) until the motor is
rebooted. orca_core's own auto-recovery (check_overload_and_reboot,
_handle_hardware_alert) only watches for the Overload bit (0x20); an
Input Voltage trip (0x01) or any other bit is never noticed or cleared.
This just reads the raw byte so we can see what's actually latched right
now, before doing anything about it.

Bit layout (X-series):
    0x01  Input Voltage Error
    0x04  Overheating Error
    0x08  Motor Encoder Error
    0x10  Electrical Shock Error
    0x20  Overload Error

Usage:
    uv run python check_hardware_errors.py /dev/ttyUSB0 3000000 1,2,3,...,17
"""
import argparse

import dynamixel_sdk as dxl

PROTOCOL_VERSION = 2.0
ADDR_HARDWARE_ERROR_STATUS = 70

BIT_NAMES = {
    0x01: "Input Voltage",
    0x04: "Overheating",
    0x08: "Encoder",
    0x10: "Electrical Shock",
    0x20: "Overload",
}


def describe(byte: int) -> str:
    if byte == 0:
        return "clean"
    return ", ".join(name for bit, name in BIT_NAMES.items() if byte & bit) or f"unknown (0x{byte:02X})"


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
        print(f"{'motor':>5}  {'raw':>5}  status")
        for motor_id in motor_ids:
            value, comm, _ = packet_handler.read1ByteTxRx(port_handler, motor_id, ADDR_HARDWARE_ERROR_STATUS)
            if comm != dxl.COMM_SUCCESS:
                print(f"{motor_id:5d}  read failed: {packet_handler.getTxRxResult(comm)}")
                continue
            print(f"{motor_id:5d}  0x{value:02X}   {describe(value)}")
    finally:
        port_handler.closePort()


if __name__ == "__main__":
    main()
