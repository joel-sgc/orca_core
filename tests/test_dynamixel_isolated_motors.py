"""Tests for DynamixelClient's isolated_motor_ids.

A motor listed there is excluded from the shared GroupBulkRead/GroupSyncWrite
and talked to individually instead, so a motor on a different physical bus
segment (e.g. RS485 off the same adapter as a TTL daisy chain) can't be
bundled into the same multi-motor transaction as the rest.

Uses a fake dynamixel_sdk (patched into sys.modules), matching the pattern in
test_dynamixel_client_locking.py, so no hardware or serial port is touched.
"""

import sys
import types

import numpy as np
import pytest

from orca_core.hardware.dynamixel_client import DynamixelClient

COMM_SUCCESS = 0
COMM_RX_FAIL = -3001
COMM_NOT_AVAILABLE = -3002


class FakeBus:
    def __init__(self):
        self.bulk_read_ids = []        # motor IDs ever added to a GroupBulkRead
        self.sync_write_ids = []       # motor IDs ever added to a GroupSyncWrite
        self.individual_read_ids = []  # motor IDs read via read2/4ByteTxRx
        self.individual_write_ids = []  # motor IDs written via write4ByteTxRx
        self.read4_hook = None         # callable(motor_id) -> (value, comm, err)


def make_fake_sdk(bus):
    class PortHandler:
        def __init__(self, port):
            self.is_open = False
            self.is_using = False

        def openPort(self):
            self.is_open = True
            return True

        def setBaudRate(self, baudrate):
            return True

        def closePort(self):
            self.is_open = False

    class PacketHandler:
        def __init__(self, protocol_version):
            pass

        def read1ByteTxRx(self, port, motor_id, address):
            return 0, COMM_SUCCESS, 0

        def write1ByteTxRx(self, port, motor_id, address, value):
            return COMM_SUCCESS, 0

        def read2ByteTxRx(self, port, motor_id, address):
            bus.individual_read_ids.append(motor_id)
            return 0, COMM_SUCCESS, 0

        def write2ByteTxRx(self, port, motor_id, address, value):
            bus.individual_write_ids.append(motor_id)
            return COMM_SUCCESS, 0

        def read4ByteTxRx(self, port, motor_id, address):
            bus.individual_read_ids.append(motor_id)
            if bus.read4_hook is not None:
                return bus.read4_hook(motor_id)
            return 0, COMM_SUCCESS, 0

        def write4ByteTxRx(self, port, motor_id, address, value):
            bus.individual_write_ids.append(motor_id)
            return COMM_SUCCESS, 0

        def readRx(self, port, motor_id, length):
            return bytes(length), COMM_SUCCESS, 0

        def getTxRxResult(self, comm_result):
            return f'comm_result={comm_result}'

        def getRxPacketError(self, dxl_error):
            return f'dxl_error={dxl_error}'

    class GroupBulkRead:
        def __init__(self, port, packet_handler):
            self.port = port
            self.ph = packet_handler
            self.data_dict = {}

        def addParam(self, motor_id, address, size):
            self.data_dict[motor_id] = [None, address, size]
            bus.bulk_read_ids.append(motor_id)
            return True

        def txPacket(self):
            return COMM_SUCCESS

        def isAvailable(self, motor_id, address, size):
            return True

        def getData(self, motor_id, address, size):
            return 0

    class GroupSyncWrite:
        def __init__(self, port, packet_handler, address, size):
            self.params = {}

        def addParam(self, motor_id, value):
            self.params[motor_id] = value
            bus.sync_write_ids.append(motor_id)
            return True

        def txPacket(self):
            return COMM_SUCCESS

        def clearParam(self):
            self.params = {}

    sdk = types.ModuleType('dynamixel_sdk')
    sdk.COMM_SUCCESS = COMM_SUCCESS
    sdk.COMM_RX_FAIL = COMM_RX_FAIL
    sdk.COMM_NOT_AVAILABLE = COMM_NOT_AVAILABLE
    sdk.PortHandler = PortHandler
    sdk.PacketHandler = PacketHandler
    sdk.GroupBulkRead = GroupBulkRead
    sdk.GroupSyncWrite = GroupSyncWrite
    return sdk


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def client(bus, monkeypatch, request):
    monkeypatch.setitem(sys.modules, 'dynamixel_sdk', make_fake_sdk(bus))
    dxl_client = DynamixelClient(
        [1, 2, 3], port='/dev/fake', baudrate=57600, isolated_motor_ids=[3]
    )
    dxl_client.port_handler.is_open = True  # pretend connected

    def cleanup():
        dxl_client.port_handler.is_open = False
        DynamixelClient.OPEN_CLIENTS.discard(dxl_client)

    request.addfinalizer(cleanup)
    return dxl_client


def test_isolated_motor_excluded_from_bulk_read(client, bus):
    assert 3 not in bus.bulk_read_ids
    assert set(bus.bulk_read_ids) == {1, 2}


def test_isolated_motor_read_individually(client, bus):
    client.read_position_velocity_current()
    assert 3 in bus.individual_read_ids


def test_isolated_motor_excluded_from_sync_write(client, bus):
    client.write_desired_pos([1, 2, 3], np.array([0.0, 0.0, 0.0]))
    assert 3 not in bus.sync_write_ids
    assert set(bus.sync_write_ids) == {1, 2}


def test_isolated_motor_written_individually(client, bus):
    client.write_desired_pos([1, 2, 3], np.array([0.0, 0.0, 0.0]))
    assert 3 in bus.individual_write_ids


def test_isolated_motor_failure_does_not_fail_grouped_read(client, bus):
    """A failing isolated motor must not mark the grouped motors' bulk
    transaction as failed — that's the exact bug this feature fixes."""
    bus.read4_hook = (
        lambda motor_id: (0, COMM_RX_FAIL, 0) if motor_id == 3 else (0, COMM_SUCCESS, 0)
    )
    client.read_position_velocity_current()
    # Overall read reports stale (the isolated motor failed)...
    assert not client.last_read_ok
    # ...but motors 1 and 2 were still part of one successful bulk transaction,
    # never blocked or aborted by motor 3's failure.
    assert set(bus.bulk_read_ids) == {1, 2}
