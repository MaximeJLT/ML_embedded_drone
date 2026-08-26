import time
from pymavlink import mavutil
from serial.serialutil import SerialException


def _set_msg_interval(master, msg_id, hz):
    interval_us = int(1e6 / hz) if hz > 0 else 0
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0, 0, 0, 0, 0
    )

def connect_serial(device="/dev/ttyUSB0", baud=57600, heartbeat_timeout=10):
    
    master = None
    while master is None:
        try:
            master = mavutil.mavlink_connection(device, baud=baud)
            msg = master.wait_heartbeat(timeout=heartbeat_timeout)
            
            if msg is None:
                raise TimeoutError("Timeout heartbeat")
                
            print(f"Heartbeat ok (system={master.target_system}, component={master.target_component})")

        except (SerialException, OSError, TimeoutError):
            time.sleep(2)
            master = None

    _set_msg_interval(master, 33,  10)
    _set_msg_interval(master, 245,  5)
    _set_msg_interval(master, 74,   5)
    _set_msg_interval(master, 1,    4)
    _set_msg_interval(master, 30,   2)

    time.sleep(0.3)
    return master

def send_gcs_heartbeat(master):
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0
    )