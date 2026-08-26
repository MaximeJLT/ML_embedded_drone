import time
import math
from pymavlink import mavutil

def condition_yaw(master, yaw_deg, yaw_rate_deg_s=60, direction=1, relative=True):

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        float(yaw_deg),          
        float(yaw_rate_deg_s),   
        float(direction),          
        1.0 if relative else 0.0, 
        0, 0, 0
    )

def wait(seconds):
    time.sleep(seconds)

def yaw_step(master, delta_deg, yaw_rate_deg_s=90):
    direction = 1 if delta_deg >= 0 else -1
    condition_yaw(master, yaw_deg=abs(delta_deg), yaw_rate_deg_s=yaw_rate_deg_s, direction=direction, relative=True)

def yaw_stop(master):
    pass


def yaw_step_rad(master, delta_rad, yaw_rate_rad_s=1.5):
    """
    delta_rad > 0 : CW, delta_rad < 0 : CCW (même logique que yaw_step)
    yaw_rate_rad_s : rad/s
    """
    delta_deg = math.degrees(delta_rad)
    yaw_rate_deg_s = math.degrees(yaw_rate_rad_s)
    yaw_step(master, delta_deg, yaw_rate_deg_s=yaw_rate_deg_s)

def yaw_rate_discrete(master, yaw_rate_rad_s, duration_s, dt=0.1):
    """
    Approxime un yaw_rate continu via petits yaw relatifs.
    """
    steps = int(duration_s / dt)
    for _ in range(steps):
        yaw_step_rad(master, yaw_rate_rad_s * dt, yaw_rate_rad_s=yaw_rate_rad_s)
        time.sleep(dt)
