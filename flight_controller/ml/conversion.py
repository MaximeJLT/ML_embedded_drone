import numpy as np
import math
from read_gps import get_lat_lon_relalt
from . import NN

def GPS_target(master):

    if NN.latest_detection is None:
        return None, None
    else:
        x, y = NN.latest_detection[:2]


    x_centré = x - 0.5
    y_centré = y - 0.5


    fov_horizontal = 62  
    fov_vertical = 48    
    cam_tilt_deg = 17.0

    angle_horizontal = x_centré * fov_horizontal
    angle_vertical = y_centré * fov_vertical


    lat, lon, altitude = get_lat_lon_relalt(master)

    depression_deg = cam_tilt_deg + angle_vertical
    depression_rad = math.radians(depression_deg)

    RELIABLE_DISTANCE = 20

    if depression_deg <= 1.0:
        print(f"[GPS_target] target too close from the horizon (depression={depression_deg:.1f}°), unreliable position")
        return None, None

    #offset au sol dans le repere drone (trigo)
    offset_forward_drone = altitude / math.tan(depression_rad)

    if offset_forward_drone > RELIABLE_DISTANCE:
        print(f"[GPS_target] target too far ({offset_forward_drone:.1f}m > {RELIABLE_DISTANCE}m), keep going forward")
        return None, None

    offset_right_drone = offset_forward_drone * math.tan(math.radians(angle_horizontal))

    msg = master.recv_match(type='ATTITUDE', blocking=True, timeout=5)
    uav_yaw = msg.yaw if msg is not None else 0.0   

    cos_y = math.cos(uav_yaw)
    sin_y = math.sin(uav_yaw)
    delta_nord = offset_forward_drone * cos_y - offset_right_drone * sin_y
    delta_est  = offset_forward_drone * sin_y + offset_right_drone * cos_y

    target_lat = lat + delta_nord / 111320 
    target_lon = lon + delta_est / (111320 * np.cos(np.radians(lat))) 

    return target_lat, target_lon