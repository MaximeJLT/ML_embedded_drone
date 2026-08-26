import time
import math
import threading
import numpy as np
import cv2
from pymavlink import mavutil
from enum import Enum, auto
from velocity import send_velocity_once

from arm_pipeline import (
    set_mode_and_confirm,
    arm_and_wait,
    takeoff,
    _drain_statustext,
    release_rc_override,
    _set_param,
)
from connection import send_gcs_heartbeat, connect_serial
from ml.conversion import GPS_target
from ml.NN import get_normalized_coordinates
import ml.NN as nn_module
from ml import conversion

METERS_PER_DEG_LAT = 111320.0
WP_APPROACH_RADIUS = 3.0
TARGET_SPEED = 5.0

class State(Enum):
    TAKEOFF         = auto()
    FOLLOW_LINE     = auto()
    TARGET_DETECTED = auto()
    RETURN_HOME     = auto()
    FAILSAFE        = auto()

def gcs_keepalive_tick(master, last_hb, period_s=1.0):
    now = time.time()
    if now - last_hb >= period_s:
        send_gcs_heartbeat(master)
        return now
    return last_hb

def meters_per_deg_lon(lat_deg: float) -> float:
    return 111320.0 * math.cos(math.radians(lat_deg))

def dist_to_wp_m(ref_lat, cur_lat, cur_lon, wp_lat, wp_lon):
    m_per_lon = meters_per_deg_lon(ref_lat)
    dN = (wp_lat - cur_lat) * METERS_PER_DEG_LAT
    dE = (wp_lon - cur_lon) * m_per_lon
    return math.sqrt(dN*dN + dE*dE)

def ned_to_latlon(center_lat, center_lon, dN_m, dE_m):
    lat = center_lat + dN_m / METERS_PER_DEG_LAT
    lon = center_lon + dE_m / meters_per_deg_lon(center_lat)
    return lat, lon

def generate_straight_line_wps(start_lat, start_lon, bearing_deg, total_dist_m, step_m=5.0):
    """
    Génère des points intermédiaires en ligne droite pour que le drone avance pas à pas.
    """
    wps = []
    num_points = int(total_dist_m / step_m)
    bearing_rad = math.radians(bearing_deg)
    
    for i in range(1, num_points + 1):
        d = i * step_m
        dN = d * math.cos(bearing_rad)
        dE = d * math.sin(bearing_rad)
        lat, lon = ned_to_latlon(start_lat, start_lon, dN, dE)
        wps.append((lat, lon))
    return wps

def send_guided_wp_global(master, lat, lon, alt_target_m):

    type_mask = 0b0000111111111000 
    
    master.mav.set_position_target_global_int_send(
        0,  
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, 
        type_mask,
        int(lat * 1e7),      
        int(lon * 1e7),      
        float(alt_target_m),
        0, 0, 0,              
        0, 0, 0,              
        0, 0                
    )

def wait_landed_and_disarmed(master, timeout=120):
    """Attend que le copter touche le sol et coupe ses moteurs."""
    t0 = time.time()
    last_hb = 0.0
    print("waiting landing...")
    while time.time() - t0 < timeout:
        last_hb = gcs_keepalive_tick(master, last_hb, period_s=1.0)
        
       
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            if not armed:
                print("Disarmed")
                return True
    raise RuntimeError("Timeout")

def compute_hold_point(drone_lat, drone_lon, target_lat, target_lon,
                       offset_m=3.0, hold_alt_m=2.5):

    dN = (drone_lat - target_lat) * METERS_PER_DEG_LAT
    dE = (drone_lon - target_lon) * meters_per_deg_lon(target_lat)

    dist = math.sqrt(dN*dN + dE*dE)

    if dist < offset_m:
        return drone_lat, drone_lon, hold_alt_m

    offset_N = (dN / dist) * offset_m
    offset_E = (dE / dist) * offset_m

    hold_lat = target_lat + offset_N / METERS_PER_DEG_LAT
    hold_lon = target_lon + offset_E / meters_per_deg_lon(target_lat)

    return hold_lat, hold_lon, hold_alt_m

def setup_flight_params(master):
    """Params de MISSION — source de vérité = ce fichier, pas la carte.
    Params matériels (SR1_, baud, GPS) restent dans le Pixhawk."""
    params = {
        "WPNAV_SPEED": 200,
        "WP_YAW_BEHAVIOR": 1,     
    }
    for name, value in params.items():
        if not _set_param(master, name, value):
            raise RuntimeError(f"{name} non confirmé par le Pixhawk — abort avant décollage")
        print(f"  param {name} = {value} OK")

def main():
    ALT_TARGET_M = 4.0
    DT           = 0.5

    master = connect_serial()
    print("initializing")
    
    threading.Thread(
        target=get_normalized_coordinates,
        args=(nn_module.cam_forward,),
        daemon=True
    ).start()
    print("thread détection démarré")
    time.sleep(3.0) 

    setup_flight_params(master)

    home_lat, home_lon = None, None
    while home_lat is None:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=15)
        if msg:
            home_lat = msg.lat / 1e7
            home_lon = msg.lon / 1e7
    print(f"home: lat={home_lat:.7f}, lon={home_lon:.7f}")

    # ---- DECOLLAGE GUIDED ----
    set_mode_and_confirm(master, "GUIDED")
    arm_and_wait(master, timeout=15)
    takeoff(master, target_altitude=ALT_TARGET_M, timeout=25)
    release_rc_override(master)

    # ---- LIGNE DROITE ----
    waypoints = generate_straight_line_wps(home_lat, home_lon, bearing_deg=11, total_dist_m=60.0, step_m=5.0)
    wp_index = 0
    last_sent_wp_index = -1 
    state = State.FOLLOW_LINE
    print(f"suivi de ligne: {len(waypoints)} points")

    last_gcs_hb = time.time()
    last_lat, last_lon = home_lat, home_lon

    send_guided_wp_global(master, *waypoints[0], ALT_TARGET_M)

    start_time = time.time()
    target_simulated = False

    while True:
        last_gcs_hb = gcs_keepalive_tick(master, last_gcs_hb, period_s=1.0)
        _drain_statustext(master, n=2)

        position_mise_a_jour = False
        while True:
            msg = master.recv_match(blocking=False)
            if msg is None:
                break 
            
            mtype = msg.get_type()
            if mtype == "GLOBAL_POSITION_INT":
                last_lat = msg.lat / 1e7
                last_lon = msg.lon / 1e7
                position_mise_a_jour = True
                
        if position_mise_a_jour:
            print(f"[DEBUG POS] Drone actuellement à Lat: {last_lat:.7f} | Lon: {last_lon:.7f}")

        if state == State.FOLLOW_LINE:
            if wp_index >= len(waypoints):
                state = State.RETURN_HOME
                continue

            tgt_lat, tgt_lon = waypoints[wp_index]

            if wp_index != last_sent_wp_index:
                send_guided_wp_global(master, tgt_lat, tgt_lon, ALT_TARGET_M)
                print(f"-> WP {wp_index+1} envoyé nativement à Ardupilot")
                last_sent_wp_index = wp_index

            msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=0.2)
            if msg is not None:
                last_lat = msg.lat / 1e7
                last_lon = msg.lon / 1e7
                
                d = dist_to_wp_m(tgt_lat, last_lat, last_lon, tgt_lat, tgt_lon)
                print(f"Navigation native active | Distance au WP {wp_index+1} : {d:.1f}m", end="\r")

                if d <= WP_APPROACH_RADIUS:
                    print(f"\n[+] WP {wp_index+1} ATTEINT !")
                    wp_index += 1  

            det = nn_module.latest_detection
            if det is not None and not target_simulated:
                state = State.TARGET_DETECTED

        elif state == State.TARGET_DETECTED:
            if nn_module.latest_detection is None:
                print("\n[TARGET_DETECTED] Cible perdue. Retour au suivi de ligne.")
                state = State.FOLLOW_LINE
                continue

            target_lat, target_lon = conversion.GPS_target(master)
            
            if target_lat is None:
                if wp_index < len(waypoints):
                    tgt_lat, tgt_lon = waypoints[wp_index]
                    if wp_index != last_sent_wp_index:
                        send_guided_wp_global(master, tgt_lat, tgt_lon, ALT_TARGET_M)
                        last_sent_wp_index = wp_index

                    d = dist_to_wp_m(tgt_lat, last_lat, last_lon, tgt_lat, tgt_lon)
                    if d <= WP_APPROACH_RADIUS:
                        wp_index += 1
                        
                print("[TARGET_DETECTED] Cible verrouillée, approche en cours...", end="\r")
                continue 
                
            print(f"\n[!] SEUIL ATTEINT ! Calcul du point de hold...")

            hold_lat, hold_lon, hold_alt_m = compute_hold_point(
                last_lat, last_lon, target_lat, target_lon,
                offset_m=3.0, hold_alt_m=2.5
            )
            print(f"point de hold: lat={hold_lat:.7f} lon={hold_lon:.7f} alt={hold_alt_m}m")

            send_guided_wp_global(master, hold_lat, hold_lon, hold_alt_m)
            print("navigation vers le point de hold...")

            while True:
                last_gcs_hb = gcs_keepalive_tick(master, last_gcs_hb, period_s=1.0)

                msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
                if msg is not None:
                    last_lat = msg.lat / 1e7
                    last_lon = msg.lon / 1e7

                    d = dist_to_wp_m(hold_lat, last_lat, last_lon, hold_lat, hold_lon)
                    print(f"distance au point de hold: {d:.1f}m", end="\r")

                    if d <= WP_APPROACH_RADIUS:
                        print(f"\narrivé au point de hold (d={d:.1f}m)")
                        break

            print("HOLD 6s devant la boîte...")
            t_hold = time.time()
            while time.time() - t_hold < 6.0:
                last_gcs_hb = gcs_keepalive_tick(master, last_gcs_hb, period_s=1.0)
                send_guided_wp_global(master, hold_lat, hold_lon, hold_alt_m)
                time.sleep(0.5)

            print("remontée à l'altitude de détection...")
            send_guided_wp_global(master, hold_lat, hold_lon, ALT_TARGET_M)
            time.sleep(3.0)  

            print("reprise de la ligne")
            target_simulated = True
            last_sent_wp_index = -1   
            state = State.FOLLOW_LINE



        elif state == State.RETURN_HOME:
            print("RTL")
            set_mode_and_confirm(master, "RTL", timeout=15)
            wait_landed_and_disarmed(master, timeout=180)
            print("fini")
            break

        time.sleep(DT)
if __name__ == "__main__":
    main()
