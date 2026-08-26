import time
from pymavlink import mavutil

ALT_REL_M = 30.0

RC_THROTTLE_HOLD = 1500
RC_IGNORE = 65535


def navigate_to_target_vtol(master, tgt_lat, tgt_lon, alt_rel_m,
                             arrival_radius_m=15.0, timeout_s=120.0,
                             gcs_keepalive_fn=None):
    from arm_pipeline import set_mode_and_confirm

    print(f"\U0001f4cd Uploading approach mission (AUTO): lat={tgt_lat:.7f} lon={tgt_lon:.7f} alt={alt_rel_m}m")

    items = [
        dict(seq=0, frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=1, autocont=1,
             p1=0, p2=arrival_radius_m, p3=0, p4=0,
             lat=tgt_lat, lon=tgt_lon, alt=alt_rel_m),

        dict(seq=1, frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
             command=mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
             current=0, autocont=1,
             p1=0, p2=0, p3=0, p4=0,
             lat=tgt_lat, lon=tgt_lon, alt=alt_rel_m),
    ]

    _upload_items(master, items, keepalive_fn=gcs_keepalive_fn)

    master.mav.mission_set_current_send(master.target_system, master.target_component, 0)
    time.sleep(0.2)

    set_mode_and_confirm(master, "AUTO", timeout=15)

    t0 = time.time()
    last_lat = None
    last_lon = None
    last_print = 0.0

    while True:
        if gcs_keepalive_fn:
            gcs_keepalive_fn()

        pos = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if pos:
            last_lat = pos.lat / 1e7
            last_lon = pos.lon / 1e7

        elapsed = time.time() - t0

        if last_lat is not None:
            d = _dist_m(last_lat, last_lon, tgt_lat, tgt_lon)

            if time.time() - last_print > 5.0:
                print(f"   [AUTO] Navigating to target: d={d:.1f}m  ({elapsed:.0f}s / {timeout_s:.0f}s)")
                last_print = time.time()

            if d <= arrival_radius_m:
                print(f"\u2705 Target reached (d={d:.1f}m) after {elapsed:.1f}s staying in AUTO/LOITER_UNLIM")
                return True

        if elapsed > timeout_s:
            print(f"\u26a0\ufe0f  Navigation timeout ({elapsed:.0f}s) proceeding with hold at current position")
            return False


def upload_loiter_unlim(master, lat_deg, lon_deg, alt_rel_m=ALT_REL_M,
                        gcs_keepalive_fn=None):
    from arm_pipeline import set_mode_and_confirm

    items = [
        dict(seq=0, frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
             command=mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
             current=1, autocont=1,
             p1=0, p2=0, p3=0, p4=0,
             lat=lat_deg, lon=lon_deg, alt=alt_rel_m),
    ]
    _upload_items(master, items, keepalive_fn=gcs_keepalive_fn)
    master.mav.mission_set_current_send(master.target_system, master.target_component, 0)
    time.sleep(0.2)
    set_mode_and_confirm(master, "AUTO", timeout=15)


def send_hold_position(master, lat_deg, lon_deg, alt_rel_m=ALT_REL_M):
    master.mav.set_position_target_global_int_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111100, 
        int(lat_deg * 1e7),
        int(lon_deg * 1e7),
        float(alt_rel_m),
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0
    )
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        RC_IGNORE, RC_IGNORE, RC_THROTTLE_HOLD, RC_IGNORE,
        RC_IGNORE, RC_IGNORE, RC_IGNORE, RC_IGNORE
    )


def release_rc_override(master):
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )


def send_guided_GPS_target_one(master, lat_deg, lon_deg, alt_rel_m=ALT_REL_M):
    """Alias pour compatibilit�."""
    send_hold_position(master, lat_deg, lon_deg, alt_rel_m)



def _dist_m(lat1, lon1, lat2, lon2):
    import math
    MLAT = 111320.0
    dN = (lat2 - lat1) * MLAT
    dE = (lon2 - lon1) * (MLAT * math.cos(math.radians(lat1)))
    return math.sqrt(dN*dN + dE*dE)


def _upload_items(master, items, timeout=15, keepalive_fn=None):
    from connection import send_gcs_heartbeat
    n = len(items)

    send_gcs_heartbeat(master)

    # Vider buffer
    _flush(master, 0.3)

    # Clear
    master.mav.mission_clear_all_send(master.target_system, master.target_component)
    send_gcs_heartbeat(master) 
    time.sleep(0.3)
    _flush(master, 0.2)

    # Count
    master.mav.mission_count_send(master.target_system, master.target_component, n, 0)

    t0 = time.time()
    last_sent = -1
    last_hb = time.time()
    while time.time() - t0 < timeout:
        if time.time() - last_hb >= 0.5:
            send_gcs_heartbeat(master)
            if keepalive_fn:
                keepalive_fn()
            last_hb = time.time()

        msg = master.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        mtype = msg.get_type()

        if mtype in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
            seq = int(msg.seq)
            if 0 <= seq < n:
                it = items[seq]
                p4 = it["p4"] if it["p4"] == it["p4"] else 0.0  # NaN \u2192 0
                master.mav.mission_item_int_send(
                    master.target_system, master.target_component,
                    it["seq"], it["frame"], it["command"],
                    it["current"], it["autocont"],
                    float(it["p1"]), float(it["p2"]),
                    float(it["p3"]), float(p4),
                    int(it["lat"] * 1e7), int(it["lon"] * 1e7),
                    float(it["alt"])
                )
                last_sent = seq

        elif mtype == "MISSION_ACK":
            if getattr(msg, "type", -1) == 0:
                print(f"   \u2705 Mission uploaded ({n} items)")
                return True
            else:
                print(f"   \u26a0\ufe0f  Mission upload ACK type={msg.type}")
                return False

    print(f"   \u26a0\ufe0f  Mission upload timeout (sent up to seq={last_sent})")
    return False


def _flush(master, duration=0.3):
    t0 = time.time()
    while time.time() - t0 < duration:
        master.recv_match(blocking=False)