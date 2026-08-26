def wait_global_position_int(master, timeout=10):
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
    if msg is None:
        print("No GLOBAL_POSITION_INT received")
        raise SystemExit(1)
    return msg

def get_lat_lon_relalt(master, timeout=10):

    msg = wait_global_position_int(master, timeout=timeout)
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    rel_alt_m = msg.relative_alt / 1000.0
    return lat, lon, rel_alt_m
