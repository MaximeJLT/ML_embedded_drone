from pymavlink import mavutil
import time

from read_gps import get_lat_lon_relalt
from connection import send_gcs_heartbeat


def _gcs_keepalive_tick(master, last_hb, period_s=1.0):
    """Send GCS heartbeat every period_s seconds. Returns updated last_hb."""
    now = time.time()
    if now - last_hb >= period_s:
        send_gcs_heartbeat(master)
        return now
    return last_hb


class _RCThrottleKeepAlive:
    """
    Envoie rc_channels_override (throttle 1500) toutes les 100 ms dans un
    thread de fond, pour maintenir l'altitude en QLOITER pendant les operations
    bloquantes (configure_failsafes, mission_upload).

    Usage :
        with _RCThrottleKeepAlive(master):
            configure_failsafes_for_flight(master)
            upload_mission_from_file(master, "...")
        # a la sortie du with, l'override est relache proprement
    """
    def __init__(self, master, interval_s=0.1):
        self._master   = master
        self._interval = interval_s
        self._stop     = False
        self._thread   = None

    def _run(self):
        IGN = 65535
        while not self._stop:
            try:
                self._master.mav.rc_channels_override_send(
                    self._master.target_system,
                    self._master.target_component,
                    1500, 1500, 1500, 1500,
                    IGN, IGN, IGN, IGN
                )
            except Exception:
                pass
            time.sleep(self._interval)

    def __enter__(self):
        self._stop   = False
        self._thread = __import__("threading").Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
        # Relacher proprement tous les overrides
        try:
            self._master.mav.rc_channels_override_send(
                self._master.target_system,
                self._master.target_component,
                0, 0, 0, 0, 0, 0, 0, 0
            )
        except Exception:
            pass


def _drain_statustext(master, n=10):
    """Print a few queued STATUSTEXT messages (non-blocking)."""
    for _ in range(n):
        st = master.recv_match(type="STATUSTEXT", blocking=False)
        if not st:
            break
        txt = st.text
        if isinstance(txt, (bytes, bytearray)):
            txt = txt.decode(errors="ignore")
        print("STATUSTEXT:", txt)

def release_rc_override(master):
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )

def _set_param(master, name: str, value: float,
               ptype=mavutil.mavlink.MAV_PARAM_TYPE_REAL32, timeout=3.0):
    """
    Envoie un PARAM_SET et attend l'echo PARAM_VALUE.
    Le type MAVLink doit correspondre au type reel du parametre ArduPlane :
      - REAL32  pour les floats  (FS_EKF_THRESH, etc.)
      - INT8    pour les entiers courts (FS_EKF_ACTION, FS_GCS_ENABL, etc.)
    Un type incorrect cause un NO_ECHO silencieux ArduPilot.
    """
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode("ascii"), float(value), ptype
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg:
            pid = msg.param_id
            if isinstance(pid, (bytes, bytearray)):
                pid = pid.decode(errors="ignore")
            pid = str(pid).strip("\x00")
            if pid == name:
                return True
    return False


def _read_param_float(master, name: str, timeout=3.0):
    """
    Lit la valeur courante d'un parametre via PARAM_REQUEST_READ.
    Retourne la valeur float, ou None si timeout.
    """
    master.mav.param_request_read_send(
        master.target_system, master.target_component,
        name.encode("ascii"), -1
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg:
            pid = msg.param_id
            if isinstance(pid, (bytes, bytearray)):
                pid = pid.decode(errors="ignore")
            pid = str(pid).strip("\x00")
            if pid == name:
                return float(msg.param_value)
    return None

def set_mode_and_confirm(master, mode_name: str, timeout=15):
    modes = master.mode_mapping()
    if mode_name not in modes:
        raise RuntimeError(f"Mode {mode_name} not available. Available: {list(modes.keys())}")

    master.set_mode(modes[mode_name])
    print(f"{mode_name} requested")

    t0 = time.time()
    last_hb = 0.0
    while time.time() - t0 < timeout:
        if time.time() - last_hb > 1.0:
            send_gcs_heartbeat(master)
            last_hb = time.time()

        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        _drain_statustext(master, n=10)

        if hb:
            try:
                print(f"DEBUG flightmode={master.flightmode} custom_mode={hb.custom_mode}")
            except Exception:
                pass
            if hb.custom_mode == modes[mode_name]:
                print(f"Mode {mode_name} confirmed")
                return True

    raise RuntimeError(f"{mode_name} timeout (mode not confirmed)")

### Action

def arm_and_wait(master, timeout = 15):
    """
    Send arming command and wait heartbeat
    """
    print("arming...")

    # 1 = Arm, 0 = Disarm
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0 
    )

    t0 = time.time()
    last_hb = t0
    
    while time.time() - t0 < timeout:
        if time.time() - last_hb > 1.0:
            send_gcs_heartbeat(master)
            last_hb = time.time()
        _drain_statustext(master, n=5)
        
        # Attendre le prochain HEARTBEAT
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg:
          
            if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                print("Armed")
                return True
                
    raise RuntimeError("Timeout")

def takeoff(master, target_altitude: float, timeout = 20):
    """
    send takeoff order at a target altitude
    """
    print("sending takeoff order...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, 
        target_altitude  
    )
    
    t0 = time.time()
    last_hb = t0
    
    while time.time() - t0 < timeout:
        if time.time() - last_hb > 1.0:
            send_gcs_heartbeat(master)
            last_hb = time.time()
            
        _drain_statustext(master, n=5)
        
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if msg:
            current_alt = msg.relative_alt / 1000.0  
            print(f"Altitude courante : {current_alt:.2f}m / {target_altitude}m")
            
            if current_alt >= (target_altitude * 0.95):
                print("target altitude reached")
                return True
                
    raise RuntimeError("timeout")