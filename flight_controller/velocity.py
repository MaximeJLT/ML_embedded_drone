from pymavlink import mavutil
import time

def send_velocity_ned(master, vx, vy, vz, duration_s): 
    type_mask = 0b0000111111000111 

    start = time.time()
    while time.time() - start < duration_s:
        master.mav.set_position_target_local_ned_send(
            0,  
            master.target_system, 
            master.target_component, 
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0, 0, 0,     
            vx, vy, vz,   
            0, 0, 0,    
            0, 0          
        )
        time.sleep(0.1)

def stop(master):
    send_velocity_ned(master, 0, 0, 0, 0.5) 

def send_velocity_once(master, vx, vy, vz): 
    """
    Envoie UNE consigne de vitesse (1 message).
    À appeler dans une boucle (ex: 10 Hz).
    """
    type_mask = 0b0000111111000111 
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
