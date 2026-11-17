# Student Task
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = '127.0.0.1'  # Replace if needed
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)

def safe_move(target_q, speed=0.3):
    if len(target_q) != 6:
        print("Invalid target: need 6 joints")
        return False
    if speed > 0.5:
        speed = 0.5
    if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
        print("Robot stopped — cannot move")
        return False
    try:
        acceleration=0.3
        ok = rtde_c.moveJ(target_q, speed, acceleration)
        print("Move OK" if ok else "Move failed")
        return ok
    except Exception as e:
        print("Move error:", e)
        return False

safe_move([0, 0, 0, 0, 0, 0], speed=0.3)

rtde_c.disconnect()
rtde_r.disconnect()