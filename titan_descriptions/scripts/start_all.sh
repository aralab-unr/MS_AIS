#!/bin/bash

# -----------------------
# 1. Load CAN Interface (needs sudo)
# -----------------------
# echo "[INFO] Loading CAN interface as root..."

# sudo bash -c "
#     if ! lsmod | grep -q gs_usb; then
#         echo '[SUDO] Loading gs_usb module...'
#         modprobe gs_usb
#     else
#         echo '[SUDO] gs_usb module already loaded.'
#     fi

#     current_bitrate=\$(ip -details link show can0 | grep -oP 'bitrate \K[^ ]+' || echo 'down')

#     if [[ \$current_bitrate != '500000' ]]; then
#         echo '[SUDO] Setting up CAN0 interface...'
#         ip link set can0 up type can bitrate 500000 || exit 1
#     else
#         echo '[SUDO] CAN0 already set up with bitrate 500000.'
#     fi
# "

# --------------------------
# 5. Launch ROS Bringup Node
# --------------------------
echo "Launching Bunker Pro..."
roslaunch bunker_bringup bunker_robot_base.launch &
PID_BASE=$!

# --------------------------
# 6. Launch Titan Perception (after delay)
# --------------------------
sleep 2
echo "Launching Perception Module..."
roslaunch titan_perception perception.launch &
PID_PERCEPTION=$!

# --------------------------
# 7. Launch Titan Description Cameras (after additional delay)
# --------------------------
sleep 5
echo "Launching Inspection Module..."
roslaunch add_obstacle xarm.launch &
PID_CAMERAS=$!

# --------------------------
# 7. Launch Titan Description Cameras (after additional delay)
# --------------------------
sleep 10
echo "Launching GUI..."
rosrun culvert_sim gui.py &
PID_GUI=$!

# --------------------------
# 8. Wait for all processes
# --------------------------
wait $PID_BASE
wait $PID_PERCEPTION
wait $PID_CAMERAS
wait $PID_GUI
