#!/bin/bash

# Launch first file (in background)
roslaunch titan_perception perception.launch &
PID1=$!

# Optional: wait a few seconds if needed
sleep 5

# Launch second file (in background)
roslaunch add_obstacle xarm.launch &
PID2=$!

# Wait for both to exit (keeps terminal open)
wait $PID1
wait $PID2
