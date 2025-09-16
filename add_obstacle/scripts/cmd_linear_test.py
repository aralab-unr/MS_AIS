#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import time

def publish_velocity(pub, twist_msg, duration):
    """
    Continuously publish a Twist message for a given duration.
    """
    start_time = time.time()
    rate = rospy.Rate(10)  # 10 Hz publishing rate
    while time.time() - start_time < duration:
        pub.publish(twist_msg)
        rate.sleep()

def move_robot():
    # Initialize the ROS node
    rospy.init_node('robot_mover', anonymous=True)

    # Create a publisher for the /cmd_vel topic
    pub = rospy.Publisher('/smoother_cmd_vel', Twist, queue_size=10)

    # Define the Twist message for moving forward
    move_forward = Twist()
    move_forward.linear.x = -0.1  # Adjust speed as needed
    move_forward.angular.z = 0.0

    # Define the Twist message for stopping
    stop = Twist()

    # Move forward for 2 seconds
    rospy.loginfo("Moving forward for 2 seconds...")
    publish_velocity(pub, move_forward, 4)

    # Stop for 2 seconds
    rospy.loginfo("Stopping for 2 seconds...")
    publish_velocity(pub, stop, 2)

    # Move forward again for 2 seconds
    rospy.loginfo("Moving forward for another 2 seconds...")
    publish_velocity(pub, move_forward, 4)

    # Stop after the second move
    rospy.loginfo("Stopping...")
    publish_velocity(pub, stop, 2)

if __name__ == "__main__":
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
