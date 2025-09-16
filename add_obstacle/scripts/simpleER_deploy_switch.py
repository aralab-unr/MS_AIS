#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import *
from std_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from xarm_planner.srv import *
from xarm_msgs.srv import *
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
import tf2_ros
from math import *
from nde_er.srv import er
import os
from datetime import datetime
import csv
from cv_bridge import *
import cv2
from message_filters import *
def closest_quaternion(q1, q2, q_input):
    """
    Finds the closest quaternion to q_input from q1 and q2 using dot product similarity.

    Args:
        q1 (list or np.array): First quaternion [x, y, z, w].
        q2 (list or np.array): Second quaternion [x, y, z, w].
        q_input (list or np.array): Input quaternion [x, y, z, w].

    Returns:
        list: The closest quaternion as a list [x, y, z, w].
    """
    # Convert inputs to NumPy arrays
    q1_np = np.array(q1)
    q2_np = np.array(q2)
    q_input_np = np.array(q_input)

    # Normalize quaternions
    q1_np /= np.linalg.norm(q1_np)
    q2_np /= np.linalg.norm(q2_np)
    q_input_np /= np.linalg.norm(q_input_np)

    # Compute dot product similarity
    dot1 = np.dot(q1_np, q_input_np)
    dot2 = np.dot(q2_np, q_input_np)

    # Return the closest quaternion as a list
    return q1 if abs(dot1) > abs(dot2) else q2
# Convert quaternion to direction vector
def quaternion_to_direction(quaternion):
    """
    Convert a quaternion into a forward direction vector (x-axis).
    Args:
        quaternion (tuple): Quaternion (x, y, z, w).
    Returns:
        numpy array: Normalized forward direction vector.
    """
    x, y, z, w = quaternion
    # Forward direction (x-axis) from quaternion
    direction_vector = np.array([
        1 - 2 * (y**2 + z**2),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ])
    return direction_vector / np.linalg.norm(direction_vector)

# Generate points function with fixed distance
def generate_points(xyz, quaternion, point_distance=0.2, start_offset=0.19, intermediate_offset=0.33, final_offset=0.5):
    """
    Generate an array of points in a given direction with specified distances.    Args:
        xyz (tuple): Starting position (x, y, z).
        quaternion (tuple): Orientation as quaternion (x, y, z, w).
        point_distance (float): Distance between consecutive intermediate points.
        start_offset (float): Starting offset in meters.
        intermediate_offset (float): Ending offset for intermediate points.
        final_offset (float): Offset for the final point, if None then no add.    
    Returns:
        list: List of generated points as (x, y, z).
    """
    direction_vector = quaternion_to_direction(quaternion)    # Generate intermediate points
    start_point = np.array(xyz) + direction_vector * start_offset
    end_point = np.array(xyz) + direction_vector * intermediate_offset
    total_distance = np.linalg.norm(end_point - start_point)
    num_points = int(np.floor(total_distance / point_distance)) + 1    
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    intermediate_points = start_point + t * (end_point - start_point)    # Add the final point at 0.5 meters
    if final_offset is not None:
        final_point = np.array(xyz) + direction_vector * final_offset
        intermediate_points = np.vstack((intermediate_points, final_point))    
    return intermediate_points

def create_timestamped_folders(base_dir):
    """
    Create a timestamped folder inside the base directory with subfolders depth, rgb, and er.    Args:
        base_dir (str): The base directory where the timestamped folder will be created.    Returns:
        dict: A dictionary containing the paths to the subfolders.
    """
    # Get the current timestamp and date
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # Create the main timestamped folder
    timestamped_folder = os.path.join(base_dir, timestamp)
    os.makedirs(timestamped_folder, exist_ok=True)    # Create the subfolders and retain their paths
    subfolder_paths = {}
    for subfolder in ["depth", "rgb", "er"]:
        subfolder_path = os.path.join(timestamped_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        subfolder_paths[subfolder] = subfolder_path    
    return subfolder_paths
    
class XArmPosePublisherAndPlanner:
    def __init__(self):
        rospy.init_node("xarm_pose_publisher_and_planner")
        self.start = False
        # Publishers
        self.pose_pub = rospy.Publisher("/target_pose", PoseStamped, queue_size=1)
        self.er_waypoints_pub = rospy.Publisher("/ER_waypoints", Marker, queue_size=10)
        self.pump_pub = rospy.Publisher("pump", Bool, queue_size=1)
        # self.cmd_vel_pub = rospy.Publisher("/smoother_cmd_vel", Twist, queue_size=1)
        # Subscribers
        self.limit_switch_sub = rospy.Subscriber("limit_switch_state", Bool, self.limitCb)
        self.ultra_sub = rospy.Subscriber("/ultrasonic_distance", Float32, self.ultraCb)
        """ TODO: depth and rgb, only get data if true  """
        self.depth_sub = Subscriber("/oak/stereo/image_raw", Image)
        self.rgb_sub = Subscriber("/oak/rgb/image_raw", Image)
        # Synchronize depth and RGB topics
        ats = ApproximateTimeSynchronizer([self.depth_sub, self.rgb_sub], queue_size=10, slop=0.2)
        ats.registerCallback(self.visualSyncCb)

        self.bridge = CvBridge()
        self.save_img = False
        self.pose = [0,0,0]
        # Arm Services
        rospy.wait_for_service("/xarm_pose_plan")
        rospy.wait_for_service("/xarm_exec_plan")
        self.pose_plan_service = rospy.ServiceProxy("/xarm_pose_plan", pose_plan)
        self.exec_plan_service = rospy.ServiceProxy("/xarm_exec_plan", exec_plan)
        rospy.wait_for_service('/xarm/set_collision_sensitivity')
        rospy.wait_for_service('/xarm/set_state')
        self.set_collision_sensitivity = rospy.ServiceProxy('/xarm/set_collision_sensitivity', SetInt16)
        self.set_state = rospy.ServiceProxy('/xarm/set_state', SetInt16)
        
        # ER service
        rospy.wait_for_service('er_service') 
        self.serial_service = rospy.ServiceProxy('er_service', er)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Save data path
        base = "/home/culvertron/robotics/culvertron_ws/src/data_collection"
        self.paths = create_timestamped_folders(base)
        print("Depth Path: ", self.paths['depth'])
        print("RGB Path: ", self.paths['rgb'])
        print("ER Path: ", self.paths['er'])
        self.data_index = 0
        self.er_data = []

        # Example target pose
        self.target_pose = {
            "x": 0.0,
            "y": 0.85,
            "z": 0.13,
            "qx": 0.0,
            "qy": 0.0,
            "qz": -0.707,
            "qw": 0.707
        }

                # Example target pose
        self.target_pose2 = {
            "x": 0.0,
            "y": 0.8,
            "z": 0.46,
            "qx": 0.0,
            "qy": 0.0,
            "qz": -0.707,
            "qw": 0.707
        }
        # waypoints
        self.waypoints =[]
        # ultrasonic data
        self.ultrasonic_data = None
        self.limit_switch = True
        # Start timer for continuous publishing
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)  # 1 Hz publishing

        try:
            response_sen = self.set_collision_sensitivity(1) # Pass the sensitivity value
            rospy.loginfo("Collision sensitivity set to: %s", response_sen)
            response_state = self.set_state(0)  # Pass the state value
            rospy.loginfo("State set to: %s", response_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

        cmd_vel = Twist()
        self.getER = False
        self.false_count = 0  # Counter for consecutive false readings
        self.false_threshold = 3  # Number of consecutive False readings required
        self.arm_state_stop = False
        self.start = True
        # 1
        # self.deploy_er(self.target_pose)
        # rospy.sleep(1.0)

        # 2
        # cmd_vel.linear.x = 0.2
        # self.cmd_vel_pub.publish(cmd_vel)
        # rospy.sleep(4.0)
        # cmd_vel.linear.x = 0.0
        # self.cmd_vel_pub.publish(cmd_vel)
        # rospy.sleep(1.0)
        # self.deploy_er(self.target_pose)
        # rospy.sleep(1.0)

        # # 3
        # cmd_vel.linear.x = 0.2
        # self.cmd_vel_pub.publish(cmd_vel)
        # rospy.sleep(4.0)
        # cmd_vel.linear.x = 0.0
        # self.cmd_vel_pub.publish(cmd_vel)
        # rospy.sleep(1.0)
        # self.deploy_er(self.target_pose)
        # rospy.sleep(1.0)

        # # 4
        # cmd_vel.linear.x = 0.2
        # self.cmd_vel_pub.publish(cmd_vel)
        # rospy.sleep(4.0)
        # cmd_vel.linear.x = 0.0
        # self.cmd_vel_pub.publish(cmd_vel)
        # rospy.sleep(1.0)
        # self.deploy_er(self.target_pose)
        # rospy.sleep(1.0)
        # Call the service once during initialization
        # self.deploy_er(self.target_pose)
        while not rospy.is_shutdown():
            """ NOTE: change self.target_pose for run"""
            try:
                key = input("Press 'z' to call the service (or 'c' to save and quit): ").strip()            
                if key == 'z':
                    self.deploy_er(self.target_pose)
                elif key == "x":
                    self.deploy_er(self.target_pose2)
                elif key == 'c':
                    self.save_data_and_exit()
                    rospy.loginfo("Data saved. Exiting...")
                    rospy.signal_shutdown('User exit')
                else:
                    rospy.loginfo("Invalid key. Press 'z' to call the service.")                    
            except KeyboardInterrupt:
                self.save_data_and_exit()


    def timer_callback(self, event):
        """
        Timer callback to publish the pose continuously.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"

        pose_msg.pose.position.x = self.target_pose["x"]
        pose_msg.pose.position.y = self.target_pose["y"]
        pose_msg.pose.position.z = self.target_pose["z"]

        pose_msg.pose.orientation.x = self.target_pose.get("qx")
        pose_msg.pose.orientation.y = self.target_pose.get("qy")
        pose_msg.pose.orientation.z = self.target_pose.get("qz")
        pose_msg.pose.orientation.w = self.target_pose.get("qw")

        self.pose_pub.publish(pose_msg)

        er_waypoints_marker = Marker()
        er_waypoints_marker.header.frame_id = "world"
        er_waypoints_marker.header.stamp = rospy.Time.now()
        er_waypoints_marker.ns = "points"
        er_waypoints_marker.id = 0
        er_waypoints_marker.type = Marker.POINTS
        er_waypoints_marker.action = Marker.ADD        # Set scale and color
        er_waypoints_marker.scale.x = 0.002  # Point size
        er_waypoints_marker.scale.y = 0.002
        er_waypoints_marker.color.r = 1.0
        er_waypoints_marker.color.g = 0.0
        er_waypoints_marker.color.b = 0.0
        er_waypoints_marker.color.a = 1.0        # Add points to the marker

        for point in self.waypoints:
            p = Point()
            p.x, p.y, p.z = point
            er_waypoints_marker.points.append(p)        
        
        self.er_waypoints_pub.publish(er_waypoints_marker)

        
    def deploy_er(self, goal_pose):
        # Params:
        point_distance = 0.04  # Distance between consecutive points in meters
        start_offset = 0.19  # Starting offset in meters
        intermediate_offset = 0.35  # Ending offset in meters 
        end_offset = 0.4   
        xyz = np.array([goal_pose.get("x"), goal_pose.get("y"), goal_pose.get("z")])  # Input position in meters
        quaternion = (goal_pose.get("qx"), goal_pose.get("qy"), goal_pose.get("qz"),goal_pose.get("qw"))  # Identity quaternion
        q1 = np.array([0.0,0.0,-0.707, 0.707])
        q2 = np.array([0.0,0.0,0.707, 0.707])
        quaternion = closest_quaternion(q1,q2,quaternion)
        # Generate points        
        self.waypoints = generate_points(xyz, quaternion, point_distance, start_offset, intermediate_offset, end_offset)
        pull_back = self.waypoints.copy()
        quat = quaternion#[goal_pose.get("qx"), goal_pose.get("qy"), goal_pose.get("qz"),goal_pose.get("qw")]
        rpy = R.from_quat(quat).as_euler('xyz', degrees=False)
        # reverse the orientation where x axis of target and z axis of arm is perpendicular 
        rpy[1] = rpy[1] - 1.57
        rpy[2] = rpy[2] #+ 1.57
        """ TODO: make sure to flip the yaw 90  """

        goal_quaternion = R.from_euler('xyz', rpy).as_quat()
        self.setxArmState(0)
        self.perform_service_call(self.waypoints[-1], goal_quaternion)
        print("- RGB goal: ", self.waypoints[-1])

        """ TODO: save depth and rgb   """
        rospy.sleep(1)
        self.save_img = True
        rospy.sleep(1)
        print("getER1: ", self.getER)
        if not self.getER:
            self.perform_service_call(self.waypoints[-2], goal_quaternion)
            # spray water for 4 secs
            self.pump_pub.publish(Bool(data=True))
            print("Spraying water")
            rospy.sleep(2)
            print("Waiting")
            self.pump_pub.publish(Bool(data=False))
            rospy.sleep(15)
        print("Get close to wall")
        # get close to wall
        for point in reversed(self.waypoints[:-2]):
            # Defense mechanism to prevent ER collision
            if self.getER:
                break
            # when close to 8 cm, don't rely on ultrasonic
            if self.ultrasonic_data < 8.0:
                break
            # when not, keep going
            self.perform_service_call(point, goal_quaternion)
        if not self.getER:
            # Params:
            point_distance = 0.005  # Distance between consecutive points in meters
            start_offset = 0.13  # Starting offset in meters

            # get current end-defector position
            """ TODO: tf transform when doing the whole thing    """
            self.tf_buffer.can_transform("world", "link5", rospy.Time(0), rospy.Duration(2.0))
            trans = self.tf_buffer.lookup_transform("world", "link5", rospy.Time(0), rospy.Duration(0.3))
            t = trans.transform.translation
            # find dist
            dist = np.linalg.norm(xyz - np.array([t.x,t.y,t.z]))
            self.waypoints = generate_points(xyz, quaternion, point_distance, start_offset, dist, None)
            print("ER press")
            # ER press
            for point in reversed(self.waypoints):
                if not self.getER:
                    if self.ultrasonic_data < 7.0:
                        try:
                        # self.serial_service = rospy.ServiceProxy('er_service', er)
                            response = self.serial_service()
                        except rospy.ServiceException as e:
                            rospy.logerr(f"Service call failed: {e}")                
                            if response.er_reading != -1:
                                print(f"ER data: {response.er_reading}")
                                self.er_data.append((self.data_index, self.pose[0],goal_pose.get("y"),goal_pose.get("z")+0.5,response.er_reading))
                                self.save_data_and_exit()
                                # up the index
                                self.data_index+=1
                                break  
                        else:
                            print("bad readings")
                            self.perform_service_call(point, goal_quaternion)   
                    else:
                        print("too far")
                        self.perform_service_call(point, goal_quaternion)      
                else:
                    break
        print("getER: ", self.getER)
        if self.getER: 
            max_retries = 5  # Maximum number of retries
            attempts = 0     # Counter for attempts
            data = -1
            while data == -1 and attempts < max_retries:
                try:
                    self.serial_service = rospy.ServiceProxy('er_service', er)
                    response = self.serial_service()                    
                    print(f"ER data: {response.er_reading}")
                    if response.er_reading != -1:
                        data =  response.er_reading
                    else:
                        data = -1
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call failed: {e}")                
                attempts += 1
            # save once proper reading
            self.er_data.append((self.data_index, self.pose[0],goal_pose.get("y"),goal_pose.get("z")+0.5,data))
            self.save_data_and_exit()
            # up the index
            self.data_index+=1 
        # set state back to 0
        self.setxArmState(0)
        # pull back to avoid triggering estop 
        print("Pull Back")          
        # self.perform_service_call(self.waypoints[-1], goal_quaternion)
        print("Pull Back More")   
        # self.perform_service_call(pull_back[-2], goal_quaternion)
        self.perform_service_call(pull_back[-1], goal_quaternion)
        print("DONE, select next action")
        self.arm_state_stop = False
        self.getER = False
    
    def perform_service_call(self, goal_position, goal_quat):
        """
        Call the /xarm_pose_plan and /xarm_exec_plan services once.
        """
        # quat = [goal_pose.get("qx"), goal_pose.get("qy"), goal_pose.get("qz"),goal_pose.get("qw")]
        
        # rpy = R.from_quat(quat).as_euler('xyz', degrees=False)
        # reverse the orientation where x axis of target and z axis of arm is perpendicular 
        # rpy[1] = rpy[1] - 1.57
        # Plan the pose
        target_pose = Pose()
        target_pose.position.x = goal_position[0]
        target_pose.position.y = goal_position[1]
        target_pose.position.z = goal_position[2]

        # quaternion = R.from_euler('xyz', rpy).as_quat()

        target_pose.orientation.x = goal_quat[0]
        target_pose.orientation.y = goal_quat[1]
        target_pose.orientation.z = goal_quat[2]
        target_pose.orientation.w = goal_quat[3]

        try:
            response = self.pose_plan_service(target=target_pose)
            if response.success:
                rospy.loginfo("Pose planning succeeded!")
                # Execute the plan if planning succeeded
                exec_response = self.exec_plan_service(exec=True)
                if exec_response.success:
                    rospy.loginfo("Plan execution succeeded!")
                else:
                    rospy.logwarn("Plan execution failed.")
            else:
                rospy.logwarn("Pose planning failed.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def save_data_and_exit(self):
        # save ER data in csv file
        file_path = self.paths['er'] + '/ER.csv'  
        rospy.loginfo(f"Saving data to {file_path}...")

        with open(file_path, mode='w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "x", "y", "z", "er"])
            writer.writerows(self.er_data)

        # rospy.loginfo("Data saved. Exiting...")
        # rospy.signal_shutdown('User exit')

    def ultraCb(self, msg):
        self.ultrasonic_data = msg.data

    def limitCb(self, msg):
        if not self.start:
            return
        if not self.getER:
            limit_switch = msg.data
            if limit_switch == False:
                self.false_count += 1
            else:
                self.false_count = 0  # Reset if True is received
        # print(limit_switch)
        # if limit switch is pressed (false) and arm hasn't been stop
        if self.false_count >= self.false_threshold and self.getER == False:#self.arm_state_stop:
            print(limit_switch, self.getER)
            self.getER = True
            self.arm_state_stop = True
            self.false_count = 0
            print("set to STOP")
            self.setxArmState(4)

    def visualSyncCb(self, msg1, msg2):
        if self.save_img:
            try:
                # Convert ROS Image to OpenCV image
                depth_image = self.bridge.imgmsg_to_cv2(msg1, desired_encoding="passthrough")            # Normalize depth data to uint8 for visualization
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)            # Save depth image
                depth_filename = f"{self.paths['depth']}/{self.data_index}.png"
                cv2.imwrite(depth_filename, depth_normalized)
                rospy.loginfo(f"Saved depth image: {depth_filename}")

                # Convert ROS Image to OpenCV image
                rgb_image = self.bridge.imgmsg_to_cv2(msg2, desired_encoding="bgr8")            
                rgb_filename = f"{self.paths['rgb']}/{self.data_index}.png"
                cv2.imwrite(rgb_filename, rgb_image)
                rospy.loginfo(f"Saved RGB image: {rgb_filename}") 
                self.save_img = False           
            except Exception as e:
                rospy.logerr(f"Failed to save RGB image: {e}")
    
    def setxArmState(self, state):
        try:
            response_state = self.set_state(state)  # Pass the state value
            rospy.loginfo("State set to: %s", response_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

if __name__ == "__main__":
    try:
        XArmPosePublisherAndPlanner()
        rospy.spin()  # Keep the node running to handle the timer
    except rospy.ROSInterruptException:
        pass