#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import *
from std_msgs.msg import *
from sensor_msgs.msg import Image, PointCloud2
from tf2_geometry_msgs.tf2_geometry_msgs import *
from nav_msgs.msg import Odometry, Path
import tf.transformations as tft
from tf.transformations import quaternion_matrix
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy
# from ultralytics import YOLO
from utils import *
from belief import *
import cv2
from scipy.spatial import Delaunay, ConvexHull
from tf_conversions import transformations
import os
import csv
from datetime import datetime
from message_filters import *
# services
from nde_er.srv import er
from xarm_planner.srv import *
from xarm_msgs.srv import *
import random

def calculate_distance(current_pose, target_pose):
    """Calculate the distance to the target in the XY plane."""
    delta = target_pose[:2] - current_pose[:2]
    return np.linalg.norm(delta)

def calculate_back_heading_error(current_pose, target_pose):
    """Calculate heading error for backward movement."""
    delta = target_pose[:2] - current_pose[:2]
    target_angle = np.arctan2(delta[1], delta[0])
    backward_target_angle = normalize_angle(target_angle + np.pi)
    heading_error = normalize_angle(backward_target_angle - current_pose[3])
    return heading_error

def normalize_quaternions(arr):
    # Extract quaternion part: last 4 columns
    quaternions = arr[:, 3:7]
    # Compute norms (Euclidean norm for each row)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    # Avoid division by zero: replace zeros with 1 (neutral quaternion)
    norms[norms == 0] = 1
    # Normalize quaternions
    normalized_quaternions = quaternions / norms
    # Update the original array
    arr[:, 3:7] = normalized_quaternions
    return arr

def normalize_angle(angle):
        """Normalize angle to [-π, π]."""
        return math.atan2(math.sin(angle), math.cos(angle))
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
class Indoor:
    def __init__(self, unique_points, robot_state, center_line):
        """
        Initialize the MarkerPublisher.

        Parameters:
        topic_name (str): The topic name for the marker.
        frame_id (str): The frame ID for the marker.
        """
        # Initialize ROS node
        rospy.init_node('indoor_os_cams', anonymous=True)
        self.start = False
        # mode: [0: visual, 1: full]
        self.mode = 1
        # Move Publishers
        self.culvert_points_pub = rospy.Publisher("culvert_points", Marker, queue_size=5)
        self.culvert_pose_pub = rospy.Publisher("culvert_pose", PoseArray, queue_size=5)
        self.robot_points_pub = rospy.Publisher("robot_points", Marker, queue_size=5)
        self.target_pub = rospy.Publisher("target", Marker, queue_size=1)
        self.cmd_pub = rospy.Publisher("smoother_cmd_vel", Twist, queue_size=5)
        self.path_pub = rospy.Publisher("robot_path", Path, queue_size=1)
        # Arm Publishers
        self.pose_pub = rospy.Publisher("/target_pose", PoseStamped, queue_size=1)
        self.er_waypoints_pub = rospy.Publisher("/ER_waypoints", Marker, queue_size=10)
        self.pump_pub = rospy.Publisher("pump", Bool, queue_size=1)
        self.visual_pub = rospy.Publisher("visual", Image, queue_size=1, latch=True)
        self.er_pub = rospy.Publisher("er_reading", Float32, queue_size=1, latch=True)
        
        # print("1")
        # Subscribers
        self.limit_switch_sub = rospy.Subscriber("limit_switch_state", Bool, self.limitCb)
        # self.limit_switch_sub = rospy.Subscriber("top_limit_switch_state", Bool, self.limitCb)
        
        self.ultra_sub = rospy.Subscriber("/ultrasonic_distance", Float32, self.ultraCb)
        self.depth_sub = Subscriber("/oak/stereo/image_raw", Image)
        self.rgb_sub = Subscriber("/oak/rgb/image_raw", Image)
        # Synchronize depth and RGB topics
        ats = ApproximateTimeSynchronizer([self.depth_sub, self.rgb_sub], queue_size=10, slop=0.2)
        ats.registerCallback(self.visualSyncCb)
        # Other subscribers
        # self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odomCb)
        # print("2")
        # timer event
        self.timer = rospy.Timer(rospy.Duration(1.0/10.0), self.mainCb)        
        # cv bridge for converting ros image to cv image
        self.bridge = CvBridge()
        self.save_img = False
        # print("3")
        # Arm Services
        rospy.wait_for_service("/xarm_pose_plan")
        rospy.wait_for_service("/xarm_exec_plan")
        self.pose_plan_service = rospy.ServiceProxy("/xarm_pose_plan", pose_plan)
        self.exec_plan_service = rospy.ServiceProxy("/xarm_exec_plan", exec_plan)
        rospy.wait_for_service('/xarm/set_collision_sensitivity')
        rospy.wait_for_service('/xarm/set_state')
        # collision detector 
        self.set_collision_sensitivity = rospy.ServiceProxy('/xarm/set_collision_sensitivity', SetInt16)
        self.set_state = rospy.ServiceProxy('/xarm/set_state', SetInt16)
        # ER service
        rospy.wait_for_service('er_service') 
        self.serial_service = rospy.ServiceProxy('er_service', er)
        # print("4")
        # Save data path
        base = "/home/culvertron/robotics/culvertron_ws/src/data_collection"
        self.paths = create_timestamped_folders(base)
        print("Depth Path: ", self.paths['depth'])
        print("RGB Path: ", self.paths['rgb'])
        print("ER Path: ", self.paths['er'])
        self.data_index = 0
        """"TODO: when no press append -1   """
        self.er_data = []
        # waypoints "world" frame
        self.waypoints =[]
        # ultrasonic data
        self.ultrasonic_data = None
        self.getER = False
        self.false_count = 0  # Counter for consecutive false readings
        self.false_threshold = 3  # Number of consecutive False readings required
        try:
            response_state = self.set_state(0)  # Pass the state value
            rospy.loginfo("State set to: %s", response_state)
            response_sen = self.set_collision_sensitivity(1) # Pass the sensitivity value
            rospy.loginfo("Collision sensitivity set to: %s", response_sen)
            response_state = self.set_state(0)  # Pass the state value
            rospy.loginfo("State set to: %s", response_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

        # get only position (orientation is for future work) and add visited
        false_column = np.full((unique_points.shape[0], 1), False)
        cls_column = -1 * np.ones((unique_points.shape[0], 1))
        self.culvert_points = np.hstack((unique_points, cls_column, false_column))        
        # crack 0 spall 1 both 2
        self.culvert_points[2,7] = 2
        # self.culvert_points[1,7] = 2
        self.culvert_points[4,7] = 1
        # self.culvert_points[10,7] = 1
        self.culvert_points[15,7] = 0
        # self.culvert_points[26,7] = 2
        self.culvert_points[14,7] = 2
        # self.culvert_points[20,7] = 2
        self.robot_state = robot_state
        print(self.culvert_points.shape, self.robot_state.shape)
        false_column = np.full((center_line.shape[0], 1), False)
        self.center_line = np.hstack((center_line, false_column)) 
        
        self.global_frame = "map" #"odom" #  
        self.arm_frame = "world" # "base_link" # 
        self.robot_frame = "robot_center"
        """ go to goal input    """
        self.ang_tol = 0.05
        self.lin_tol = 0.03
        self.ang = 0.3
        self.lin = 0.11
        self.print_goal_action = False
        # 0 = move, 1 = turn, 2 = back, 3 = declare, 4 = exit
        self.ACTION = -1
        self.action_name = [ "move", "declare", "exit"]
        self.action_goal = None
        # self.publish_markers()
        self.cmd_vel = Twist()
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # path
        self.path = Path()
        self.path.header.frame_id = self.global_frame
        self.new_action = True
        self.print_goal_action = False
        self.move_goal_index = None
        self.arm_goal_index = None
        self.arm_deploying = False
        
        self.robot = None
        self.arm_pose = None
        self.previous_move_goal_index = -2 # -2 so it doesn't get affected by initial self.ACTION = -1
        self.publish_markers()
        self.start = True
        print(self.start)

    def mainCb(self,event=None):
        if not self.start:
            return
        # self.publish_markers()
        # print("publish")
        # return
        try:
            tf_ = self.tf_buffer.can_transform(self.global_frame, self.robot_frame, rospy.Time(0), rospy.Duration(0.5))
            transform = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, rospy.Time(0), rospy.Duration(0.3))
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            # Extract rotation (quaternion)
            q = transform.transform.rotation
            quaternion = [q.x, q.y, q.z, q.w]

            # ✅ Convert quaternion to Euler angles (roll, pitch, yaw)
            (roll, pitch, yaw) = transformations.euler_from_quaternion(quaternion)
            self.robot = np.array([x, y, z, yaw, 1.0])

            tf_arm = self.tf_buffer.can_transform(self.global_frame, self.arm_frame, rospy.Time(0), rospy.Duration(0.5))
            if tf_arm:
                transform_arm = self.tf_buffer.lookup_transform(self.global_frame, self.arm_frame, rospy.Time(0), rospy.Duration(0.3))
                x_arm = transform_arm.transform.translation.x
                y_arm = transform_arm.transform.translation.y
                z_arm = transform_arm.transform.translation.z
            else:
                rospy.logerr_once("Can't transform world to map")
                return

            # Extract rotation (quaternion)
            q_arm = transform_arm.transform.rotation
            quaternion_arm = [q_arm.x, q_arm.y, q_arm.z, q_arm.w]

            # ✅ Convert quaternion to Euler angles (roll, pitch, yaw)
            (roll, pitch, yaw_arm) = transformations.euler_from_quaternion(quaternion_arm)
            self.arm_pose = np.array([x_arm, y_arm, z_arm, yaw_arm, 1.0])
            
            # update path
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0 
            self.path.poses.append(pose)
            self.path_pub.publish(self.path)
            # print(self.robot)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Transform not available yet...")

        self.publish_markers()
        if self.arm_deploying:
            return         

        if self.ACTION == -1:
            # assign ACTION and its goal
            self.planner()
            print("Action: ", self.ACTION)
            self.print_goal_action = True
        # move
        if self.ACTION == 0:
            # if already there go to next action
            if self.previous_move_goal_index == self.move_goal_index:
                print("0: Already there, go to ACTION 3")
                self.ACTION = 3
            else:
                if self.goToGoal(self.robot_state[self.move_goal_index]):
                    self.ACTION = 1
                    # once reach then set that previous is this in case next goal index is same
                    # so n
                    self.previous_move_goal_index = self.move_goal_index
                    # reset
                    # self.move_goal_index = None
                    self.print_goal_action = True
                else:
                    # if first time, print action
                    if self.print_goal_action:
                        print(f'\r 0: Going to {self.robot_state[self.move_goal_index]}')
                        self.print_goal_action = False
        # turning for arm
        if self.ACTION == 1:
            if self.turn_setup_for_arm(self.culvert_points[self.arm_goal_index]):
                self.ACTION = 2
                # reset                
                self.print_goal_action = True
            else:
                # if first time, print action
                if self.print_goal_action:
                    print(f'1: Turning for arm')
                    self.print_goal_action = False
        # TODO: move_backwards for arm
        if self.ACTION == 2:
            # no need to transform since self.arm_pose is in map frame
            if self.move_backwards_for_arm(self.robot_state[self.move_goal_index]):
                self.ACTION = 3
                self.move_goal_index = None
                self.print_goal_action = True
            else:
                if self.print_goal_action:
                    print(f'2: Backwards for arm')
                    self.print_goal_action = False
        # arm deployment
        if self.ACTION == 3:
            # self.culvert_points[self.arm_goal_index, 8] = True
            # print("set ", self.arm_goal_index, " to True: ",  self.culvert_points[self.arm_goal_index])
            # self.arm_goal_index = None
            # self.arm_deploying = False
            # self.ACTION = -1   
            # rospy.sleep(15.0 + random.uniform(-5, 5)) 
            # return            
            print("3: Deploy arm")
            # set arm deploy to true, then do the wait, 
            self.arm_deploying = True
            is_crack = (self.culvert_points[self.arm_goal_index, 7] == 0) #and (self.culvert_points[self.arm_goal_index, 7] == 2)
            # TODO: transform it from map to world
            tf_pose = self.transform_pose(self.culvert_points[self.arm_goal_index], self.global_frame, self.arm_frame)
            self.deploy_er(tf_pose, is_crack)
            # ONCE done            
            # mark as visited
            self.culvert_points[self.arm_goal_index, 8] = True
            print("set ", self.arm_goal_index, " to True: ",  self.culvert_points[self.arm_goal_index])
            self.arm_goal_index = None
            self.arm_deploying = False
            self.ACTION = -1   

    def planner(self):
        """ 
        set self.move_goal_index and self.arm_goal_index to go to and inspect
        called only when ACTION is -1    
        """
        # 8th column !=-1 => defect, 9th column == False => not visited
        mask = (self.culvert_points[:, -1] == False) 
        filtered_culvert_points = self.culvert_points[mask]
        if filtered_culvert_points.size == 0:
            print("Action 4: Done")
            self.ACTION = 4
            rospy.signal_shutdown("Task Done.")
            return None
        else:
            # Find the closest culvert point to self.robot
            robot_position = self.robot[:3]  # [x, y, z]
            culvert_positions = filtered_culvert_points[:, :3]  # First three columns are positions

            distances_to_culvert_points = np.linalg.norm(culvert_positions - robot_position, axis=1)
            closest_culvert_index_filtered = np.argmin(distances_to_culvert_points)
            closest_culvert_point = filtered_culvert_points[closest_culvert_index_filtered]

            # Map back to the original index in self.culvert_points
            original_indexes = np.where(mask)[0]
            closest_culvert_index = original_indexes[closest_culvert_index_filtered]
            self.arm_goal_index = closest_culvert_index

            # Step 3: Find the closest robot state to the closest culvert point
            distances_to_robot_states = np.linalg.norm(self.robot_state - closest_culvert_point[:3], axis=1)
            closest_robot_state_index = np.argmin(distances_to_robot_states)

            # Index of the closest robot state
            closest_robot_state_index = np.argmin(distances_to_robot_states)
            self.move_goal_index = closest_robot_state_index
            self.ACTION = 0

    def goToGoal(self, goal):
        """ goal: np.array([x,y,z])"""
        self.visualizeTarget(goal[0], goal[1])
        lin_vel = 0
        ang_vel = 0  
        angleToGoal, distance = compute_angle_distance(goal, self.robot)
        """ if close enough """
        if distance < self.lin_tol:
            self.publish_velocity(lin_vel, ang_vel)
            print("Goal Reached")
            return True
        
        yaw_error = normalize_angle(self.robot[3] - angleToGoal)
        if abs(yaw_error) < self.ang_tol:
            lin_vel = self.lin
            ang_vel = 0
        else:
            lin_vel = 0
            ang_vel = -self.ang if yaw_error > 0 else self.ang
        self.publish_velocity(lin_vel, ang_vel)
        return False

    def turn_setup_for_arm(self, goal):
        """" 
        Turn robot perpendicular to goal
        Args:
            pose quaternion (tuple): Quaternion [x, y, z, w]]
        """
        # _, _, yaw = tft.euler_from_quaternion(goal_orientation)
        # TODO: find a better way shortest only if last point otherwise use 2 points 
        closest_angle = np.pi if goal[1] > 0 else 0#np.pi if self.data_index > 11 else 0
        # angle1 = yaw + np.pi / 2 + 0.001
        # angle2 = yaw - np.pi / 2 - 0.001 

        # # Normalize angles to the range [-π, π]
        # angle1 = np.arctan2(np.sin(angle1), np.cos(angle1))
        # angle2 = np.arctan2(np.sin(angle2), np.cos(angle2))

        # # Compute the angular differences
        # diff1 = np.abs(np.arctan2(np.sin(angle1 - self.robot[3]), np.cos(angle1 - self.robot[3])))
        # diff2 = np.abs(np.arctan2(np.sin(angle2 - self.robot[3]), np.cos(angle2 - self.robot[3])))

        # # Choose the angle with the smallest angular difference
        # closest_angle = angle1 if diff1 < diff2 else angle2
        # print("yaw: ", np.degrees(yaw),  "(", np.degrees(angle1), np.degrees(angle2), ") = ", np.degrees(closest_angle))
        # print("angle diff: ", abs(shortest_angular_difference(self.robot[3], closest_angle)))
        # print("Turn Angle Goal", closest_angle)
        # if abs(shortest_angular_difference(self.robot[3], closest_angle)) < self.ang_tol:
        #     self.publish_velocity(0, 0)
        #     return True
        # # TODO: TURN TO angle closest to angle to goal (later)
        # ang_vel = turning(self.robot[3], closest_angle, self.ang)
        # self.publish_velocity(0, ang_vel)
        yaw_error = normalize_angle(self.robot[3] - closest_angle)
        if abs(yaw_error) < self.ang_tol:
            self.publish_velocity(0, 0)
            return True
        ang_vel = -self.ang if yaw_error > 0 else self.ang
        self.publish_velocity(0, ang_vel)
        return False
    
    def move_backwards_for_arm(self, goal):
        distance = calculate_distance(self.arm_pose, goal)
        distance = abs(self.arm_pose[0] - goal[0])
        print("distance: ", distance)
        # if distance < 0.09:
        if distance < 0.08:
            self.publish_velocity(0, 0)
            return True
        self.publish_velocity(-self.lin, 0)        
        # heading_error = calculate_back_heading_error(self.current_pose, self.target_pose)
        # Control angular velocity to correct heading
        # angular_z = np.clip(self.angular_kp * heading_error, -self.angular_speed_limit, self.angular_speed_limit)
        return False
    
    # reverse the goal_pose from dictionary to array
    def deploy_er(self, goal_pose, is_crack=True):
        """ This is all in the "world" frame """
        # Params:
        print("Arm Goal: ", goal_pose)
        point_distance = 0.04  # Distance between consecutive points in meters
        start_offset = 0.19  # Starting offset in meters
        intermediate_offset = 0.4  # Ending offset in meters 
        end_offset = 0.45   
        xyz = goal_pose[:3]  # Input position in meters
        # TODO: find better than bandage solution always 0 in world frame
        xyz[0] = 0
        xyz[1] = -0.88
        if xyz[2] < 0.2:
            xyz[2] = 0.13
        else:
            xyz[2] = 0.4
        # quaternion = goal_pose[3:7]  # Identity quaternion
        # quaternion = np.array([0.0,0.0,0.707, 0.707])
        q1 = np.array([0.0,0.0,-0.707, 0.707])
        q2 = np.array([0.0,0.0,0.707, 0.707])
        quaternion = closest_quaternion(q1,q2,goal_pose[3:7])
        # Generate points        
        self.waypoints = generate_points(xyz, quaternion, point_distance, start_offset, intermediate_offset, end_offset)
        pull_back = self.waypoints.copy()
        quat = quaternion#[0.0,0.0,0.707, 0.707]#[goal_pose[3], goal_pose[4], goal_pose[5],goal_pose[6]]
        rpy = R.from_quat(quat).as_euler('xyz', degrees=False)
        # reverse the orientation where x axis of target and z axis of arm is perpendicular 
        rpy[1] = rpy[1] - 1.57
        rpy[2] = rpy[2] #+ 1.57
        """ make sure to flip the yaw 90  """

        goal_quaternion = R.from_euler('xyz', rpy).as_quat()
        self.setxArmState(0)
        self.perform_service_call(self.waypoints[-1], goal_quaternion)
        rgbd_point = self.waypoints[-1]
        """ save depth and rgb   """
        rospy.sleep(1)
        self.save_img = True
        rospy.sleep(1)
        if self.mode == 0:
            self.data_index+=1
            return
        print("getER1: ", self.getER)

        # don't do it if spall or crack if less than 0.33
        # if is_crack: #or xyz[2] < 0.33:
        #     # save a dummy er value 
        #     self.er_data.append((self.data_index, self.culvert_points[self.arm_goal_index, 0],self.culvert_points[self.arm_goal_index,1],self.culvert_points[self.arm_goal_index,2], -1))
        #     self.save_data_and_exit()
        #     # up the index
        #     self.data_index+=1
        #     return
        print("ER ARM time!")
   
        # constrain since arm is short so we can only do ER in some
        self.waypoints = generate_points(xyz, quaternion, point_distance, start_offset, intermediate_offset, end_offset)
        pull_back = self.waypoints.copy()
        # Defense mechanism to prevent ER collision
        if not self.getER:
            self.perform_service_call(self.waypoints[-2], goal_quaternion)
            # spray water for 2 secs
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
        # Defense mechanism to prevent ER collision
        if not self.getER:
            # Params:
            point_distance = 0.005  # Distance between consecutive points in meters
            start_offset = 0.13  # Starting offset in meters
            # intermediate_offset = 0.5 
            # end_offset = 0.7

            # get current end-defector position
            """ TODO: tf transform when doing the whole thing    """
            self.tf_buffer.can_transform("world", "link5", rospy.Time(0), rospy.Duration(2.0))
            trans = self.tf_buffer.lookup_transform("world", "link5", rospy.Time(0), rospy.Duration(0.3))
            t = trans.transform.translation
            # find dist
            dist = np.linalg.norm(xyz - np.array([t.x,t.y,t.z]))
            self.waypoints = generate_points(xyz, quaternion, point_distance, start_offset, dist, None)
            pull_back = self.waypoints.copy()
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
                            """ transform world to"""
                            self.er_data.append((self.data_index, self.culvert_points[self.arm_goal_index, 0],self.culvert_points[self.arm_goal_index,1],self.culvert_points[self.arm_goal_index,2],response.er_reading))
                            self.er_pub.publish(Float32(data=response.er_reading))
                            self.save_data_and_exit()
                            break  
                        else:
                            print("bad readings")
                            if not self.perform_service_call(point, goal_quaternion):
                                rospy.logerr(f"Arm Service failed, start pulling back")   
                                break   
                    else:
                        print("too far")
                        if not self.perform_service_call(point, goal_quaternion):
                            rospy.logerr(f"Arm Service failed, start pulling back")   
                            break 
                else:
                    break
        print("getER: ", self.getER)
        data = -1
        if self.getER: 
            max_retries = 3
            attempts = 0
            
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
            self.er_data.append((self.data_index, self.culvert_points[self.arm_goal_index, 0],self.culvert_points[self.arm_goal_index,1],self.culvert_points[self.arm_goal_index,2],data))
            self.er_pub.publish(Float32(data=data))
            self.save_data_and_exit()

        # up the index
        self.data_index+=1    
        # set state back to 0
        self.setxArmState(0)
        if self.getER:
            self.setxArmState(0)
        # pull back to avoid triggering estop 
        print("Pull Back")          
        self.perform_service_call(pull_back[-3], goal_quaternion)
        print("Pull Back more")
        self.perform_service_call(pull_back[-2], goal_quaternion)
        print("Pull Back RGBD")
        self.perform_service_call(rgbd_point, goal_quaternion)
        self.getER = False
        print("Back to origin")
        # self.perform_service_call(np.array([0.000, -0.180, 0.4]), np.array([0.498, -0.498, 0.502, 0.502]))
        print("DONE, select next action")

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
        self.setxArmState(0)
        try:
            response = self.pose_plan_service(target=target_pose)
            if response.success:
                rospy.loginfo("Pose planning succeeded!")
                # Execute the plan if planning succeeded
                exec_response = self.exec_plan_service(exec=True)
                if exec_response.success:
                    rospy.loginfo("Plan execution succeeded!")
                    return True
                else:
                    rospy.logwarn("Plan execution failed.")
                    # self.setxArmState(0)
                    return False
            else:
                rospy.logwarn("Pose planning failed.")
                # self.setxArmState(0)
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            # self.setxArmState(0)
            return False

    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        """Publishes velocity commands to cmd_vel."""
        self.cmd_vel.linear.x = linear_x
        self.cmd_vel.angular.z = angular_z
        self.cmd_pub.publish(self.cmd_vel)

    def transform_pose(self, pose, child_frame, parent_frame):
        """ transform point from parent to child  """
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = parent_frame
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose.position.x = pose[0]
        pose_stamped.pose.position.y = pose[1]
        pose_stamped.pose.position.z = pose[2]
        pose_stamped.pose.orientation.x = pose[3]
        pose_stamped.pose.orientation.y = pose[4]
        pose_stamped.pose.orientation.z = pose[5]
        pose_stamped.pose.orientation.w = pose[6]      

        try:   
            self.tf_buffer.can_transform(parent_frame, child_frame, rospy.Time(0), rospy.Duration(2.0))
            trans = self.tf_buffer.lookup_transform(parent_frame, child_frame,
                                rospy.get_rostime() - rospy.Duration(0.2),
                                rospy.Duration(0.1))
            transformed_pose = do_transform_pose(pose_stamped, trans)
            transformed_array = np.array([
            transformed_pose.pose.position.x,
            transformed_pose.pose.position.y,
            transformed_pose.pose.position.z,
            transformed_pose.pose.orientation.x,
            transformed_pose.pose.orientation.y,
            transformed_pose.pose.orientation.z,
            transformed_pose.pose.orientation.w,
            ])
            return transformed_array
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("TF Transform not available.")
            return None

    def transform_point(self, x, y, z, child_frame, parent_frame):
        p = Point()
        p.x = x 
        p.y = y
        p.z = z                
        self.tf_buffer.can_transform(parent_frame, child_frame, rospy.Time(0), rospy.Duration(2.0))
        trans = self.tf_buffer.lookup_transform(parent_frame, child_frame,
                            rospy.get_rostime() - rospy.Duration(0.2),
                            rospy.Duration(0.1))
        # for bag
        # trans = tf_buffer.lookup_transform("odom", "zed_left_camera_frame",
        #                     header.stamp,
        #                     rospy.Duration(0.1))

        point_stamp = PointStamped()
        point_stamp.point = p
        point_stamp.header = parent_frame
        p_tf = do_transform_point(point_stamp, trans)

        return p_tf.point.x, p_tf.point.y, p_tf.point.z

    def ultraCb(self, msg):
        if not self.start:
            return
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
                            
                ros_img = self.bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
                self.visual_pub.publish(ros_img)
                self.save_img = False           
            except Exception as e:
                rospy.logerr(f"Failed to save RGB image: {e}")  

    def publish_culvert_points_marker(self):
        if not self.start:
            return
        """
        Create a Marker message with the given points.

        Parameters:
        unique_points (list of tuple): List of points to include in the marker.
        
        Returns:
        Marker: The constructed Marker message.
        """        
        marker = Marker()
        marker.header.frame_id = "map"#self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "culvert_points"
        marker.id = 10
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1  # Adjust size of points as needed
        marker.color.r = 1.0
        marker.color.a = 1.0
        # pose_array = PoseArray()
        # Convert points to Marker format
        for point in self.culvert_points:
            p = Point()
            p.x, p.y, p.z = point[:3]
            marker.points.append(p)
            color = ColorRGBA()
            if point[-1]:
                color.r, color.g, color.b, color.a = 1.0, 1.0, 1.0, 1.0
            else:
                if point[-2] == -1:                    
                    color.r, color.g, color.b, color.a = 0.5, 0.5, 0.5, 1.0
                elif point[-2] == 0:                    
                    color.r, color.g, color.b, color.a = 1.0, 0.0, 0.0, 1.0
                elif point[-2] == 1:                    
                    color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 1.0
                elif point[-2] == 2:                    
                    color.r, color.g, color.b, color.a = 1.0, 1.0, 0.0, 1.0
                else:                    
                    color.r, color.g, color.b, color.a = 0.0, 0.0, 0.0, 1.0
            marker.colors.append(color)        
            
            # pose_array.header.frame_id = "map"#"odom"#
            # pose_array.header.stamp = rospy.Time.now()

            # pose = Pose()
            # pose.position.x = point[0]
            # pose.position.y = point[1]
            # pose.position.z = point[2]
            # pose.orientation.x = point[3]
            # pose.orientation.y = point[4]
            # pose.orientation.z = point[5]
            # pose.orientation.w = point[6]
            # pose_array.poses.append(pose)
        self.culvert_points_pub.publish(marker)
        
        # self.culvert_pose_pub.publish(pose_array)
    
    def publish_robot_points_marker(self):
        """
        Create a Marker message with the given points.

        Parameters:
        unique_points (list of tuple): List of points to include in the marker.
        
        Returns:
        Marker: The constructed Marker message.
        """
        if not self.start:
            return
        marker = Marker()
        marker.header.frame_id = "map"#self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_points"
        marker.id = 100
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1  # Adjust size of points as needed
        marker.color.b = 1.0
        marker.color.a = 1.0
        # Convert points to Marker format
        for point in self.robot_state:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)
            color = ColorRGBA()
            color.r, color.g, color.b, color.a = 0.1, 0.1, 0.1, 1.0
            marker.colors.append(color)  
                
        for point in self.center_line:
            p = Point()
            p.x, p.y, p.z = point[:3]
            marker.points.append(p)
            color = ColorRGBA()
            if point[-1]:
                color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 1.0
            else:
                color.r, color.g, color.b, color.a = 0.1, 0.1, 0.1, 1.0
            marker.colors.append(color)

        self.robot_points_pub.publish(marker)

    def visualizeTarget(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.global_frame  # Adjust according to your TF frames
        # marker.header.stamp = rospy.Time.now()
        marker.ns = "arrow_marker"
        marker.id = 10000
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set the pose of the marker
        # The arrow points up from the point (x, y) to (x, y, 1)
        marker.points.append(Point(x, y, 0))  # Start point
        marker.points.append(Point(x, y, 0.3))  # End point - pointing straight up
        
        # Set the scale of the arrow
        marker.scale.x = 0.02  # Shaft diameter
        marker.scale.y = 0.05  # Head diameter
        marker.scale.z = 0.05  # Head length
        
        # Set the color of the marker
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Make sure to set the alpha to something non-zero!
        
        # Publish the Marker
        self.target_pub.publish(marker)

    def publish_er_waypoints(self):
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
    
    def save_data_and_exit(self):
        # save ER data in csv file
        file_path = self.paths['er'] + '/ER.csv'  
        rospy.loginfo(f"Saving data to {file_path}...")

        with open(file_path, mode='w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "x", "y", "z", "er"])
            writer.writerows(self.er_data)
    
    def publish_markers(self):
        self.publish_culvert_points_marker()
        self.publish_robot_points_marker()
        self.publish_er_waypoints()
    
    def setxArmState(self, state):
        try:
            response_state = self.set_state(state)  # Pass the state value
            rospy.loginfo("State set to: %s", response_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

def get_transformation_matrix(transform):
    # Extract translation and rotation (quaternion) from the TransformStamped message
    t = transform.transform.translation
    q = transform.transform.rotation
    # Create a 4x4 translation matrix
    translation_matrix = np.array([
        [1, 0, 0, t.x],
        [0, 1, 0, t.y],
        [0, 0, 1, t.z],
        [0, 0, 0, 1]
    ])
    # Create a 4x4 rotation matrix from the quaternion
    rotation_matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
    # Combine the translation and rotation into one 4x4 transformation matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix

def transform_points_batch(points, transformation_matrix):
    # Add the 4th homogeneous coordinate (1) to all points
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the transformation matrix to all points
    transformed_points_homogeneous = transformation_matrix.dot(points_homogeneous.T).T

    # Convert back to 3D by removing the homogeneous coordinate
    transformed_points = transformed_points_homogeneous[:, :3]
    return transformed_points

if __name__ == '__main__':
    try:

        # Retrieve parameters from the ROS parameter server or use defaults
        # w = rospy.get_param('~w', 3.3)
        # l = rospy.get_param('~l', 2.4)
        # h = rospy.get_param('~h', 1.0)
        # scale = rospy.get_param('~scale', 4)
        # l_scale = rospy.get_param('~scale', 3)
        # offset_x = rospy.get_param('~offset_x', 1.49-1.3)
        # offset_y = rospy.get_param('~offset_y', -1.93 + 0.013+1.05)
        # offset_z = rospy.get_param('~offset_z', -0.05+0.1)

        # rospy.loginfo(offset_x, offset_y, offset_z)

        # h_fov = rospy.get_param('~h_fov', 61)
        # v_fov = rospy.get_param('~v_fov', 49)
        # near = rospy.get_param('~near', 0.2)
        # far = rospy.get_param('~far', 4.0)

        # robot_l = 1.0
        # robot_offset_x = rospy.get_param('~robot_offset_x', offset_x-0.92)
        # robot_offset_y = rospy.get_param('~robot_offset_y', offset_y+0.52+0.03)
        # robot_offset_z = rospy.get_param('~robot_offset_y', offset_z)
        # robot_x_scale = rospy.get_param('~robot_x_scale', 2.7)
        # robot_y_scale = rospy.get_param('~robot_y_scale', 1.7)
        
        w = rospy.get_param('~w', 3.3)
        l = rospy.get_param('~l', 2.4)
        h = rospy.get_param('~h', 1.0)
        scale = rospy.get_param('~scale', 4.5)
        l_scale = rospy.get_param('~scale', 3)
        offset_x = rospy.get_param('~offset_x', 1.49-1.3-0.12)
        offset_y = rospy.get_param('~offset_y', -1.93 + 0.013+0.615)
        offset_z = rospy.get_param('~offset_z', -0.05+0.1-0.05)

        rospy.loginfo(offset_x, offset_y, offset_z)

        robot_l = 1.0
        robot_offset_x = rospy.get_param('~robot_offset_x', offset_x+0.04)
        robot_offset_y = rospy.get_param('~robot_offset_y', offset_y+0.52+0.03+0.05-0.04)
        robot_offset_z = rospy.get_param('~robot_offset_y', offset_z)
        robot_x_scale = rospy.get_param('~robot_x_scale', 3.1)
        robot_y_scale = rospy.get_param('~robot_y_scale', 1.7+0.7+0.02)

        # generate culvert points & rgbd frostum
        unique_points = generate_uniform_grid(w, l, h, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z, scale=scale, l_scale=l_scale) # 
        robot_state, center_line = generate_robot_state(w*2/3, l, offset_x=robot_offset_x, offset_y=robot_offset_y, offset_z=robot_offset_z, scale_x=robot_x_scale, scale_y=robot_y_scale)
        print(unique_points.shape, robot_state.shape)
        unique_points = unique_points[unique_points[:, 2] >= 0.33]
        # print(unique_points.shape)
        # sim
        sim = Indoor(normalize_quaternions(unique_points), robot_state, center_line)
        sim.publish_markers()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
