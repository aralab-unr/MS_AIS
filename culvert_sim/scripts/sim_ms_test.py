#!/usr/bin/env python3
import rospy
# python lib
import numpy as np
import cv2
from scipy.spatial import cKDTree, KDTree
from scipy.spatial.transform import Rotation as R
from tf_conversions import transformations
# ROS lib
import tf2_ros
from cv_bridge import CvBridge
from std_msgs.msg import ColorRGBA, Bool
from visualization_msgs.msg import *
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points, create_cloud
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry, Path
import message_filters
# deep learning lib
from ultralytics import YOLO
# custom lib
from utils import *
from belief import *
import time


def merge_point_clouds(observations, threshold=0.01, min_matches=5):
    merged_observations = []
    
    while observations:
        pc1, class1, score1 = observations.pop(0)  # Take the first observation
        merged = False
        
        for i, (pc2, class2, score2) in enumerate(merged_observations):
            if class1 != class2:
                continue  # Only merge if classes are the same
            
            # Use KD-Tree for fast nearest neighbor search
            tree = cKDTree(pc2[:, :3])  # Use only XYZ for distance checking
            distances, _ = tree.query(pc1[:, :3], k=min_matches, distance_upper_bound=threshold)
            
            # Count how many points are within the threshold
            valid_matches = np.sum(distances < threshold)
            
            if valid_matches >= min_matches:
                # Merge point clouds
                merged_pc = np.vstack((pc2, pc1))
                avg_score = (score1 + score2) / 2  # Average scores

                merged_observations[i] = [merged_pc, class1, avg_score]
                merged = True
                break  # Stop checking after merging
        
        if not merged:
            merged_observations.append([pc1, class1, score1])  # Keep as separate if not merged
    
    return merged_observations

def stack_pc(pc_list):
    merged_pc = []

    for item in pc_list:  # Loop through each element in the list
        pc, classes, scores = item  # Extract pc, classes, and scores

        # Convert classes and scores to N x 1 arrays
        classes_column = np.full((pc.shape[0], 1), classes)  # Expand to match number of rows
        scores_column = np.full((pc.shape[0], 1), scores)  # Expand to match number of rows

        # Concatenate class and score columns to pc
        pc_with_class_score = np.hstack((pc, classes_column, scores_column))

        # Append to merged list
        merged_pc.append(pc_with_class_score)
    
    if not merged_pc:  # Check if merged_pc is empty
        # rospy.logwarn("Warning: No data to stack, returning empty array.")
        return np.empty((0, 6))

    final_pc = np.vstack(merged_pc)
    return final_pc

def obs_avg(culvert_points, merge_observations, radius=0.2):
    """
    For each point in culvert_points (N x 7):
      - Find neighbors in merge_observations (M x 6) within 'radius' in 3D (x,y,z).
      - Average their scores.
      - Determine new class (0, 1, or 2 if both 0 & 1 exist).
      - Count the number of neighboring points.
    Returns a (filtered N x 6) array: [x, y, z, final_class, avg_score, num_points], removing points with no neighbors.
    """

    # Build a KDTree on the observation coordinates
    obs_xyz = merge_observations[:, :3]   # shape (M, 3)
    obs_class = merge_observations[:, 4]  # shape (M,)
    obs_score = merge_observations[:, 5]  # shape (M,)

    tree = KDTree(obs_xyz)

    # Query all culvert points in one call -> list of neighbor indices for each culvert point
    neighbors_list = tree.query_ball_point(culvert_points[:, :3], r=radius)

    valid_results = []

    for i, neighbor_indices in enumerate(neighbors_list):
        num_neighbors = len(neighbor_indices)
        if num_neighbors == 0:
            continue  # Skip this culvert point if no neighbors

        # Gather neighbor class and scores
        n_classes = obs_class[neighbor_indices]
        n_scores = obs_score[neighbor_indices]

        # Determine final class
        unique_classes = set(n_classes)
        if unique_classes == {0}:
            final_cls = 0
        elif unique_classes == {1}:
            final_cls = 1
        else:
            final_cls = 2

        # Compute average score
        avg_score = np.mean(n_scores)

        # Store valid result with neighbor count
        valid_results.append([*culvert_points[i, :3], final_cls, avg_score, num_neighbors])

    # Convert to NumPy array
    if valid_results:
        return np.array(valid_results, dtype=float)
    else:
        return np.empty((0, 6))  # Return an empty array if no valid points

    # # Build a KDTree on the observation coordinates
    # obs_xyz = merge_observations[:, :3]   # shape (M, 3)
    # obs_class = merge_observations[:, 4]  # shape (M,)
    # obs_score = merge_observations[:, 5]  # shape (M,)

    # tree = KDTree(obs_xyz)

    # # Query all culvert points in one call -> list of neighbor indices for each culvert point
    # neighbors_list = tree.query_ball_point(culvert_points[:, :3], r=radius)

    # valid_results = []

    # for i, neighbor_indices in enumerate(neighbors_list):
    #     if len(neighbor_indices) == 0:
    #         continue  # Skip this culvert point if no neighbors

    #     # Gather neighbor class and scores
    #     n_classes = obs_class[neighbor_indices]
    #     n_scores = obs_score[neighbor_indices]

    #     # Determine final class
    #     unique_classes = set(n_classes)
    #     if unique_classes == {0}:
    #         final_cls = 0
    #     elif unique_classes == {1}:
    #         final_cls = 1
    #     else:
    #         final_cls = 2

    #     # Compute average score
    #     avg_score = np.mean(n_scores)

    #     # Store valid result
    #     valid_results.append([*culvert_points[i, :3], final_cls, avg_score])

    # # Convert to NumPy array
    # if valid_results:
    #     return np.array(valid_results, dtype=float)
    # else:
    #     return np.empty((0, 5))  # Return an empty array if no valid points

def phi_(N, avg_points, si_d, psi=0.5, Xi=0.001):
    dist = np.linalg.norm(avg_points - si_d)
    return (np.exp(psi * N)) / (np.log(dist + Xi + 1))

def calculate_distance(current_pose, target_pose):
    """Calculate the distance to the target in the XY plane."""
    delta = target_pose[:2] - current_pose[:2]
    return np.linalg.norm(delta)

def normalize_angle(angle):
        """Normalize angle to [-π, π]."""
        return math.atan2(math.sin(angle), math.cos(angle))
class Sim_MS:
    def __init__(self, unique_points, robot_state, center_line):
        rospy.init_node("camera_lidar_fusion_seg_node")
        self.start = False
        self.pose_received = False
        self.publish_od_pc = True
        self.max_readings = 1
        self.global_frame = "odom" # map
        self.arm_frame = "camera_0_link" # "world"
        self.robot_frame = "base_link"  
        # Topics
        self.lidar_topic = rospy.get_param("~lidar_topic", "/velodyne_points")
        self.image_topics = ["/camera_0/rgb/image_raw", "/camera_1/rgb/image_raw", "/camera_2/rgb/image_raw"]
        self.camera_frames = ["camera_0_rgb_optical_frame", "camera_1_rgb_optical_frame", "camera_2_rgb_optical_frame"]
        self.lidar_frame = rospy.get_param("~lidar_frame", "velodyne")
        self.fusion_frame = "odom" # velodyne, NOT odom        

        # self.lidar_topic = rospy.get_param("~lidar_topic", "/robot/dlio/odom_node/pointcloud/deskewed")
        # self.image_topics = ["/lucid_camera_0/image_rect_color", "/lucid_camera_1/image_rect_color", "/lucid_camera_2/image_rect_color"]
        # self.camera_frames = ["lucid_camera_0/optical_frame", "lucid_camera_1/optical_frame", "lucid_camera_2/optical_frame"]
        # self.lidar_frame = rospy.get_param("~lidar_frame", "robot/odom")
        # self.fusion_frame = self.lidar_frame#"map" # velodyne, NOT odom

        model_path = "/home/ara/catkin_ws/src/culvert_sim/model/inst_seg_sim_arti_best.pt"
        rospy.loginfo("Loading YOLO model....")
        self.model = YOLO(model_path)
        rospy.loginfo("Done loading YOLO model!")
        # Camera intrinsic parameters
        # Small scale testing matrix
        # self.cam_intrinsic = [
        #     [[532.6322518776815, 0, 320.5], [0, 532.6322518776815, 240.5], [0, 0, 1]],
        #     [[532.6322518776815, 0, 320.5], [0, 532.6322518776815, 240.5], [0, 0, 1]],
        #     [[532.6322518776815, 0, 320.5], [0, 532.6322518776815, 240.5], [0, 0, 1]],
        # ]
        # Sim big matrix
        self.cam_intrinsic = [
            [[1611.2125619299866, 0, 968.5], [0, 1611.2125619299866, 732.5], [0, 0, 1]],
            [[1611.2125619299866, 0, 968.5], [0, 1611.2125619299866, 732.5], [0, 0, 1]],
            [[1611.2125619299866, 0, 968.5], [0, 1611.2125619299866, 732.5], [0, 0, 1]]
        ]
        # MS-CAIS matrix
        # self.cam_intrinsic = [
        #     [[1086.6397021551602, 0, 972.4382905708281], [0, 1084.952616596583, 723.9269557471057], [0, 0, 1]],
        #     [[1094.185524912348, 0, 969.5755461397443], [0, 1092.22624232087, 721.3087738914188], [0, 0, 1]],
        #     [[1100.0098785812986, 0, 963.2317598612985], [0, 1097.6586457020053, 723.9269557471057], [0, 0, 1]],
        # ]

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        # Publishers
        self.culvert_points_pub = rospy.Publisher("culvert_points", Marker, queue_size=5)
        self.belief_pub = rospy.Publisher("belief", Marker, queue_size=5)
        self.culvert_pose_pub = rospy.Publisher("culvert_pose", PoseArray, queue_size=5)
        self.robot_points_pub = rospy.Publisher("robot_points", Marker, queue_size=5)
        self.target_pub = rospy.Publisher("target", Marker, queue_size=1)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=5)
        self.path_pub = rospy.Publisher('robot_path', Path, queue_size=5)
        self.pc_pub = rospy.Publisher("/seg_pc_obs", PointCloud2, queue_size=1)
        # Subscribers
        image_sub_0 = message_filters.Subscriber(self.image_topics[0], Image)
        image_sub_1 = message_filters.Subscriber(self.image_topics[1], Image)
        image_sub_2 = message_filters.Subscriber(self.image_topics[2], Image)
        pointcloud_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)
        # Synchronize the subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub_0, image_sub_1, image_sub_2, pointcloud_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        # timer event
        self.timer = rospy.Timer(rospy.Duration(1.0/10.0), self.mainCb)
    
        # Create fields for PointCloud2
        self.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        self.img_pub_0 = rospy.Publisher("/camera_0/rgb/image_raw/seg", Image, queue_size=1)
        self.img_pub_1 = rospy.Publisher("/camera_1/rgb/image_raw/seg", Image, queue_size=1)
        self.img_pub_2 = rospy.Publisher("/camera_2/rgb/image_raw/seg", Image, queue_size=1)
        
        # get only position (orientation is for future work) and add visited
        false_column = np.full((unique_points.shape[0], 1), False)
        self.culvert_points = np.hstack((unique_points, false_column))        
        self.robot_state = robot_state
        false_column = np.full((center_line.shape[0], 1), False)
        self.center_line = np.hstack((center_line, false_column)) 
        # 0 = coverage, 1 = goal, 2 = declare, 3 = exit
        self.ACTION = -1
        self.new_action = True

        self.cmd_vel = Twist()
        # observations
        
        self.robot = None
        self.arm_pose = None
        self.path_var = Path()
        self.path_var.header.frame_id = self.global_frame
        self.path_poses = []
        # POMDP stuff
        self.getObservation = False # True#
        self.reading_count = 0
        self.action_goal = None
        self.observations = None
        self.delta = 0.2
        self.OOBelief = []
        self.reward = 0
        self.reward_ = 0
        self.gamma = 0.99
        self.def_id = 0
        self.timestep = 1
        self.sigma = 0.1
        self.TP = 0.88
        self.FOV_vol = 19.32
        self.covariance = np.array([[self.sigma**2, 0, 0],
                       [0, self.sigma**2, 0],
                       [0, 0, self.sigma**2]])
        # Move/Coverage params
        self.move_goal_index = None
        self.bef_idx = None
        self.cur_goal = None
        self.previous_goal = None
        self.ang_tol = 0.1
        self.lin_tol = 0.1 #0.05
        self.ang = 0.5 #0.25
        self.lin = 0.2 #0.1

        self.start = True
        self.start_time = time.time()
        # rospy.wait_for_message("/lucid_camera_0/image_rect_color", Image)
        # rospy.wait_for_message("/lucid_camera_1/image_rect_color", Image)
        # rospy.wait_for_message("/lucid_camera_2/image_rect_color", Image)
        # rospy.wait_for_message("/robot/dlio/odom_node/pointcloud/deskewed", PointCloud2)

    def mainCb(self,event=None): 
        # self.action_goal = self.planner()
        # print("Action: ", self.ACTION, "|", self.action_goal)
        # self.update_visited()
        # self.update_belief()
        # print(self.robot)
        # self.publish_markers()
        # self.getObservation = True
        # self.updatePoseTF()
        # self.update_path()
        # self.planner()
        # print(self.ACTION)
        # self.visualizeTarget(self.cur_goal[0], self.cur_goal[1])
        # return
        # if not init or is in middle of getting observation and updating
        if not self.start or self.getObservation:
            return
        self.updatePoseTF()
        self.update_path()
        self.publish_markers()
        # need new action 
        if self.ACTION == -1:

            self.print_goal_action = True
            
            self.getObservation = True
            rospy.sleep(1)
            self.planner()
            print("plan new goal")

        if self.new_action:    
            print("==========================")
            print("Timestep ", self.timestep)        
            print("Action: ", self.ACTION, "|")
            self.timestep += 1
            self.getObservation = True
        self.new_action = False
        # move
        if self.ACTION == 0:
            # beginning or too faraway
            if self.previous_goal is None or calculate_distance(self.previous_goal, self.cur_goal) > 0.05:
                if self.goToGoal(self.cur_goal):
                    self.ACTION = 1
                    # once reach then set that previous is this in case next goal index is same
                    self.previous_goal = self.cur_goal
                    # reset
                    self.print_goal_action = True
                else:
                    # if first time, print action
                    if self.print_goal_action:
                        print(f'\r 0: Going to {self.cur_goal}')
                        self.print_goal_action = False
            else:
                print("\r 0: Already there, go to ACTION 3")
                self.ACTION = 3
                self.timestep -= 1
                self.new_action = True
        elif self.ACTION == 1:
            if self.turn_setup_for_arm(self.culvert_points[self.arm_goal_index]):
                self.ACTION = 2
                # reset                
                self.print_goal_action = True
            else:
                if self.print_goal_action:
                    print(f'Turning for arm')
                    self.print_goal_action = False
        # TODO: move_backwards for arm
        elif self.ACTION == 2:
            # no need to transform since self.arm_pose is in map frame
            # if self.move_backwards_for_arm(self.robot_state[self.move_goal_index]):
            if self.move_backwards_for_arm(self.cur_goal):
                self.ACTION = 3
                # self.move_goal_index = None
                self.cur_goal = None
                self.print_goal_action = True
                self.reward += -2 * (self.gamma **self.timestep )
                self.reward_ += -2 
                print("Reward: ", -2 * (self.gamma **self.timestep ))
                print("Total Discounted Reward: ", self.reward)
                print("Total Reward: ", self.reward_)
                self.new_action = True
            else:
                if self.print_goal_action:
                    print(f'Backwards for arm')
                    self.print_goal_action = False
        # arm deployment
        elif self.ACTION == 3:
            # visit to True
            self.OOBelief[self.bef_idx].visited()
            self.arm_goal_index = None
            self.cur_goal = None
            self.arm_deploying = False            
            self.new_action = True 
            # while True:
            #     r = input("Please enter the reward: ")
            #     try:
            #         # Try to convert the reward to an integer
            #         number = int(r)
            #         print(f"Reward entered: {number}")
            #         self.reward += number * (self.gamma **self.timestep)
            #         self.ACTION = -1
            #         self.new_action = True
            #         break  # Exit the loop once a valid integer is entered
            #     except ValueError:
            #         # If conversion fails, prompt the user to try again
            #         print("That's not a valid integer. Please try again.")
            # print("==========================")
            # print("Timestep ", self.timestep)        
            # print("Action: ", self.ACTION, "|")
            print(f"Reward: {1000 * (self.gamma **self.timestep)}")
            print("Reward: ", 1000)
            self.reward += 1000 * (self.gamma **self.timestep)
            self.reward_ += 1000 
            
            # self.update()
            # shutdown when exist
            print("Total Discounted Reward: ", self.reward)
            print("Total Reward: ", self.reward_)
            print("Set ", self.OOBelief[self.bef_idx]._id, "to ", self.OOBelief[self.bef_idx]._visited)
            self.ACTION = -1 
        # coverage
        elif self.ACTION == 4:
            if self.goToGoal(self.cur_goal):
                self.ACTION = -1
                self.new_action = True
                # once reach then set that previous is this in case next goal index is same
                self.previous_goal = self.cur_goal
                # reset
                self.print_goal_action = True
                self.reward += -2 * (self.gamma **self.timestep )
                self.reward_ += -2 
                print("Reward: ", -2 * (self.gamma **self.timestep ))
                print("Total Discounted Reward: ", self.reward)
                print("Total Reward: ", self.reward_)
            else:
                # if first time, print action
                if self.print_goal_action:
                    print(f'\r 4: Going to {self.cur_goal}')
                    self.print_goal_action = False
        elif self.ACTION == 5:
            # while True:
            #     r = input("Please enter the reward: ")
            #     try:
            #         # Try to convert the reward to an integer
            #         number = int(r)
            #         print(f"Reward entered: {number}")
            #         self.reward += number * (self.gamma **self.timestep)
            #         break  # Exit the loop once a valid integer is entered
            #     except ValueError:
            #         # If conversion fails, prompt the user to try again
            #         print("That's not a valid integer. Please try again.")
            # print(f"Reward entered: {10000 * (self.gamma **self.timestep)}")
            # self.reward += 10000 * (self.gamma **self.timestep)
            # shutdown when exist
            print("Total Discounted Reward: ", self.reward)
            print("Total Reward: ", self.reward_)
            print("Time: ", time.time() - self.start_time)
            rospy.signal_shutdown("Task Done.")
        
        # if self.new_action:
            
            
    def planner(self, dist=0.3):
        # belief is empty
        if len(self.OOBelief) < 1:
            # self.ACTION = 0
            return self.coverage()
        else:
            # Store original indices before filtering
            original_indices = np.array([i for i, def_ in enumerate(self.OOBelief) if not def_._visited])
            # filter belief not visited
            not_visited_def = [def_ for def_ in self.OOBelief if not def_._visited]
            print("\r Not visited: ", [def_._visited for def_ in self.OOBelief])
            # all beliefs are visited
            if not not_visited_def:
                # exit or coverage
                self.ACTION = 4
                return self.coverage()
            else:
                # sample belief                
                befs = np.array(self.sample_beliefs(not_visited_def))
                tree = KDTree(befs)
                # # closest sample bef
                # _, self.bef_idx = tree.query(self.robot[:3])
                # # get index from culvert_points
                # tree = KDTree(self.culvert_points[:, :3])
                # _, self.arm_goal_index = tree.query(befs[self.bef_idx])         
                
                # Closest sample belief index in filtered list
                _, filtered_idx = tree.query(self.robot[:3])

                # Get the **original** index from OOBelief
                self.bef_idx = original_indices[filtered_idx]

                # Find closest match in culvert_points
                tree = KDTree(self.culvert_points[:, :3])
                _, self.arm_goal_index = tree.query(befs[filtered_idx])  

                quat = self.culvert_points[self.arm_goal_index, 3:7]
                (_, _, yaw) = transformations.euler_from_quaternion(quat)
                x_goal = self.culvert_points[self.arm_goal_index, 0] + dist * np.cos(yaw)
                y_goal = self.culvert_points[self.arm_goal_index, 1] + dist * np.sin(yaw)
                self.cur_goal = np.array([x_goal, y_goal])
                # self.visualizeTarget(x_goal, y_goal)
                self.ACTION = 0
                # print(dist, befs[self.bef_idx], self.culvert_points[self.arm_goal_index], quat)
                # x_goal = self.culvert_points[]


    def sample_beliefs(self, not_visited_def):
        b = []
        for belif in not_visited_def:
            b.append(belif.sample_belief())
        return b              
    
    def coverage(self):
        # no more to cover
        if np.all(self.center_line[:, 3]):
            self.ACTION = 5
            return "exit"
        robot = self.robot[:3]
        unvisited_center = self.center_line[self.center_line[:, 3] == False, :3]
        distances = np.linalg.norm(unvisited_center - robot, axis=1)
        closest_idx = np.argmin(distances)
        self.cur_goal =  unvisited_center[closest_idx, 0:2]
        self.ACTION = 4
        # distances_robot_to_neighbor = np.linalg.norm(self.robot_state - robot, axis=1)
        # # neighbors that are not itself TODO: need calibrate in real life
        # neighbors = self.robot_state[(distances_robot_to_neighbor > 0.1) & (distances_robot_to_neighbor <=0.35)]
        # # distances_neighbors_to_goal = np.linalg.norm(neighbors - goal[:3], axis=1)
        # distance_robot_to_goal = np.linalg.norm(robot - goal[:3])
        # # heuristics
        # g = np.linalg.norm(neighbors - robot, axis=1)
        # h = np.linalg.norm(neighbors - goal[:3], axis=1)
        # f = g+h
        # # if you can't find anything, go to nearest center line
        # if f.size == 0:
        #     distances = np.linalg.norm(self.center_line[:, :3] - robot, axis=1)
        #     closest_idx = np.argmin(distances)
        #     self.ACTION = 0
        #     return self.center_line[closest_idx,:3]
        # best_neighbor_idx = np.argmin(f)
        # self.ACTION = 0
        # return neighbors[best_neighbor_idx]

    def callback(self, img0, img1, img2, cloud):
        if self.model is None:
            return
        
        if not self.getObservation:
            return
        images =[
            self.bridge.imgmsg_to_cv2(img0, "bgr8"),
            self.bridge.imgmsg_to_cv2(img1, "bgr8"),
            self.bridge.imgmsg_to_cv2(img2, "bgr8")
        ]
        images_result = [
        self.model(images[0],save=False, verbose=False)[0],
        self.model(images[1],save=False, verbose=False)[0],
        self.model(images[2],save=False, verbose=False)[0]
        ]
        
        combined_points = []
        # reset observations array
        observations = []
        # for each image
        for i in range(3):
            try:
                # Transform LiDAR points to the camera frame
                transform = self.tf_buffer.lookup_transform(self.camera_frames[i], self.lidar_frame, rospy.Time(0), rospy.Duration(1.0))
                transformed_points = self.transform_lidar_to_camera(cloud, transform)
                # print(type(transform), transform)
            except Exception as e:
                rospy.logerr(f"TF2 Transform Error: {e}")
                return
            # observations
            for result in images_result[i]:
                # get 
                mask = result.masks.data.cpu().numpy() > 0.5
                # print(mask[0])
                # print(i, mask.shape, mask[0].shape)
                # resize back up
                mask = cv2.resize(mask[0].astype(np.uint8), (1936,1464), interpolation=cv2.INTER_LINEAR)
                # 0 = crack = red, 1 = spalls = green
                classes = result.boxes.cls.cpu().numpy()[0]
                scores = result.boxes.conf.cpu().numpy()[0]
                # row, col = np.where(mask.any(axis=0))
                row, col = np.where(mask == 1)
                rc_coords = np.column_stack((row,col))
                """TODO: OBSERVATION """
                pc = self.get_pc_observation(transformed_points, images[0].shape, i, rc_coords, classes)
                # self.observations.append(pc)
                observations.append([pc, classes, scores])
                # colorized_pc = self.project_and_colorize(transformed_points, images[i], i, rc_coords)
                # combined_points.append(colorized_pc)
        
        self.img_pub_0.publish(self.bridge.cv2_to_imgmsg(images_result[0].plot(boxes=False), "bgr8"))
        self.img_pub_1.publish(self.bridge.cv2_to_imgmsg(images_result[1].plot(boxes=False), "bgr8"))
        self.img_pub_2.publish(self.bridge.cv2_to_imgmsg(images_result[2].plot(boxes=False), "bgr8"))
        # TODO: clear when done
        # merge overlapping observations
        self.merge_observations = merge_point_clouds(observations)
        # since they are merge, stack them
        self.merge_observations = stack_pc(self.merge_observations)
        # print(self.merge_observations.shape)
        # print(len(self.merge_observations))
        # visualization
        if self.merge_observations.size !=0 and self.publish_od_pc:  
            # cloud = np.vstack(list(zip(*self.merge_observations))[0])
            cloud = self.merge_observations[:, :4]
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.fusion_frame
            # return create_cloud(header, fields, points_with_rgb)
            cloud_convert = create_cloud(header, self.fields, cloud)
            self.pc_pub.publish(cloud_convert)
            # self.observations.clear()
        else:
            print("No PC")
        self.publish_markers()
        self.observations = obs_avg(self.culvert_points, self.merge_observations, self.delta*0.6)
        # print("obs",obs)
        # update belief
        self.update_belief()
        self.reading_count+= 1
        if self.reading_count > self.max_readings:
            self.getObservation = False
            self.reading_count = 0
        
    def update_belief(self, threshold=0.3):
        if self.observations is None:
            for def_ in self.OOBelief:
                # o^i = null
                o = 1 - self.TP
                bel = def_._belief
                bel[:, 3] *= o 
                total_b = np.sum(bel[:, 3])
                if total_b > 0:  # Avoid division by zero
                    bel[:, 3] /= total_b
            return
        if self.observations.size > 0:
            obs = self.observations[:, :3]
            # print(obs)
            # add new belief if there's none
            if len(self.OOBelief) == 0:                
                to_add = self.observations
            else:
                existing_poses = np.array([belief._pose for belief in self.OOBelief])
                distances = np.linalg.norm(obs[:, np.newaxis] - existing_poses, axis=2)
                # Check if the minimum distance to any existing belief is greater than the threshold
                min_distances = np.min(distances, axis=1)
                to_add = self.observations[min_distances >= threshold]
            
            for point in to_add:
                # print(point)
                distances = np.linalg.norm(self.culvert_points[:, :3] - np.array([point[0], point[1], point[2]]), axis=1)
                closest_idx = np.argmin(distances)
                # default class to 0 for now when update prob do class
                self.OOBelief.append(defect_belief(self.culvert_points[closest_idx][:3], self.def_id, point[3], self.culvert_points[:, :3]))
                self.def_id += 1 

            for def_ in self.OOBelief:
                # create KDtree
                tree = KDTree(self.observations[:, :3])
                dist, idx = tree.query(def_._pose)
                if dist < 0.3 * self.delta:
                    phi = phi_(self.observations[idx, 5], self.observations[idx, :3], def_._pose)
                    # We need the inverse of the covariance matrix for the PDF calculation
                    inv_covariance = np.linalg.inv(self.covariance)
                    # Compute the difference between each point and the "pose"
                    diff = def_._belief[:, :3] - self.observations[idx, :3] 
                    # Calculate the exponent term in the multivariate normal distribution
                    exponent = -0.5 * np.sum(diff @ inv_covariance * diff, axis=1)
                    # Calculate the normalization constant for the multivariate normal distribution
                    normalization_const = 1 / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(self.covariance))
                    # Compute the PDF for all points at once
                    pdf_values = normalization_const * np.exp(exponent)
                    # Reshape the result to n x 1
                    pdf_values = pdf_values.reshape(-1, 1)
                    # element-wise multiplication
                    new_belief = def_._belief[:, 3] * pdf_values.flatten() * phi
                    # normalize
                    total_sum = np.sum(new_belief)
                    new_belief /= total_sum
                    # update
                    def_._belief[:, 3] = new_belief
                    # cls update
                    if def_._cls != self.observations[idx,3] and def_._cls != 2:
                        def_._cls = 2
                elif dist < 0.6 * self.delta:
                    phi = phi_(self.observations[idx, 5], self.observations[idx, :3], def_._pose)
                    new_belief = def_._belief[:,3] * phi/self.FOV_vol
                    total_sum = np.sum(new_belief)
                    new_belief /= total_sum
                    def_._belief[:, 3] = new_belief
                    # cls update
                    if def_._cls != self.observations[idex, 3] and def_._cls != 2:
                        def_._cls = 2
                else:
                    o = 1 - self.TP
                    bel = def_._belief
                    bel[:, 3] *= o 
                    total_b = np.sum(bel[:, 3])
                    if total_b > 0:  # Avoid division by zero
                        bel[:, 3] /= total_b

    def transform_lidar_to_camera(self, pointcloud_msg, transform):
        # Convert PointCloud2 to array
        points = np.array(list(read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True)))

        # Extract translation and rotation
        translation = [transform.transform.translation.x,
                       transform.transform.translation.y,
                       transform.transform.translation.z]
        rotation = [transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w]

        # Convert quaternion to rotation matrix
        rot_matrix = R.from_quat(rotation).as_matrix()
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = translation

        # Transform points
        points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
        points_camera = (trans_matrix @ points_homo.T).T[:, :3]

        return points_camera
    """TODO:    """
    def get_pc_observation(self, points_camera, img_shape, camera_index, mask_coord, classes):
        uvs = (self.cam_intrinsic[camera_index] @ points_camera.T).T
        uvs = uvs[:, :2] / uvs[:, 2:3]  # Normalize by Z

        # Filter valid points
        valid_idx = (uvs[:, 0] >= 0) & (uvs[:, 0] < img_shape[1]) & \
                    (uvs[:, 1] >= 0) & (uvs[:, 1] < img_shape[0]) & \
                    (points_camera[:, 2] > 0)
        uvs = uvs[valid_idx].astype(int)
        points_camera = points_camera[valid_idx]

        # Convert selected_points to (col, row) format for comparison with uvs
        selected_uvs = mask_coord[:, [1, 0]]  # Swap columns to get [col, row]

        # Create a 2D array of [u, v] coordinates from uvs
        uvs_2d = uvs[:, :2]

        # Use broadcasting to find matches between uvs and selected_uvs
        match_mask = (uvs_2d[:, None] == selected_uvs).all(axis=2).any(axis=1)

        # Filter points and uvs based on the match mask
        matched_uvs = uvs[match_mask]
        matched_points = points_camera[match_mask]

        if classes == 0.0:
            color = (255 << 16) | (0 << 8) | 0  # Packed RGB value for red/crack
        elif classes == 1.0:
            color = (0 << 16) | (255 << 8) | 0  # Packed RGB value for green
        else:
            print("Class not known:, ", classes)

        # Create an array of red values with the same shape as the original
        rgb = np.full(matched_uvs.shape[0], color, dtype=np.uint32)

        # Pack RGB into float32
        rgb_as_float = rgb.view(np.float32)

        # Combine points and packed RGB
        points_with_rgb = np.zeros((matched_points.shape[0], 4), dtype=np.float32)
        points_with_rgb[:, 0:3] = matched_points  # x, y, z
        points_with_rgb[:, 3] = rgb_as_float     # packed RGB as float32
        # print(points_with_rgb.shape) = 4

        # transform matrix
        transform_matrix = self.get_transform_matrix(self.camera_frames[camera_index], self.fusion_frame)
        transformed_points = self.transform_points(points_with_rgb, transform_matrix)

        return transformed_points

    def project_and_colorize(self, points_camera, image, i, selected_points=None):
        # Project points to 2D
        uvs = (self.cam_intrinsic[i] @ points_camera.T).T
        uvs = uvs[:, :2] / uvs[:, 2:3]  # Normalize by Z

        # Filter valid points
        valid_idx = (uvs[:, 0] >= 0) & (uvs[:, 0] < image.shape[1]) & \
                    (uvs[:, 1] >= 0) & (uvs[:, 1] < image.shape[0]) & \
                    (points_camera[:, 2] > 0)
        uvs = uvs[valid_idx].astype(int)
        points_camera = points_camera[valid_idx]

        """TODO: GET OBSERVATION"""
        # for selected_point in selected_points:
        # Convert selected_points to (col, row) format for comparison with uvs
        selected_uvs = selected_points[:, [1, 0]]  # Swap columns to get [col, row]

        # Create a 2D array of [u, v] coordinates from uvs
        uvs_2d = uvs[:, :2]

        # Use broadcasting to find matches between uvs and selected_uvs
        match_mask = (uvs_2d[:, None] == selected_uvs).all(axis=2).any(axis=1)

        # Filter points and uvs based on the match mask
        matched_uvs = uvs[match_mask]
        matched_points = points_camera[match_mask]
        # print(matched_points)
        # Get colors from the image
        colors = image[matched_uvs[:, 1], matched_uvs[:, 0]]  # BGR format
        rgb = (colors[:, 2].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 0].astype(np.uint32)
        # Set colors to red (RGB: 255, 0, 0)
        red_color = (255 << 16) | (0 << 8) | 0  # Packed RGB value for red

        # Create an array of red values with the same shape as the original
        rgb = np.full(matched_uvs.shape[0], red_color, dtype=np.uint32)

        # Pack RGB into float32
        rgb_as_float = rgb.view(np.float32)

        # Combine points and packed RGB
        points_with_rgb = np.zeros((matched_points.shape[0], 4), dtype=np.float32)
        points_with_rgb[:, 0:3] = matched_points  # x, y, z
        points_with_rgb[:, 3] = rgb_as_float     # packed RGB as float32
        # print(points_with_rgb.shape) = 4

        # Transform points to the desired frame
        transform_matrix = self.get_transform_matrix(self.camera_frames[i], self.fusion_frame)
        transformed_points = self.transform_points(points_with_rgb, transform_matrix)

        return transformed_points

    def get_transform_matrix(self, current, to):
        """
        Retrieve the transform between two frames and return a 4x4 transformation matrix.
        """
        try:
            # Lookup the transform
            transform = self.tf_buffer.lookup_transform(
                to, current, rospy.Time(0), rospy.Duration(1.0)
            )
        except Exception as e:
            rospy.logerr(f"TF2 Error: {e}")
            return None

        # Extract translation and rotation
        translation = [transform.transform.translation.x,
                       transform.transform.translation.y,
                       transform.transform.translation.z]
        rotation = [transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w]

        # Convert quaternion to rotation matrix
        rot_matrix = R.from_quat(rotation).as_matrix()

        # Build the 4x4 transformation matrix
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = translation
        return trans_matrix

    def transform_xyz_points(self, points, transform_matrix):
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points, ones))  # Shape: (N, 4)

        # Apply the transformation matrix
        transformed_points_homogeneous = (transform_matrix @ homogeneous_points.T).T
        return transformed_points_homogeneous[:, :3]
        
    def transform_points(self, points, transform_matrix):
        """
        Apply a 4x4 transformation matrix to the first 3 columns of a points array.
        
        Parameters:
            points (np.ndarray): Array of shape (N, 4) where the first 3 columns are x, y, z.
            transform_matrix (np.ndarray): 4x4 homogeneous transformation matrix.
        
        Returns:
            np.ndarray: Transformed points array of shape (N, 4).
        """
        # Extract x, y, z and append a column of ones for homogeneous coordinates
        xyz = points[:, :3]
        ones = np.ones((xyz.shape[0], 1))
        homogeneous_points = np.hstack((xyz, ones))  # Shape: (N, 4)

        # Apply the transformation matrix
        transformed_points_homogeneous = (transform_matrix @ homogeneous_points.T).T
        # Retain transformed x, y, z and the original 4th column
        transformed_points = np.hstack((transformed_points_homogeneous[:, :3], points[:, 3:4]))

        return transformed_points

    def updatePoseTF(self):
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
            self.robot = np.array([x, y, z, yaw])

            tf_arm = self.tf_buffer.can_transform(self.global_frame, self.arm_frame, rospy.Time(0), rospy.Duration(0.5))
            transform_arm = self.tf_buffer.lookup_transform(self.global_frame, self.arm_frame, rospy.Time(0), rospy.Duration(0.3))
            x_arm = transform_arm.transform.translation.x
            y_arm = transform_arm.transform.translation.y
            z_arm = transform_arm.transform.translation.z

            # Extract rotation (quaternion)
            q_arm = transform_arm.transform.rotation
            quaternion_arm = [q_arm.x, q_arm.y, q_arm.z, q_arm.w]

            # ✅ Convert quaternion to Euler angles (roll, pitch, yaw)
            (roll, pitch, yaw_arm) = transformations.euler_from_quaternion(quaternion_arm)
            self.arm_pose = np.array([x_arm, y_arm, z_arm, yaw_arm, 1.0])
            self.pose_received = True
            # print(self.robot)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Transform not available yet...")

    def publish_robot_points_marker(self):
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
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_points"
        marker.id = 100
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Adjust size of points as needed
        marker.color.b = 1.0
        marker.color.a = 1.0
        # Convert points to Marker format
        # for point in self.robot_state:
        #     p = Point()
        #     p.x = point[0]
        #     p.y = point[1]
        #     p.z = point[2]
        #     marker.points.append(p)
        #     color = ColorRGBA()
        #     color.r, color.g, color.b, color.a = 0.0, 0.0, 1.0, 1.0
        #     marker.colors.append(color)  
                
        for point in self.center_line:
            p = Point()
            p.x, p.y, p.z = point[:3]
            marker.points.append(p)
            color = ColorRGBA()
            if point[-1]:
                color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 1.0
            else:
                color.r, color.g, color.b, color.a = 1.0, 0.5, 0.0, 1.0
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

    def publish_culvert_points_marker(self):
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
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "culvert_points"
        marker.id = 10
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Adjust size of points as needed
        marker.color.r = 1.0
        marker.color.a = 1.0

        # Convert points to Marker format
        for point in self.culvert_points:
            p = Point()
            p.x, p.y, p.z = point[:3]
            marker.points.append(p)
            color = ColorRGBA()
            if point[-1]:
                color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 1.0
            else:
                color.r, color.g, color.b, color.a = 1.0, 1.0, 1.0, 1.0
            marker.colors.append(color)        
        self.culvert_points_pub.publish(marker)
        # pose_array = PoseArray()
        # pose_array.header.frame_id = "odom"
        # pose_array.header.stamp = rospy.Time.now()

        # point = self.culvert_points[111]
        # pose = Pose()
        # pose.position.x = point[0]
        # pose.position.y = point[1]
        # pose.position.z = point[2]
        # pose.orientation.x = point[3]
        # pose.orientation.y = point[4]
        # pose.orientation.z = point[5]
        # pose.orientation.w = point[6]
        # pose_array.poses.append(pose)
        # self.culvert_pose_pub.publish(pose_array)

    def publish_markers(self):
        self.publish_culvert_points_marker()
        self.publish_robot_points_marker()
        self.publish_max_b_marker()
    
    def publish_max_b_marker(self):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "belief"
        marker.id = 20
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.06  # Adjust size of points as needed
        marker.scale.y = 0.06  # Adjust size of points as needed
        for bef in self.OOBelief:            
            p = Point()
            # p.x, p.y, p.z = bef._pose
            b = bef.sample_belief_b()
            p.x, p.y, p.z = b[:3]
            marker.points.append(p)
            color = ColorRGBA()
            # spalls
            if bef._cls == 1.0:
                # color.r, color.g, color.b, color.a = 0, 1, 0, 1.0
                color.r, color.g, color.b, color.a = 1- b[3], 1, 1-b[3], 1.0                
            # cracks
            elif bef._cls == 0.0:
                # color.r, color.g, color.b, color.a = 1, 0, 0, 1.0
                color.r, color.g, color.b, color.a = 1, 1-b[3], 1-b[3], 1.0
            # both
            else:
                color.r, color.g, color.b, color.a = 1, 1, 1- b[3], 1.0
            marker.colors.append(color)       
        self.belief_pub.publish(marker)

    def update_path(self):
        if not self.start or not self.pose_received:
            return
        # update centerline
        idx = np.abs(self.center_line[:,0] - self.robot[0]) <= 1.0#0.5
        self.center_line[idx,3] = True
        # update path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = self.global_frame
        pose_stamped.pose.position.x = self.robot[0]
        pose_stamped.pose.position.y = self.robot[1]
        pose_stamped.pose.position.z = self.robot[2]
        self.path_poses.append(pose_stamped)
        self.path_var.poses = self.path_poses
        self.path_var.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path_var)

    # TODO: move backwards needs update from real life code
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

        # angle_diff = abs(shortest_angular_difference(self.robot[3], angleToGoal)) 
        # print(goal)
        # print(self.robot[3], "->", angleToGoal, distance)
        # if angle_diff < self.ang_tol:
        #     lin_vel = self.lin
        #     ang_vel = 0
        # else:
        #     lin_vel = 0
        #     ang_vel = turning(self.robot[3], angleToGoal, self.ang)
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
        # closest_angle = np.pi if goal[1] > 0 else 0#np.pi if self.data_index > 3 else 0
        closest_angle = 0
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
        # print("distance: ", distance)
        if distance < 0.1:
            self.publish_velocity(0, 0)
            return True
        self.publish_velocity(-self.lin, 0)
        # heading_error = calculate_back_heading_error(self.current_pose, self.target_pose)
    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        """Publishes velocity commands to cmd_vel."""
        self.cmd_vel.linear.x = linear_x
        self.cmd_vel.angular.z = angular_z
        self.cmd_pub.publish(self.cmd_vel)

if __name__ == "__main__":
    try:
        # Retrieve parameters from the ROS parameter server or use defaults
        w = rospy.get_param('~w', 1.2)
        l = rospy.get_param('~l', 4.0)
        h = rospy.get_param('~h', 0.7)
        scale = rospy.get_param('~scale', 3.5)
        offset_x = rospy.get_param('~offset_x', 0.0)
        offset_y = rospy.get_param('~offset_y', -0.6 + 0.013)
        offset_z = rospy.get_param('~offset_z', 0.0)

        rospy.loginfo(offset_x, offset_y, offset_z)

        h_fov = rospy.get_param('~h_fov', 61)
        v_fov = rospy.get_param('~v_fov', 49)
        near = rospy.get_param('~near', 0.2)
        far = rospy.get_param('~far', 3.0)

        robot_l = 1.0
        robot_offset_x = rospy.get_param('~robot_offset_x', offset_x-1.0)
        robot_offset_y = rospy.get_param('~robot_offset_y', offset_y+0.287)
        robot_offset_z = rospy.get_param('~robot_offset_y', offset_z)
        robot_y_scale = rospy.get_param('~robot_y_scale', 7)

        # generate culvert points & rgbd frostum
        unique_points = generate_uniform_grid(w, l, h, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z, scale=scale)
        robot_state, center_line = generate_robot_state(w/2, l+robot_l, offset_x=robot_offset_x, offset_y=robot_offset_y, offset_z=robot_offset_z, scale_x=scale, scale_y=robot_y_scale)
        unique_points = unique_points[unique_points[:, 2] < 0.6]
        # sim
        print(unique_points.shape)
        sim = Sim_MS(unique_points, robot_state, center_line)
        # sim.publish_markers()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass