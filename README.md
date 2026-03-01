# MS-AIS: Multi-Sensor Autonomous Inspection System
**The effective operation of civil infrastructure is crucial for economic stability. To ensure continued performance, regular maintenance is essential. However, underground infrastructure, like culverts, posed more significant challenges in managing and preserving these critical assets, such as maneuverability and danger, slow labor intensive, defect localization, and superficial assessment. In this paper, we propose a cost-effective solution for infrastructure inspection through the development of MS-AIS, an autonomous inspection system. Our solution integrates multiple vision cameras, a densed LiDAR, a deep learning based defect segmentation system, lighting systems, and non-destructive evaluation (NDE) methods for a comprehensive condition assessment. The system consists of three different autonomous modes that support targeted inspection, visual inspection, and full inspection. Experimental validation simulated and real-world culverts demonstrates the effectiveness of MS-AIS in enhancing inspection effectiveness and practicality.**

<p align='center'>
    <img src="pic/robot(2).png" alt="drawing" width="800"/>
</p>

## Dependencies
The framework has been tested with ROS Noetic and Ubuntu 20.04. The following configuration, along with the required dependencies, has been verified for compatibility:

- [Ubuntu 20.04](https://releases.ubuntu.com/focal/)
- [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) 
- [Depthai Dependencies](https://docs.luxonis.com/software/ros/depthai-ros/build/)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (Recommend to use CUDA toolkit >= 11 for Ubuntu 20.04)
- [ultralytics](https://github.com/ultralytics)


## ROS package
- [depthai-ros](https://github.com/luxonis/depthai-ros/tree/noetic)
- [lucid_camera_driver]() (private third-party; not publicly available) -> use [arena_camera_ros](https://github.com/lucidvisionlabs/arena_camera_ros) instead.
- [kalibr](https://github.com/ethz-asl/kalibr) for camera calibration
- [multi_sensor_alignment](https://github.com/Nosille/multi_sensor_alignment)
- [dlio](https://github.com/vectr-ucla/direct_lidar_inertial_odometry)
- [bunker_ros](https://github.com/agilexrobotics/bunker_ros)
- [turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) (for simulations)
- [turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations) (for simulations)
- [cv_bridge](https://github.com/ros-perception/vision_opencv)
- [velodyne](https://github.com/ros-drivers/velodyne) (for simulations)
- [gazebo](https://gazebosim.org/docs/latest/ros_installation/)
- [rosserial](https://github.com/ros-drivers/rosserial)
- [ouster_ros](https://github.com/ouster-lidar/ouster-ros)
- [rosserial_arduino](http://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup)
- [xarm_ros](https://github.com/xArm-Developer/xarm_ros)
- [ER_resipod]() (private third-party; not publicly available)

## Install
Use the following commands to download and build the package: (The code is implemented in ROS1)

```
# caktin_ws or your workspace dir 
mkdir -p ~/catkin_ws/src 
cd ~/caktin_ws/src    
git clone https://github.com/aralab-unr/MS_CAIS.git
cd ..
catkin build
source devel/setup.bash
```
Put [bigger_rough_3crack_2spall](https://github.com/aralab-unr/MS_CAIS/tree/master/model/bigger_rough_3crack_2spall) and all other model folder in ```.gazebo/model``` folder

```
roscd culvert_sim
cd model
mv bigger_rough_3crack_2spall ~/.gazebo/model/
```

## Simulations
```
# launch sim environment
# 5DF
roslaunch culvert_sim turtlebot3_culvert.launch
rosrun culvert_sim sim_ms_test.py
# 10DF
roslaunch turtlebot3_culvert_2x.launch
rosrun culvert_sim sim_ms_test_10df.py
```
## Results
### Simulations
#### 5DF
##### MOS
<p align='center'>
    <img src="pic/mos-5df.png" alt="drawing" width="800"/>
</p>

##### AIS
<p align='center'>
    <img src="pic/5df_ais.png" alt="drawing" width="800"/>
</p>

### Indoor
<p align='center'>
    <img src="pic/indoor_fusion_traj.png" alt="drawing" width="800"/>
</p>

### Outdoor
<p align='center'>
    <img src="pic/outdoor_fusion_traj.png" alt="drawing" width="800"/>
</p>
