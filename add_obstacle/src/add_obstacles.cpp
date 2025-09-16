#include <ros/ros.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include <moveit_msgs/CollisionObject.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

void addCollisionObject(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface)
{
    std::vector<moveit_msgs::CollisionObject> collision_objects;
    collision_objects.resize(2);

    // add computer at the back
    collision_objects[0].id = "IVX";
    collision_objects[0].header.frame_id = "link_base";
    // dimension
    collision_objects[0].primitives.resize(1);
    collision_objects[0].primitives[0].type = shape_msgs::SolidPrimitive::BOX;
    collision_objects[0].primitives[0].dimensions.resize(3);
    collision_objects[0].primitives[0].dimensions[0] = 0.09;  // X dimension
    collision_objects[0].primitives[0].dimensions[1] = 0.4;  // Y dimension 
    collision_objects[0].primitives[0].dimensions[2] = 0.25;  // Z dimension height

    // pose
    collision_objects[0].primitive_poses.resize(1);
    collision_objects[0].primitive_poses[0].position.x = -0.17-0.05 - 0.045;
    collision_objects[0].primitive_poses[0].position.y = 0.0;
    collision_objects[0].primitive_poses[0].position.z = 0.15;
    collision_objects[0].primitive_poses[0].orientation.w = 1.0;
    // add
    collision_objects[0].operation = collision_objects[0].ADD;

    // add perpection module in front
    collision_objects[1].id = "perpection";
    collision_objects[1].header.frame_id = "link_base";
    // dimension
    collision_objects[1].primitives.resize(1);
    collision_objects[1].primitives[0].type = shape_msgs::SolidPrimitive::BOX;
    collision_objects[1].primitives[0].dimensions.resize(3);
    collision_objects[1].primitives[0].dimensions[0] = 0.29;  // X dimension
    collision_objects[1].primitives[0].dimensions[1] = 0.4;  // Y dimension 
    collision_objects[1].primitives[0].dimensions[2] = 0.30;  // Z dimension height

    // pose
    collision_objects[1].primitive_poses.resize(1);
    collision_objects[1].primitive_poses[0].position.x = 0.125+0.05 + 0.145;
    collision_objects[1].primitive_poses[0].position.y = 0.0;
    collision_objects[1].primitive_poses[0].position.z = 0.125;
    collision_objects[1].primitive_poses[0].orientation.w = 1.0;
    // add
    collision_objects[1].operation = collision_objects[0].ADD;

    planning_scene_interface.applyCollisionObjects(collision_objects);

}

int main(int argc, char**argv)
{
    ros::init(argc, argv, "add_obstacles");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);

    ros::WallDuration(1.0).sleep();
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    moveit::planning_interface::MoveGroupInterface group("xarm5");
    group.setPlanningTime(45.0);

    addCollisionObject(planning_scene_interface);

    ros::WallDuration(1.0).sleep();

    ros::waitForShutdown();
    return 0;

}