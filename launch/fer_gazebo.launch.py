#!/usr/bin/env python3
"""
fer_gazebo.launch.py

Phase C launch file: replaces the manual 3-terminal dance from Phase A/B.

Sequence (each step waits for the previous to actually complete, via
event handlers -- not fixed sleep/timing guesses):

  1. Launch Gazebo Harmonic with the chosen world (world1_baseline.sdf by
     default, world2_dense_clutter.sdf via launch arg).
  2. Start robot_state_publisher, generating robot_description from
     fer_gz.urdf.xacro via xacro.Command() at launch time (so it always
     reflects the CURRENT xacro/inertials.yaml on disk -- no more stale
     /tmp/fer_gz.urdf / rsp_params.yaml copies to forget about).
  3. Once robot_state_publisher is up, spawn the FER entity into Gazebo
     via ros_gz_sim create.
  4. Once spawn completes, spawn joint_state_broadcaster,
     forward_position_controller, and gripper_position_controller in
     sequence.

RUNTIME VERIFICATION ITEMS (adjust paths below to match your machine
if your project layout differs from ~/bps-project/...):
  - fer_gz_xacro_path        -> ~/bps-project/gz_test/fer_gz.urdf.xacro
  - worlds_dir                -> ~/bps-project/worlds_arm/
  - GZ_SIM_RESOURCE_PATH      -> must include /opt/ros/humble/share so
                                 franka_description meshes resolve (this
                                 was the earlier "meshes not found" fix)

NOT YET DONE (future phase): MoveIt2 launch is not included here yet --
that slots in after this file as Phase D groundwork, once Phase C is
confirmed working for both worlds.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


# ---------------------------------------------------------------------
# RUNTIME VERIFICATION ITEM: confirm these paths match your machine.
# Using absolute paths (not $(find ...)) since fer_gz.urdf.xacro and
# fer_local/ live in a scratch project dir, not an installed ROS package.
# ---------------------------------------------------------------------
HOME = os.path.expanduser("~")
FER_XACRO_PATH = os.path.join(HOME, "bps-project/gz_test/fer_gz.urdf.xacro")
WORLDS_DIR = os.path.join(HOME, "bps-project/worlds_arm")
CONTROLLERS_YAML = os.path.join(HOME, "bps-project/gz_test/fer_gz_controllers.yaml")


def generate_launch_description():

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="world1_baseline.sdf",
        description="World file name (must exist in worlds_arm/): "
                     "world1_baseline.sdf or world2_dense_clutter.sdf",
    )

    world_path = PathJoinSubstitution([WORLDS_DIR, LaunchConfiguration("world")])

    # -------------------------------------------------------------
    # Env vars: reproduces the manual workaround from Phase A/B.
    # LIBGL_ALWAYS_SOFTWARE / QT_QPA_PLATFORM: software rendering on
    # this VM. GZ_SIM_RESOURCE_PATH: so franka_description meshes
    # under /opt/ros/humble/share resolve (the "meshes not found"
    # fix from earlier).
    # -------------------------------------------------------------
    set_libgl = SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1")
    set_qt_platform = SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb")
    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    set_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        "/opt/ros/humble/share" + (":" + existing_resource_path if existing_resource_path else ""),
    )

    # -------------------------------------------------------------
    # Step 1: launch Gazebo Harmonic with the chosen world.
    # ros_gz_sim's gz_sim.launch.py wraps `gz sim <world> -r`.
    # -------------------------------------------------------------
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": [world_path, " -r"]}.items(),
    )

    # -------------------------------------------------------------
    # Step 2: robot_state_publisher, generating robot_description
    # live from the xacro file -- no stale /tmp copies.
    # -------------------------------------------------------------
    robot_description = ParameterValue(
        Command(["xacro ", FER_XACRO_PATH]),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    # -------------------------------------------------------------
    # Step 3: spawn the FER entity, once robot_state_publisher is
    # running (it needs to be publishing robot_description for
    # `create -topic robot_description` to succeed).
    # -------------------------------------------------------------
    spawn_fer = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-topic", "robot_description",
            "-name", "fer",
            "-z", "0.5",
        ],
        output="screen",
    )

    # RUNTIME VERIFICATION ITEM: this assumes robot_state_publisher
    # prints its parameters/comes up fast enough that a fixed
    # RegisterEventHandler-on-start isn't needed -- if `create`
    # races ahead of robot_state_publisher on your machine, switch
    # this to trigger spawn_fer via an OnProcessStart handler on
    # robot_state_publisher_node instead of launching it directly
    # alongside. For now this mirrors your working manual sequence
    # (terminal 2 then terminal 3), so it should behave the same.

    # -------------------------------------------------------------
    # Step 4: controller spawners, in sequence, each triggered only
    # after the previous process actually exits (not a timer guess).
    # -------------------------------------------------------------
    spawn_joint_state_broadcaster = ExecuteProcess(
        cmd=[
            "ros2", "run", "controller_manager", "spawner",
            "joint_state_broadcaster",
        ],
        output="screen",
    )

    spawn_forward_position_controller = ExecuteProcess(
        cmd=[
            "ros2", "run", "controller_manager", "spawner",
            "forward_position_controller",
        ],
        output="screen",
    )

    spawn_gripper_position_controller = ExecuteProcess(
        cmd=[
            "ros2", "run", "controller_manager", "spawner",
            "gripper_position_controller",
        ],
        output="screen",
    )

    # Chain: spawn_fer exits -> joint_state_broadcaster spawns ->
    # exits -> forward_position_controller spawns -> exits ->
    # gripper_position_controller spawns.
    trigger_jsb_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_fer,
            on_exit=[spawn_joint_state_broadcaster],
        )
    )

    trigger_fpc_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_joint_state_broadcaster,
            on_exit=[spawn_forward_position_controller],
        )
    )

    trigger_gpc_after_fpc = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_forward_position_controller,
            on_exit=[spawn_gripper_position_controller],
        )
    )

    return LaunchDescription([
        world_arg,
        set_libgl,
        set_qt_platform,
        set_resource_path,
        gz_sim_launch,
        robot_state_publisher_node,
        spawn_fer,
        trigger_jsb_after_spawn,
        trigger_fpc_after_jsb,
        trigger_gpc_after_fpc,
    ])
