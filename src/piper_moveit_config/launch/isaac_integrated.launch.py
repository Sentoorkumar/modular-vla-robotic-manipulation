"""
isaac_integrated.launch.py
MoveIt2 integrated with Isaac Sim (Method A bridge), on SIMULATION TIME.

Starts: robot_state_publisher + move_group + rviz (all use_sim_time=True).
Isaac Sim provides /joint_states AND /clock.
Loads moveit_controllers.yaml -> MoveIt sends plans to trajectory_bridge.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("piper", package_name="piper_moveit_config")
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    use_sim_time = {"use_sim_time": True}

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description, use_sim_time],
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            use_sim_time,
            {"trajectory_execution.allowed_start_tolerance": 0.5},
            {"trajectory_execution.execution_duration_monitoring": False},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", str(moveit_config.package_path / "config/moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            use_sim_time,
        ],
    )

    return LaunchDescription([rsp_node, move_group_node, rviz_node])
