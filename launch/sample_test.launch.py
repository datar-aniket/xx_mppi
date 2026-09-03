from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_directory = PathJoinSubstitution(
        [FindPackageShare("xx_mppi"), "config"]
    )
    sample_topic = "xx_mppi/sample_ekf_state"
    return LaunchDescription(
        [
            Node(
                package="xx_mppi",
                executable="publish_sample_ekf_state.py",
                name="xx_mppi_sample_ekf_state",
                output="screen",
                parameters=[{"state_topic": sample_topic}],
            ),
            Node(
                package="xx_mppi",
                executable="xx_mppi_node",
                name="xx_mppi_node",
                output="screen",
                parameters=[
                    {
                        "config_directory": config_directory,
                        "state_topic": sample_topic,
                        "trajectory_topic": "xx_mppi/sample_vehicle_control_trajectory",
                        # Never inherit vehicle actuation from mppi.yaml in this
                        # isolated visualization/trajectory bench launch.
                        "publish_direct_control": False,
                        "direct_control_topic": "xx_mppi/sample_direct_control",
                        "publish_visualization": True,
                    }
                ],
            ),
        ]
    )
