from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = get_package_share_directory("xx_mppi") + "/config"
    arguments = [
        DeclareLaunchArgument("config_directory", default_value=default_config),
        DeclareLaunchArgument("state_topic", default_value="ekf/state"),
        DeclareLaunchArgument(
            "trajectory_topic", default_value="vehicle_control_trajectory"
        ),
        DeclareLaunchArgument("maximum_state_age_s", default_value="0.10"),
        DeclareLaunchArgument("future_tolerance_s", default_value="0.02"),
        DeclareLaunchArgument("require_absolute_yaw", default_value="true"),
        DeclareLaunchArgument("require_vesc", default_value="true"),
        DeclareLaunchArgument("state_qos_depth", default_value="1"),
        DeclareLaunchArgument("steering_scale_to_rad", default_value="1.0"),
        DeclareLaunchArgument("steering_offset_rad", default_value="0.0"),
        DeclareLaunchArgument("torque_scale_to_nm", default_value="1.0"),
        DeclareLaunchArgument("motor_speed_scale_to_mps", default_value="1.0"),
    ]
    node = Node(
        package="xx_mppi",
        executable="xx_mppi_node",
        name="xx_mppi_node",
        output="screen",
        parameters=[
            {
                "config_directory": LaunchConfiguration("config_directory"),
                "state_topic": LaunchConfiguration("state_topic"),
                "trajectory_topic": LaunchConfiguration("trajectory_topic"),
                "maximum_state_age_s": LaunchConfiguration("maximum_state_age_s"),
                "future_tolerance_s": LaunchConfiguration("future_tolerance_s"),
                "require_absolute_yaw": LaunchConfiguration("require_absolute_yaw"),
                "require_vesc": LaunchConfiguration("require_vesc"),
                "state_qos_depth": LaunchConfiguration("state_qos_depth"),
                "steering_scale_to_rad": LaunchConfiguration(
                    "steering_scale_to_rad"
                ),
                "steering_offset_rad": LaunchConfiguration("steering_offset_rad"),
                "torque_scale_to_nm": LaunchConfiguration("torque_scale_to_nm"),
                "motor_speed_scale_to_mps": LaunchConfiguration(
                    "motor_speed_scale_to_mps"
                ),
            }
        ],
    )
    return LaunchDescription(arguments + [node])
