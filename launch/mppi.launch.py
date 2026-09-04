from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"expected a boolean launch value, got '{value}'")


def _launch_node(context):
    parameters = {
        "config_directory": LaunchConfiguration("config_directory"),
        "state_topic": LaunchConfiguration("state_topic"),
        "trajectory_topic": LaunchConfiguration("trajectory_topic"),
        "scan_topic": LaunchConfiguration("scan_topic"),
        "base_frame": LaunchConfiguration("base_frame"),
        "laser_frame": LaunchConfiguration("laser_frame"),
        "scan_qos_depth": LaunchConfiguration("scan_qos_depth"),
        "maximum_state_age_s": LaunchConfiguration("maximum_state_age_s"),
        "future_tolerance_s": LaunchConfiguration("future_tolerance_s"),
        "require_absolute_yaw": LaunchConfiguration("require_absolute_yaw"),
        "require_vesc": LaunchConfiguration("require_vesc"),
        "state_qos_depth": LaunchConfiguration("state_qos_depth"),
        "publish_visualization": LaunchConfiguration("publish_visualization"),
        "visualization_frame_id": LaunchConfiguration("visualization_frame_id"),
        "expected_path_topic": LaunchConfiguration("expected_path_topic"),
        "rollouts_topic": LaunchConfiguration("rollouts_topic"),
        "raceline_path_topic": LaunchConfiguration("raceline_path_topic"),
        "track_left_boundary_topic": LaunchConfiguration(
            "track_left_boundary_topic"
        ),
        "track_right_boundary_topic": LaunchConfiguration(
            "track_right_boundary_topic"
        ),
    }

    # Empty launch arguments leave these values to config/mppi.yaml. A supplied
    # value remains a normal launch-time ROS parameter override.
    yaml_overrides = {
        "require_solution_validity": _parse_bool,
        "publish_direct_control": _parse_bool,
        "direct_control_topic": str,
        "control_mode": str,
        "direct_control_torque_to_throttle_scale": float,
        "direct_control_throttle_min": float,
        "direct_control_throttle_max": float,
        "direct_control_steering_scale": float,
        "direct_control_steering_limit_rad": float,
    }
    for name, convert in yaml_overrides.items():
        value = LaunchConfiguration(name).perform(context)
        if value:
            parameters[name] = convert(value)

    return [
        Node(
            package="xx_mppi",
            executable="xx_mppi_node",
            name="xx_mppi_node",
            output="screen",
            parameters=[parameters],
        )
    ]


def generate_launch_description():
    # Resolve through the ament index so this works for both normal and
    # --symlink-install workspaces. Paths inside model.yaml are subsequently
    # resolved relative to this directory by the controller config loader.
    default_config = PathJoinSubstitution([FindPackageShare("xx_mppi"), "config"])
    arguments = [
        DeclareLaunchArgument("config_directory", default_value=default_config),
        DeclareLaunchArgument("state_topic", default_value="ekf/state"),
        DeclareLaunchArgument(
            "trajectory_topic", default_value="vehicle_control_trajectory"
        ),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("laser_frame", default_value="laser"),
        DeclareLaunchArgument("scan_qos_depth", default_value="1"),
        DeclareLaunchArgument("publish_direct_control", default_value=""),
        DeclareLaunchArgument("direct_control_topic", default_value=""),
        DeclareLaunchArgument("control_mode", default_value=""),
        DeclareLaunchArgument(
            "direct_control_torque_to_throttle_scale", default_value=""
        ),
        DeclareLaunchArgument("direct_control_throttle_min", default_value=""),
        DeclareLaunchArgument("direct_control_throttle_max", default_value=""),
        DeclareLaunchArgument("direct_control_steering_scale", default_value=""),
        DeclareLaunchArgument("direct_control_steering_limit_rad", default_value=""),
        DeclareLaunchArgument("maximum_state_age_s", default_value="0.10"),
        DeclareLaunchArgument("future_tolerance_s", default_value="0.02"),
        DeclareLaunchArgument("require_solution_validity", default_value=""),
        DeclareLaunchArgument("require_absolute_yaw", default_value="true"),
        DeclareLaunchArgument("require_vesc", default_value="true"),
        DeclareLaunchArgument("state_qos_depth", default_value="1"),
        DeclareLaunchArgument("publish_visualization", default_value="false"),
        DeclareLaunchArgument("visualization_frame_id", default_value="map"),
        DeclareLaunchArgument(
            "expected_path_topic", default_value="xx_mppi/expected_path"
        ),
        DeclareLaunchArgument(
            "rollouts_topic", default_value="xx_mppi/rollouts"
        ),
        DeclareLaunchArgument(
            "raceline_path_topic", default_value="xx_mppi/raceline"
        ),
        DeclareLaunchArgument(
            "track_left_boundary_topic",
            default_value="xx_mppi/track_left_boundary",
        ),
        DeclareLaunchArgument(
            "track_right_boundary_topic",
            default_value="xx_mppi/track_right_boundary",
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_node)])
