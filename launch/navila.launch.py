from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import os
import yaml
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("moma_vla")
    config_dir = os.path.join(pkg_share, "config")

    # pkg_share = <ws>/install/moma_vla/share/moma_vla
    # ws_root   = <ws>
    ws_root = os.path.normpath(os.path.join(pkg_share, "..", "..", "..", ".."))
    default_model_path = os.path.join(ws_root, "models", "navila-llama3-8b-8f")

    # yaml에서 navila_node의 command_topic을 읽어 regex_parshing_node의
    # input_topic과 동일하게 맞춤
    config_file = os.path.join(config_dir, "navila.yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    navila_output_topic = (
        config.get("navila_node", {})
        .get("ros__parameters", {})
        .get("command_topic", "/navila/output")
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value="navila.yaml",
            description="Config filename inside the config directory",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value=default_model_path,
            description="Path to NaVILA model directory",
        ),
        DeclareLaunchArgument(
            "model_base",
            default_value="",
            description="Base model path (LoRA 사용 시에만 지정)",
        ),
        Node(
            package="moma_vla",
            executable="navila_node",
            name="navila_node",
            output="screen",
            parameters=[
                PathJoinSubstitution([config_dir, LaunchConfiguration("params_file")]),
                {
                    "model_path": LaunchConfiguration("model_path"),
                    "model_base": LaunchConfiguration("model_base"),
                },
            ],
        ),
        Node(
            package="moma_vla",
            executable="regex_parshing_node",
            name="regex_parshing_node",
            output="screen",
            parameters=[{
                "input_topic": navila_output_topic,
            }],
        ),
    ])
