from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("MoMa_vla")
    default_params = os.path.join(pkg_share, "config", "navila_params.yaml")

    # pkg_share = <ws>/install/MoMa_vla/share/MoMa_vla
    # ws_root   = <ws>
    ws_root = os.path.normpath(os.path.join(pkg_share, "..", "..", "..", ".."))
    default_model_path = os.path.join(ws_root, "models", "navila-llama3-8b-8f")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to navila_params.yaml",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value=default_model_path,
            description="Path to NaVILA model directory",
        ),
        Node(
            package="MoMa_vla",
            executable="navila_node",
            name="navila_node",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"model_path": LaunchConfiguration("model_path")},
            ],
        ),
    ])
