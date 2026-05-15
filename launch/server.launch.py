from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "websocket_port",
            default_value="8080",
            description="WebSocket 서버 포트",
        ),
        Node(
            package="na_vila_ros",
            executable="server_node",
            name="server_node",
            output="screen",
            parameters=[{
                "websocket_port": LaunchConfiguration("websocket_port"),
            }],
        ),
    ])
