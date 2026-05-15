import asyncio
import json
import threading

import cv2
import numpy as np
import rclpy
import websockets
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class ServerNode(Node):
    def __init__(self):
        super().__init__("server_node")

        self.declare_parameter("websocket_port", 8080)
        port = self.get_parameter("websocket_port").value

        # navila_node가 BEST_EFFORT로 구독하므로 동일하게 맞춤
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(Image, "/camera/image_raw", qos)
        self._port = port
        self._client = None
        self._loop = None

        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)

        t = threading.Thread(target=self._start_ws_server, daemon=True)
        t.start()

        self.get_logger().info(f"서버 노드 시작 | WebSocket 포트: {port}")

    def _start_ws_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_server())

    def _publish_frame(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = h
        msg.width = w
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = w * 3
        msg.data = frame_bgr.tobytes()
        self.pub.publish(msg)

    def _cmd_vel_callback(self, msg: Twist):
        if self._client is None or self._loop is None:
            return
        data = json.dumps({
            "linear":  {"x": msg.linear.x,  "y": msg.linear.y,  "z": msg.linear.z},
            "angular": {"x": msg.angular.x, "y": msg.angular.y, "z": msg.angular.z},
        })
        asyncio.run_coroutine_threadsafe(self._client.send(data), self._loop)

    async def _handle(self, websocket):
        self._client = websocket
        self.get_logger().info("클라이언트 연결됨")
        try:
            async for data in websocket:
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._publish_frame(frame)
                else:
                    self.get_logger().warn("이미지 디코딩 실패")
        except websockets.exceptions.ConnectionClosed:
            self.get_logger().info("클라이언트 연결 종료")
        finally:
            self._client = None

    async def _ws_server(self):
        async with websockets.serve(self._handle, "0.0.0.0", self._port):
            await asyncio.Future()


def main(args=None):
    rclpy.init(args=args)
    node = ServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
