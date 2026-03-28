"""
NaVILA 모델의 텍스트 출력을 정규식으로 파싱하여 cmd_vel 명령으로 변환합니다.

Subscribes:
  /navila/output    (std_msgs/String)      - NaVILA 모델 텍스트 출력

Publishes:
  /cmd_vel          (geometry_msgs/Twist)  - 파싱된 이동 명령

액션 매핑:
  "stop"           → Twist (모두 0)
  "move forward X cm"  → linear.x  = X / 100.0  [m]
  "turn left X degree" → angular.z = +radians(X) [rad]
  "turn right X degree"→ angular.z = -radians(X) [rad]

※ linear.x / angular.z 는 목표 변위(거리·각도)를 담습니다.
  상위 컨트롤러에서 이 값을 목표 변위로 해석하여 모터를 구동하세요.

유효 스냅 값:
  거리: 25, 50, 75 cm 중 가장 가까운 값으로 반올림
  각도: 15, 30, 45 도 중 가장 가까운 값으로 반올림
"""

import math
import re

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


class RegexParshingNode(Node):
    def __init__(self):
        super().__init__("regex_parshing_node")

        # 파라미터
        self.declare_parameter("input_topic", "/navila/output")
        self.declare_parameter("output_topic", "/cmd_vel")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        # 액션 분류 패턴 (navila_trainer.py 기준)
        self._action_patterns = {
            "stop":    re.compile(r"\bstop\b", re.IGNORECASE),
            "forward": re.compile(r"\bis move forward\b", re.IGNORECASE),
            "left":    re.compile(r"\bis turn left\b", re.IGNORECASE),
            "right":   re.compile(r"\bis turn right\b", re.IGNORECASE),
        }

        # 거리·각도 추출 패턴
        self._distance_re   = re.compile(r"move forward (\d+) cm", re.IGNORECASE)
        self._turn_left_re  = re.compile(r"turn left (\d+) degree", re.IGNORECASE)
        self._turn_right_re = re.compile(r"turn right (\d+) degree", re.IGNORECASE)

        # 유효 스냅 값
        self._valid_distances = [25, 50, 75]   # cm
        self._valid_degrees   = [15, 30, 45]   # degree

        # Subscriber / Publisher
        self.sub = self.create_subscription(String, input_topic, self._output_callback, 10)
        self.pub = self.create_publisher(Twist, output_topic, 10)

        self.get_logger().info(
            f"RegexParshingNode 시작 | "
            f"input: {input_topic} | output: {output_topic}"
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _classify_action(self, text: str) -> str:
        """텍스트에서 액션 종류를 분류합니다. 매칭 실패 시 'stop' 반환."""
        for action, pattern in self._action_patterns.items():
            if pattern.search(text):
                return action
        self.get_logger().warn(
            f"알 수 없는 액션 텍스트, STOP으로 처리: {text!r}"
        )
        return "stop"

    def _snap(self, value: int, candidates: list) -> int:
        """value를 candidates 중 가장 가까운 값으로 스냅합니다."""
        if value in candidates:
            return value
        return min(candidates, key=lambda x: abs(x - value))

    def _parse_distance_cm(self, text: str) -> int:
        match = self._distance_re.search(text)
        raw = int(match.group(1)) if match else 25
        return self._snap(raw, self._valid_distances)

    def _parse_degree(self, text: str, direction: str) -> int:
        pattern = self._turn_left_re if direction == "left" else self._turn_right_re
        match = pattern.search(text)
        raw = int(match.group(1)) if match else 15
        return self._snap(raw, self._valid_degrees)

    # ------------------------------------------------------------------
    # 콜백
    # ------------------------------------------------------------------

    def _output_callback(self, msg: String):
        text = msg.data

        action = self._classify_action(text)
        twist = Twist()

        if action == "forward":
            dist_cm = self._parse_distance_cm(text)
            dist_m = dist_cm / 100.0
            twist.linear.x = dist_m

        elif action == "left":
            degree = self._parse_degree(text, "left")
            rad = math.radians(degree)
            twist.angular.z = rad

        elif action == "right":
            degree = self._parse_degree(text, "right")
            rad = math.radians(degree)
            twist.angular.z = -rad

        self.pub.publish(twist)
        self.get_logger().info(
            f"cmd_vel | linear.x={twist.linear.x:.3f}  angular.z={twist.angular.z:.4f}"
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RegexParshingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
