"""
NaVILA 모델의 텍스트 출력을 정규식으로 파싱하여 /cmd_vel/chunk 로 전달합니다.

Subscribes:
  /navila/output    (std_msgs/String)             - NaVILA 모델 텍스트 출력

Publishes:
  /cmd_vel/chunk    (std_msgs/String, JSON array) - 파싱된 액션 목록
                                                    → action_executor_node 가 수신

출력 형식 자동 감지:
  단일: "The action is move forward 75 cm."
  다중: "The next 5 actions are: move forward 75 cm, turn left 30 degree, ..."

파라미터:
  input_topic  (str, default=/navila/output) - 모델 출력 토픽
  chunk_topic  (str, default=/cmd_vel/chunk) - 청크 출력 토픽

액션 매핑:
  "stop"               → linear_x=0,    angular_z=0
  "move forward X cm"  → linear_x=X/100 [m],  angular_z=0
  "turn left X degree" → linear_x=0,    angular_z=+radians(X)
  "turn right X degree"→ linear_x=0,    angular_z=-radians(X)

유효 스냅 값:
  거리: 25, 50, 75 cm / 각도: 15, 30, 45 도
"""

import json
import math
import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RegexParshingNode(Node):
    def __init__(self):
        super().__init__("regex_parshing_node")

        self.declare_parameter("input_topic", "/navila/output")
        self.declare_parameter("chunk_topic", "/cmd_vel/chunk")

        input_topic = self.get_parameter("input_topic").value
        chunk_topic = self.get_parameter("chunk_topic").value

        # 다중 액션 형식: "The next N actions are: a1, a2, ..."
        self._multi_action_re = re.compile(
            r"next \d+ actions? (?:are|is):?\s*(.+?)\.?\s*$",
            re.IGNORECASE | re.DOTALL,
        )
        self._action_patterns = {
            "stop":    re.compile(r"\bstop\b",                 re.IGNORECASE),
            "forward": re.compile(r"\b(?:is )?move forward\b", re.IGNORECASE),
            "left":    re.compile(r"\b(?:is )?turn left\b",    re.IGNORECASE),
            "right":   re.compile(r"\b(?:is )?turn right\b",   re.IGNORECASE),
        }
        self._distance_re   = re.compile(r"move forward (\d+) cm",   re.IGNORECASE)
        self._turn_left_re  = re.compile(r"turn left (\d+) degree",  re.IGNORECASE)
        self._turn_right_re = re.compile(r"turn right (\d+) degree", re.IGNORECASE)

        self._valid_distances = [25, 50, 75]
        self._valid_degrees   = [15, 30, 45]

        self.sub = self.create_subscription(String, input_topic, self._output_callback, 10)
        self.pub = self.create_publisher(String, chunk_topic, 10)

        self.get_logger().info(
            f"RegexParshingNode 시작 | input: {input_topic} | chunk: {chunk_topic}"
        )

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def _classify_action(self, text: str) -> str | None:
        """strict: 숫자 포함 완전한 형식이 있어야 인식, 아니면 None 반환."""
        if self._action_patterns["stop"].search(text):
            return "stop"
        if self._distance_re.search(text):
            return "forward"
        if self._turn_left_re.search(text):
            return "left"
        if self._turn_right_re.search(text):
            return "right"
        return None

    def _snap(self, value: int, candidates: list) -> int:
        return value if value in candidates else min(candidates, key=lambda x: abs(x - value))

    def _parse_distance_cm(self, text: str) -> int:
        m = self._distance_re.search(text)
        return self._snap(int(m.group(1)), self._valid_distances) if m else 25

    def _parse_degree(self, text: str, direction: str) -> int:
        pattern = self._turn_left_re if direction == "left" else self._turn_right_re
        m = pattern.search(text)
        return self._snap(int(m.group(1)), self._valid_degrees) if m else 15

    def _text_to_action(self, action_text: str) -> dict | None:
        action = self._classify_action(action_text)
        if action is None:
            return None
        linear_x, angular_z = 0.0, 0.0
        if action == "forward":
            linear_x  =  self._parse_distance_cm(action_text) / 100.0
        elif action == "left":
            angular_z =  math.radians(self._parse_degree(action_text, "left"))
        elif action == "right":
            angular_z = -math.radians(self._parse_degree(action_text, "right"))
        return {"action": action, "linear_x": linear_x, "angular_z": angular_z}

    def _extract_action_texts(self, text: str) -> list:
        """모델 출력 형식을 자동 감지하여 액션 문자열 목록을 반환합니다."""
        m = self._multi_action_re.search(text)
        if m:
            return [s.strip() for s in m.group(1).split(",") if s.strip()]
        return [text]

    # ------------------------------------------------------------------
    # 콜백
    # ------------------------------------------------------------------

    def _output_callback(self, msg: String):
        action_texts = self._extract_action_texts(msg.data)
        payload = [self._text_to_action(t) for t in action_texts]
        payload = [p for p in payload if p is not None]

        if not payload:
            self.get_logger().warn(f"유효한 액션 없음, 스킵: {msg.data[:80]!r}")
            return

        out      = String()
        out.data = json.dumps(payload)
        self.pub.publish(out)

        for p in payload:
            self.get_logger().info(
                f"cmd_vel = [{p['linear_x']:.3f}, 0.000, {p['angular_z']:.4f}]"
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
