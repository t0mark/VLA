"""
/cmd_vel/chunk (JSON) 로 수신한 액션 청크를 30 Hz 제어 루프로 부드럽게 실행합니다.

Subscribes:
  /cmd_vel/chunk  (std_msgs/String, JSON) - regex_parshing_node 의 다중 액션 출력

Publishes:
  /cmd_vel        (geometry_msgs/Twist)   - EMA 스무딩이 적용된 속도 명령

동작 원리:
  1. 청크 수신 → 각 액션을 (목표 속도, 지속 시간) 으로 변환 후 큐에 적재
  2. 30 Hz 타이머가 큐 헤드의 목표 속도를 향해 EMA 필터로 부드럽게 접근
  3. 경과 시간이 지속 시간을 넘으면 다음 액션으로 전환
  4. 큐가 빌 때까지 반복; 비면 0 으로 수렴

파라미터:
  chunk_topic       (str,   /cmd_vel/chunk) - 입력 토픽
  output_topic      (str,   /cmd_vel)       - 출력 토픽
  control_hz        (float, 30.0)           - 제어 루프 주파수 [Hz]
  max_linear_speed  (float, 0.2)            - 전진 최대 속도 [m/s]
  max_angular_speed (float, 0.5)            - 회전 최대 속도 [rad/s]
  smoothing_alpha   (float, 0.15)           - EMA 계수 (0<α≤1, 작을수록 부드러움)
  stop_duration     (float, 0.5)            - stop 액션 지속 시간 [s]
"""

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

# EMA 시정수 참고 (30 Hz 기준):
#   τ ≈ (1/hz) / alpha  →  alpha=0.15 → τ ≈ 0.22 s
_DEADBAND = 1e-4   # 이 속도 이하는 0 으로 처리해 정지 떨림 방지


class ActionExecutorNode(Node):
    def __init__(self):
        super().__init__("action_executor_node")

        self.declare_parameter("chunk_topic",       "/cmd_vel/chunk")
        self.declare_parameter("output_topic",      "/cmd_vel")
        self.declare_parameter("control_hz",        30.0)
        self.declare_parameter("max_linear_speed",  0.2)
        self.declare_parameter("max_angular_speed", 0.5)
        self.declare_parameter("smoothing_alpha",   0.15)
        self.declare_parameter("stop_duration",     0.5)

        chunk_topic  = self.get_parameter("chunk_topic").value
        output_topic = self.get_parameter("output_topic").value
        hz           = self.get_parameter("control_hz").value

        self._max_lin      = self.get_parameter("max_linear_speed").value
        self._max_ang      = self.get_parameter("max_angular_speed").value
        self._alpha        = self.get_parameter("smoothing_alpha").value
        self._stop_dur     = self.get_parameter("stop_duration").value

        # 액션 큐: 각 원소 = (target_linear_x, target_angular_z, duration_s)
        self._queue: list[tuple[float, float, float]] = []
        self._elapsed      = 0.0   # 현재 액션 경과 시간 [s]
        self._dt           = 1.0 / hz

        # EMA 상태
        self._cur_lin      = 0.0
        self._cur_ang      = 0.0

        self.sub   = self.create_subscription(String, chunk_topic, self._chunk_callback, 10)
        self.pub   = self.create_publisher(Twist, output_topic, 10)
        self.timer = self.create_timer(self._dt, self._control_loop)

        self.get_logger().info(
            f"ActionExecutorNode 시작 | chunk: {chunk_topic} | output: {output_topic} | "
            f"{hz:.0f} Hz | max_lin={self._max_lin} m/s | max_ang={self._max_ang} rad/s | "
            f"alpha={self._alpha}"
        )

    # ------------------------------------------------------------------
    # 청크 수신
    # ------------------------------------------------------------------

    def _action_to_step(self, action: dict) -> tuple[float, float, float]:
        """액션 딕셔너리 → (target_linear_x, target_angular_z, duration_s)."""
        kind = action.get("action", "stop")

        if kind == "forward":
            dist     = abs(action.get("linear_x", 0.0))   # [m]
            duration = dist / self._max_lin if self._max_lin > 0 else 1.0
            return self._max_lin, 0.0, duration

        elif kind == "left":
            rad      = abs(action.get("angular_z", 0.0))  # [rad]
            duration = rad / self._max_ang if self._max_ang > 0 else 1.0
            return 0.0, self._max_ang, duration

        elif kind == "right":
            rad      = abs(action.get("angular_z", 0.0))  # [rad]
            duration = rad / self._max_ang if self._max_ang > 0 else 1.0
            return 0.0, -self._max_ang, duration

        else:  # stop
            return 0.0, 0.0, self._stop_dur

    def _chunk_callback(self, msg: String):
        try:
            actions = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON 파싱 실패: {e}")
            return

        # 새 청크가 오면 즉시 교체 (최신 관측 반영)
        self._queue   = [self._action_to_step(a) for a in actions]
        self._elapsed = 0.0

        preview = " → ".join(
            f"{a.get('action','?')}({a.get('linear_x', a.get('angular_z', 0)):.2f})"
            for a in actions
        )
        self.get_logger().info(f"새 청크 | {len(self._queue)}개: {preview}")

    # ------------------------------------------------------------------
    # 30 Hz 제어 루프
    # ------------------------------------------------------------------

    def _control_loop(self):
        # 목표 속도 결정
        if self._queue:
            target_lin, target_ang, duration = self._queue[0]
            self._elapsed += self._dt

            if self._elapsed >= duration:
                self._queue.pop(0)
                self._elapsed = 0.0
                remaining = len(self._queue)
                self.get_logger().info(f"액션 완료 | 남은 액션: {remaining}개")
        else:
            target_lin, target_ang = 0.0, 0.0

        # EMA 스무딩
        self._cur_lin = self._alpha * target_lin + (1.0 - self._alpha) * self._cur_lin
        self._cur_ang = self._alpha * target_ang + (1.0 - self._alpha) * self._cur_ang

        # 데드밴드: 목표가 0이고 잔류 속도가 미미하면 완전 정지
        if target_lin == 0.0 and abs(self._cur_lin) < _DEADBAND:
            self._cur_lin = 0.0
        if target_ang == 0.0 and abs(self._cur_ang) < _DEADBAND:
            self._cur_ang = 0.0

        twist           = Twist()
        twist.linear.x  = self._cur_lin
        twist.angular.z = self._cur_ang
        self.pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ActionExecutorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
