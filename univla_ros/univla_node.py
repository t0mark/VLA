"""
UniVLA ROS2 Node

Subscribes:
  /camera/color/image_raw   (sensor_msgs/Image)      -- main RGB camera
  /gripper/color/image_raw  (sensor_msgs/Image)      -- optional gripper cam
  /task_instruction         (std_msgs/String)        -- natural language task

Publishes:
  /univla/action_chunk      (std_msgs/Float32MultiArray)
      layout.dim[0]: name="timestep",  size=action_predict_frame (10)
      layout.dim[1]: name="action_dim", size=7
      data: flattened (timestep * action_dim) float32 values

Parameters (declare via ros2 run or yaml):
  model_path         (str)  path to UNIVLA checkpoint dir
  vision_hub         (str)  path to Emu3-VisionVQ dir
  fast_path          (str)  path to FAST tokenizer dir
  norm_stats_path    (str)  path to norm_stats.json  [optional]
  device             (str)  "cuda:0" or "cpu"
  action_predict_frame (int) default 10
  inference_rate_hz  (float) max inference rate, default 2.0
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from cv_bridge import CvBridge

from .univla_inference import UniVLAInference


class UniVLANode(Node):
    def __init__(self):
        super().__init__("univla_node")

        # ── Parameters ──────────────────────────────────────────────────────
        _base = "/home/t0mark/ros_ws/models"
        self.declare_parameter("model_path",          f"{_base}/UNIVLA_LIBERO_IMG_BS192_8K")
        self.declare_parameter("vision_hub",          f"{_base}/Emu3-VisionVQ")
        self.declare_parameter("fast_path",           f"{_base}/fast")
        self.declare_parameter("norm_stats_path",     f"{_base}/norm_stats.json")
        self.declare_parameter("device",              "cuda:0")
        self.declare_parameter("action_predict_frame", 10)
        self.declare_parameter("inference_rate_hz",   2.0)

        model_path           = self.get_parameter("model_path").value
        vision_hub           = self.get_parameter("vision_hub").value
        fast_path            = self.get_parameter("fast_path").value
        norm_stats_path      = self.get_parameter("norm_stats_path").value or None
        device               = self.get_parameter("device").value
        action_predict_frame = self.get_parameter("action_predict_frame").value
        rate_hz              = self.get_parameter("inference_rate_hz").value

        # ── Model ────────────────────────────────────────────────────────────
        self.model = UniVLAInference(
            model_path=model_path,
            vision_hub=vision_hub,
            fast_path=fast_path,
            norm_stats_path=norm_stats_path,
            device=device,
            action_predict_frame=action_predict_frame,
        )

        # ── State ────────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.latest_rgb: np.ndarray = None
        self.latest_gripper: np.ndarray = None
        self.instruction: str = ""
        self._action_predict_frame = action_predict_frame

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Image, "/camera/color/image_raw", self._cb_image, 10
        )
        self.create_subscription(
            Image, "/gripper/color/image_raw", self._cb_gripper, 10
        )
        self.create_subscription(
            String, "/task_instruction", self._cb_instruction, 10
        )

        # ── Publisher ────────────────────────────────────────────────────────
        self.pub_action = self.create_publisher(
            Float32MultiArray, "/univla/action_chunk", 10
        )

        # ── Inference timer ──────────────────────────────────────────────────
        period = 1.0 / rate_hz
        self.create_timer(period, self._inference_cb)

        self.get_logger().info(
            f"UniVLA node ready. inference_rate={rate_hz:.1f} Hz  "
            f"action_horizon={action_predict_frame}"
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _cb_gripper(self, msg: Image):
        self.latest_gripper = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _cb_instruction(self, msg: String):
        if msg.data != self.instruction:
            self.instruction = msg.data
            self.get_logger().info(f"Instruction updated: '{self.instruction}'")

    def _inference_cb(self):
        if self.latest_rgb is None:
            return
        if not self.instruction:
            self.get_logger().warn("No task instruction received yet — skipping inference.")
            return

        try:
            actions = self.model.predict(
                rgb=self.latest_rgb,
                instruction=self.instruction,
                gripper_rgb=self.latest_gripper,
            )  # (action_predict_frame, 7)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        self._publish_actions(actions)

    # ── Publisher helpers ─────────────────────────────────────────────────────

    def _publish_actions(self, actions: np.ndarray):
        """
        Publish action chunk as Float32MultiArray.

        Layout:
          dim[0]: timestep  (size = action_predict_frame)
          dim[1]: action_dim (size = 7)
          data:   row-major flattened float32
        """
        T, D = actions.shape
        msg = Float32MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label="timestep",   size=T, stride=T * D),
                MultiArrayDimension(label="action_dim", size=D, stride=D),
            ],
            data_offset=0,
        )
        msg.data = actions.flatten().astype(np.float32).tolist()
        self.pub_action.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UniVLANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
