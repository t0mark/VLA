"""
NaVILA ROS2 Inference Node

Subscribes:
  /camera/image_raw    (sensor_msgs/Image)  - 카메라 입력
  /navila/instruction  (std_msgs/String)    - 내비게이션 언어 지시

Publishes:
  /navila/command      (std_msgs/String)    - 모델 출력 명령
"""

import sys
import os

# CUDA 메모리 할당기 초기화 전에 설정해야 효과가 있음
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# llava 패키지 경로를 sys.path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from collections import deque

import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class NaViLANode(Node):
    def __init__(self):
        super().__init__("navila_node")

        # 파라미터 선언
        self.declare_parameter("model_path", "")
        self.declare_parameter("model_base", "")
        self.declare_parameter("num_frames", 8)
        self.declare_parameter("inference_hz", 1.0)
        self.declare_parameter("max_new_tokens", 256)
        self.declare_parameter("load_4bit", True)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("instruction_topic", "/navila/instruction")
        self.declare_parameter("command_topic", "/navila/command")

        self.model_path = self.get_parameter("model_path").value
        self.model_base = self.get_parameter("model_base").value or None
        self.num_frames = self.get_parameter("num_frames").value
        self.max_new_tokens = self.get_parameter("max_new_tokens").value
        load_4bit = self.get_parameter("load_4bit").value
        inference_hz = self.get_parameter("inference_hz").value

        image_topic = self.get_parameter("image_topic").value
        instruction_topic = self.get_parameter("instruction_topic").value
        command_topic = self.get_parameter("command_topic").value

        if not self.model_path:
            self.get_logger().error("model_path 파라미터가 설정되지 않았습니다.")
            raise RuntimeError("model_path is required")

        # 프레임 버퍼 및 현재 지시
        self.frame_buffer: deque = deque(maxlen=self.num_frames)
        self.current_instruction: str = ""
        self.bridge = CvBridge()

        # 모델 로딩
        self.get_logger().info(f"모델 로딩 중: {self.model_path} (4-bit: {load_4bit})")
        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path, model_name, self.model_base,
            load_4bit=load_4bit,
            torch_dtype=torch.float16,  # prepare_config_for_eval이 pop하여 소비
        )
        self.model.eval()
        self.get_logger().info("모델 로딩 완료")

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, 10
        )
        self.instruction_sub = self.create_subscription(
            String, instruction_topic, self._instruction_callback, 10
        )

        # Publisher
        self.command_pub = self.create_publisher(String, command_topic, 10)

        # 추론 타이머
        self.timer = self.create_timer(1.0 / inference_hz, self._inference_callback)

        self.get_logger().info(
            f"NaVILA 노드 시작 | image: {image_topic} | "
            f"instruction: {instruction_topic} | command: {command_topic}"
        )

    def _image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_img = PILImage.fromarray(cv_img)
            self.frame_buffer.append(pil_img)
        except Exception as e:
            self.get_logger().warn(f"이미지 변환 실패: {e}")

    def _instruction_callback(self, msg: String):
        self.current_instruction = msg.data
        self.get_logger().info(f"지시 수신: {self.current_instruction}")

    def _inference_callback(self):
        if len(self.frame_buffer) < self.num_frames:
            self.get_logger().debug(
                f"프레임 버퍼 부족: {len(self.frame_buffer)}/{self.num_frames}"
            )
            return

        if not self.current_instruction:
            self.get_logger().debug("지시(instruction)가 없어 추론 스킵")
            return

        images = list(self.frame_buffer)

        # 프롬프트 구성 (run_navigation.py 기준)
        image_token = "<image>\n"
        qs = (
            f"Imagine you are a robot programmed for navigation tasks. "
            f"You have been given a video of historical observations "
            f"{image_token * (self.num_frames - 1)}, and current observation <image>\n. "
            f'Your assigned task is: "{self.current_instruction}" '
            f"Analyze this series of images to decide your next action, which could be "
            f"turning left or right by a specific degree, moving forward a certain distance, "
            f"or stop if the task is completed."
        )

        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        try:
            torch.cuda.empty_cache()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()

            if output.endswith(stop_str):
                output = output[: -len(stop_str)].strip()

            self.get_logger().info(f"명령 출력: {output}")
            self.command_pub.publish(String(data=output))

        except Exception as e:
            self.get_logger().error(f"추론 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = NaViLANode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[ERROR] {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
